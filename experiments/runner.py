"""Run each measurement condition in its own Python process (research.md R2).

Why a process per condition:

* Peak memory never decreases within a process, so a per-condition peak needs a fresh process.
* Running out of memory may not raise a Python exception. On Linux/macOS the kernel kills the
  process with SIGKILL. On Windows (the measuring machine) there is no OOM killer: an allocation
  fails with an allocator error, the process dies with STATUS_NO_MEMORY, or the page file grows and
  the condition slows into the time cap. Only a parent process can record all of these (FR-015).

Parent side: `run_condition`. Child side: ``python -m experiments.runner --child FUNC SPEC RESULT``,
which calls ``FUNC(spec)`` (``"package.module:function"``) and writes its dict to RESULT as JSON.

`load_agent` (T009) is the one place the harness loads Laya: CPU only, fixed thread count,
optional pinned revision. It never edits `laya/` (research.md R1).
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import signal as _signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .results import Status

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Seconds the parent waits beyond `time_cap` before killing the child. The child sees
#: `time_cap` in its spec and is expected to stop by itself and report `partial`; the grace
#: covers model load and writing the result.
DEFAULT_GRACE_SECONDS = 30.0

# Windows NTSTATUS exit codes that mean the process ran out of memory.
_WIN_NO_MEMORY = 0xC0000017          # STATUS_NO_MEMORY
_WIN_COMMITMENT_LIMIT = 0xC000012D   # STATUS_COMMITMENT_LIMIT
_WIN_OOM_CODES = {_WIN_NO_MEMORY, _WIN_COMMITMENT_LIMIT}

_OOM_MESSAGES = ("not enough memory", "defaultcpuallocator", "out of memory",
                 "cannot allocate memory", "can't allocate memory")


# --------------------------------------------------------------------------- memory

def _windows_memory() -> Dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = kernel32.K32GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "K32GetProcessMemoryInfo failed")
    return {
        "peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "peak_commit_bytes": int(counters.PeakPagefileUsage),
        "memory_source": "windows:PeakWorkingSetSize,PeakPagefileUsage",
    }


def _ru_maxrss_to_bytes(value: int, platform: str) -> int:
    """`ru_maxrss` is bytes on macOS and kilobytes on Linux and other POSIX systems."""
    return int(value) if platform == "darwin" else int(value) * 1024


def _posix_memory() -> Dict[str, Any]:
    import resource  # POSIX only; never imported on Windows

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_bytes": _ru_maxrss_to_bytes(usage.ru_maxrss, sys.platform),
        "peak_commit_bytes": None,
        "memory_source": "getrusage:ru_maxrss",
    }


def peak_memory() -> Dict[str, Any]:
    """Peak memory of the current process, in bytes. One function for every platform."""
    if sys.platform == "win32":
        return _windows_memory()
    return _posix_memory()


def physical_ram_bytes() -> Optional[int]:
    """Installed physical memory, or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
            return None
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        return None


def _memory_fields(mem: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    mem = mem or {}
    commit = mem.get("peak_commit_bytes")
    ram = physical_ram_bytes()
    return {
        "peak_rss_bytes": mem.get("peak_rss_bytes"),
        "peak_commit_bytes": commit,
        "memory_source": mem.get("memory_source"),
        "paging_suspected": bool(commit is not None and ram is not None and commit > ram),
    }


# --------------------------------------------------------------------------- failure causes

def is_oom_error(exc_type: str, message: str) -> bool:
    """An exception that means the process ran out of memory."""
    if exc_type == "MemoryError":
        return True
    low = (message or "").lower()
    return any(m in low for m in _OOM_MESSAGES)


def classify_exit(returncode: int, platform: Optional[str] = None) -> Dict[str, Any]:
    """Cause for a child that exited abnormally without writing a result."""
    platform = platform or sys.platform
    if platform == "win32":
        code = returncode & 0xFFFFFFFF
        if code in _WIN_OOM_CODES:
            return {"cause": "oom", "signal": None, "exit_code": code,
                    "reason": "process exited with 0x%08X (out of memory)" % code}
        return {"cause": "crash", "signal": None, "exit_code": code,
                "reason": "process exited with code 0x%08X without a result" % code}
    if returncode < 0:
        sig = -returncode
        try:
            name = _signal.Signals(sig).name
        except ValueError:
            name = "signal %d" % sig
        if sig == getattr(_signal, "SIGKILL", 9):
            return {"cause": "oom", "signal": sig, "exit_code": returncode,
                    "reason": "killed by SIGKILL, most likely the kernel OOM killer"}
        return {"cause": "signal", "signal": sig, "exit_code": returncode,
                "reason": "killed by %s" % name}
    return {"cause": "crash", "signal": None, "exit_code": returncode,
            "reason": "process exited with code %d without a result" % returncode}


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        data = path.read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", "replace").strip()


# --------------------------------------------------------------------------- parent side

def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    root = str(REPO_ROOT)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_condition(module_function: str, spec: Dict[str, Any], time_cap: Optional[float] = None,
                  grace: float = DEFAULT_GRACE_SECONDS) -> Dict[str, Any]:
    """Run ``module_function(spec)`` in a fresh Python child and return one result item.

    The child sees ``spec["time_cap"]`` and should stop by itself (status ``partial``). The parent
    kills it at ``time_cap + grace`` and records ``failed`` with cause ``timeout``. Every outcome
    is returned as an item; nothing raises for a failed condition, and the parent is unaffected.
    """
    spec = dict(spec)
    if time_cap is not None:
        spec["time_cap"] = float(time_cap)
    work = Path(tempfile.mkdtemp(prefix="laya-cond-"))
    spec_path, result_path = work / "spec.json", work / "result.json"
    out_path, err_path = work / "stdout.txt", work / "stderr.txt"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [sys.executable, "-m", "experiments.runner", "--child", module_function,
           str(spec_path), str(result_path)]
    hard_limit = None if time_cap is None else float(time_cap) + float(grace)
    started = time.perf_counter()
    base = {"runner": {"function": module_function}}
    try:
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=_child_env(), stdout=out, stderr=err)
            try:
                returncode = proc.wait(timeout=hard_limit)
            except subprocess.TimeoutExpired:
                proc.kill()  # TerminateProcess on Windows, SIGKILL elsewhere
                proc.wait()
                elapsed = time.perf_counter() - started
                item = dict(base)
                item.update({
                    "status": Status.FAILED.value, "cause": "timeout", "signal": None,
                    "exit_code": proc.returncode,
                    "reason": "no result after %.1f s (time cap %.1f s + %.1f s grace); "
                              "process terminated" % (elapsed, time_cap, grace),
                })
                item.update(_memory_fields(None))
                item["runner"]["elapsed_s"] = elapsed
                return item
        elapsed = time.perf_counter() - started
        base["runner"]["elapsed_s"] = elapsed
        result = None
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
        if result is not None and result.get("ok"):
            item = dict(base)
            payload = result.get("payload") or {}
            item.update(payload)
            item.setdefault("status", Status.MEASURED.value)
            item.update(_memory_fields(result.get("memory")))
            item["exit_code"] = returncode
            return item
        if result is not None:  # the child caught an exception and reported it
            exc_type, message = result.get("exception_type", ""), result.get("message", "")
            item = dict(base)
            item.update({
                "status": Status.FAILED.value,
                "cause": "oom" if is_oom_error(exc_type, message) else "exception",
                "signal": None, "exit_code": returncode,
                "exception_type": exc_type,
                "reason": "%s: %s" % (exc_type, message) if message else exc_type,
                "traceback": result.get("traceback"),
            })
            item.update(_memory_fields(result.get("memory")))
            return item
        item = dict(base)
        item.update({"status": Status.FAILED.value})
        item.update(classify_exit(returncode))
        stderr = _tail(err_path)
        if stderr:
            item["stderr_tail"] = stderr
        item.update(_memory_fields(None))
        return item
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- child side

def resolve_function(module_function: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    if ":" not in module_function:
        raise ValueError("expected 'package.module:function', got %r" % module_function)
    mod_name, func_name = module_function.split(":", 1)
    return getattr(importlib.import_module(mod_name), func_name)


def _write_result(path: str, body: Dict[str, Any]) -> None:
    from .results import _jsonable  # same NaN/enum handling as result files
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(body), f)
    os.replace(tmp, path)


def child_main(module_function: str, spec_path: str, result_path: str) -> int:
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    try:
        payload = resolve_function(module_function)(spec)
    except BaseException as exc:  # includes MemoryError; SystemExit/KeyboardInterrupt re-raised below
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        mem = None
        with contextlib.suppress(Exception):
            mem = peak_memory()
        _write_result(result_path, {
            "ok": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "memory": mem,
        })
        return 1
    if not isinstance(payload, dict):
        _write_result(result_path, {"ok": False, "exception_type": "TypeError",
                                    "message": "%s returned %s, expected dict"
                                               % (module_function, type(payload).__name__),
                                    "memory": peak_memory()})
        return 1
    _write_result(result_path, {"ok": True, "payload": payload, "memory": peak_memory()})
    return 0


# --------------------------------------------------------------------------- model loading (T009)

_FALLBACK_MARKERS = ("falling back to cpu", "running on cpu", "retrying this request on cpu")


class ModelLoadError(RuntimeError):
    """The model could not be loaded: a tool-level failure, not a measurement result."""


def resolve_model_revision(model: str, revision: Optional[str]) -> Tuple[Optional[str], str]:
    """(revision to request, how it was chosen). ``reviewed`` uses `laya.revisions.PINNED_REVISIONS`."""
    if os.path.exists(model):
        return None, "local"
    if revision is None or revision == "":
        return None, "hub_default"
    from laya.revisions import PINNED_REVISIONS, REVIEWED
    if revision == REVIEWED:
        if model not in PINNED_REVISIONS:
            raise ValueError("revision 'reviewed' requested but %r has no entry in "
                             "laya.revisions.PINNED_REVISIONS" % model)
        return PINNED_REVISIONS[model], "reviewed"
    return revision, "explicit"


def load_agent(model: str, revision: Optional[str] = None, threads: Optional[int] = None,
               compile: bool = False, mha_fastpath: bool = True):
    """Load `laya.Agent` on CPU and return ``(agent, info)``.

    Raises if the effective device is not CPU. `info` records load time, the thread count, the
    requested and resolved revision, and any device fallback seen during load.
    """
    import torch
    import laya

    if threads is not None:
        threads = int(threads)
        if threads < 1:
            raise ValueError("threads must be >= 1, got %d" % threads)
        torch.set_num_threads(threads)
    # PyTorch's nn.TransformerEncoderLayer fast path (research.md R17). On (the default) is native Laya;
    # off is a labelled variant that routes the decision head through Laya's own SDPA attention.
    torch.backends.mha.set_fastpath_enabled(bool(mha_fastpath))
    captured = io.StringIO()
    started = time.perf_counter()
    try:
        rev, rev_source = resolve_model_revision(model, revision)
        with contextlib.redirect_stdout(captured):
            agent = laya.Agent(model, device="cpu", revision=rev, compile=compile)
    except Exception as exc:
        raise ModelLoadError("%s: %s" % (type(exc).__name__, exc)) from exc
    load_seconds = time.perf_counter() - started
    printed = captured.getvalue()
    if printed:
        sys.stdout.write(printed)  # keep the library's own messages visible
    if agent.device.type != "cpu":
        raise RuntimeError("effective device is %s, not cpu; refusing to measure" % agent.device)
    fallback_messages = [line.strip() for line in printed.splitlines()
                         if any(m in line.lower() for m in _FALLBACK_MARKERS)]
    is_local = rev_source == "local"
    info = {
        "model": model,
        "revision_requested": revision,
        "revision_source": rev_source,
        "revision": "local" if is_local else (agent.revision or rev),
        "load_seconds": load_seconds,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(agent.device),
        "dtype": str(agent.dtype).replace("torch.", ""),
        "amp_enabled": bool(agent.amp_enabled),
        "compile": bool(compile),
        "mha_fastpath": bool(torch.backends.mha.get_fastpath_enabled()),
        "cpu_fallback_count": int(getattr(agent, "cpu_fallback_count", 0) or 0),
        "last_fallback_reason": getattr(agent, "last_fallback_reason", None),
        "fallback_messages": fallback_messages,
    }
    return agent, info


def default_threads() -> int:
    """PyTorch's default intra-op thread count in a fresh process (the physical core count)."""
    import torch
    return int(torch.get_num_threads())


def _main(argv) -> int:
    if len(argv) == 4 and argv[0] == "--child":
        return child_main(argv[1], argv[2], argv[3])
    sys.stderr.write("usage: python -m experiments.runner --child MODULE:FUNC SPEC.json RESULT.json\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

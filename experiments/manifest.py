"""Run manifest: everything needed to reproduce and compare a run (T015, data-model "Run Manifest").

Constitution V: model and code revisions, software versions, hardware, threads, precision, cache
policy, seeds and fallback events. Anything that cannot be read is recorded as ``"unknown"`` with
a note, never guessed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import REPO_ROOT, physical_ram_bytes

UNKNOWN = "unknown"

#: How seeds are used by timed runs; recorded so a run can be repeated exactly.
SEED_RULE = "repeat r of a condition uses seed base_seed + r (a fresh document per repeat)"

CACHE_POLICY = {
    "question_token_reuse": "on",
    "question_token_reuse_scope": "per predict call (laya.common._reuse_question_tokens); nothing kept between calls",
    "document_caches": "none",
    "document_kv_cache": "none",
}


class ManifestError(ValueError):
    """The manifest would be incomplete in a way the contract forbids."""


# --------------------------------------------------------------------------- code

def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def code_info(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Git SHA, dirty flag, and whether `laya/` matches the recorded SHA (SC-005)."""
    sha = _git(["rev-parse", "HEAD"], repo_root)
    if sha is None:
        return {"git_sha": UNKNOWN, "git_dirty": None, "laya_diff_empty": None,
                "note": "git unavailable or not a repository"}
    status = _git(["status", "--porcelain", "--untracked-files=all"], repo_root) or ""
    laya_status = _git(["status", "--porcelain", "--untracked-files=all", "--", "laya/"], repo_root)
    return {
        "git_sha": sha.strip(),
        "git_dirty": bool(status.strip()),
        "laya_diff_empty": laya_status is not None and not laya_status.strip(),
        "dirty_paths": [line[3:] for line in status.splitlines()][:50],
    }


# --------------------------------------------------------------------------- software

def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return UNKNOWN


def software_info() -> Dict[str, Any]:
    import torch
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "tokenizers": _version("tokenizers"),
        "numpy": _version("numpy"),
        "safetensors": _version("safetensors"),
        "huggingface_hub": _version("huggingface_hub"),
        "laya_sparse": _version("laya-sparse"),
    }


# --------------------------------------------------------------------------- hardware

def _windows_cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    notes = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["cpu_model"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError as e:
        notes.append("cpu name: %s" % e)
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel32.GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        RELATION_PROCESSOR_CORE = 0
        size = wintypes.DWORD(0)
        fn(RELATION_PROCESSOR_CORE, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not fn(RELATION_PROCESSOR_CORE, buf, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx failed")
        raw, offset, classes = buf.raw, 0, []
        while offset < size.value:
            entry_size = int.from_bytes(raw[offset + 4:offset + 8], "little")
            classes.append(raw[offset + 9])  # PROCESSOR_RELATIONSHIP.EfficiencyClass
            offset += entry_size
        info["physical_cores"] = len(classes)
        info["core_efficiency_classes"] = sorted(set(classes))
        info["hybrid_cores"] = len(set(classes)) > 1
    except (OSError, AttributeError, ValueError) as e:
        notes.append("core layout: %s" % e)
    try:
        import ctypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        st = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            info["on_ac_power"] = {0: False, 1: True}.get(st.ACLineStatus, UNKNOWN)
    except (OSError, AttributeError) as e:
        notes.append("power status: %s" % e)
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=15)
        m = re.search(r"\(([^)]+)\)\s*$", out.stdout.strip())
        info["power_scheme"] = m.group(1) if m else (out.stdout.strip() or UNKNOWN)
        guid = re.search(r"([0-9a-fA-F-]{36})", out.stdout)
        if guid:
            info["power_scheme_guid"] = guid.group(1)
    except (OSError, subprocess.SubprocessError) as e:
        notes.append("power scheme: %s" % e)
    if notes:
        info["notes"] = notes
    return info


def _linux_cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        text = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            info["cpu_model"] = m.group(1).strip()
        cores = set()
        phys = core = None
        for line in text.splitlines() + [""]:
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip():
                if core is not None:
                    cores.add((phys, core))
                phys = core = None
        if cores:
            info["physical_cores"] = len(cores)
    except OSError as e:
        info["notes"] = ["/proc/cpuinfo: %s" % e]
    # Intel hybrid parts expose separate PMUs for P- and E-cores.
    if os.path.isdir("/sys/devices/cpu_core") and os.path.isdir("/sys/devices/cpu_atom"):
        info["hybrid_cores"] = True
    return info


def _mac_cpu() -> Dict[str, Any]:
    def sysctl(name):
        try:
            out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    info: Dict[str, Any] = {}
    brand = sysctl("machdep.cpu.brand_string")
    if brand:
        info["cpu_model"] = brand
    phys = sysctl("hw.physicalcpu")
    if phys and phys.isdigit():
        info["physical_cores"] = int(phys)
    levels = sysctl("hw.nperflevels")
    if levels and levels.isdigit():
        info["hybrid_cores"] = int(levels) > 1
    return info


def hardware_info() -> Dict[str, Any]:
    if sys.platform == "win32":
        info = _windows_cpu()
    elif sys.platform == "darwin":
        info = _mac_cpu()
    else:
        info = _linux_cpu()
    info.setdefault("cpu_model", platform.processor() or UNKNOWN)
    info.setdefault("physical_cores", UNKNOWN)
    info.setdefault("hybrid_cores", UNKNOWN)
    info["logical_cores"] = os.cpu_count() or UNKNOWN
    info["architecture"] = platform.machine() or UNKNOWN
    info["ram_bytes"] = physical_ram_bytes() or UNKNOWN
    info["os"] = platform.platform()
    return info


# --------------------------------------------------------------------------- runtime

def torch_build_info() -> Dict[str, Any]:
    import torch
    cfg = torch.__config__.show()
    keep = [ln.strip() for ln in cfg.splitlines()
            if re.search(r"blas|lapack|mkl|openmp|mkldnn|onednn|cpu capability|build settings", ln, re.I)]
    par = torch.__config__.parallel_info()
    try:
        capability = torch.backends.cpu.get_cpu_capability()
    except AttributeError:
        capability = UNKNOWN
    return {
        "cpu_capability": capability,
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "openmp_available": bool(getattr(torch.backends, "openmp", None) and torch.backends.openmp.is_available()),
        "build_config": keep[:40],
        "parallel_info": [ln.strip() for ln in par.splitlines() if ln.strip()][:20],
    }


def config_sha256(agent) -> str:
    """SHA-256 of the agent config and encoder config as canonical JSON."""
    body = {"agent_cfg": agent.cfg, "encoder_config": agent.model.encoder.config.to_dict()}
    text = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(agent, load_info: Dict[str, Any], *, run_id: str, seed: int = 0,
                   threads_source: str = "default", requested_device: str = "cpu",
                   repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Assemble the manifest. Raises `ManifestError` if a Hub load has no resolved revision."""
    source = load_info.get("revision_source")
    revision = load_info.get("revision")
    if source != "local" and not revision:
        raise ManifestError("model %r was loaded from the Hub but no commit revision was resolved; "
                            "a Hub load must record its revision" % load_info.get("model"))
    hardware = hardware_info()
    fallbacks = []
    if load_info.get("cpu_fallback_count"):
        fallbacks.append({"event": "cpu_fallback", "count": load_info["cpu_fallback_count"],
                          "reason": load_info.get("last_fallback_reason")})
    for msg in load_info.get("fallback_messages") or []:
        fallbacks.append({"event": "device_fallback_message", "reason": msg})
    effective = str(getattr(agent, "device", load_info.get("device")))
    non_comparable = []
    if requested_device != "cpu":
        non_comparable.append("requested device %r is not cpu" % requested_device)
    if effective != "cpu":
        non_comparable.append("effective device %r is not cpu" % effective)
    return {
        "run_id": run_id,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "non_comparable": bool(non_comparable),
        "non_comparable_reasons": non_comparable,
        "fixture_model": bool(agent.cfg.get("laya_sparse_fixture", False)),
        "code": code_info(repo_root),
        "model": {
            "id": load_info.get("model"),
            "revision": revision,
            "revision_requested": load_info.get("revision_requested"),
            "revision_source": source,
            "config_sha256": config_sha256(agent),
            "load_seconds": load_info.get("load_seconds"),
        },
        "software": software_info(),
        "hardware": hardware,
        "runtime": {
            "intra_op_threads": load_info.get("threads"),
            "inter_op_threads": load_info.get("interop_threads"),
            "threads_source": threads_source,
            "hybrid_cores": hardware.get("hybrid_cores", UNKNOWN),
            "torch_build": torch_build_info(),
            "laya_cpu_amp": os.environ.get("LAYA_CPU_AMP") or None,
            "dtype": load_info.get("dtype"),
            "amp_enabled": load_info.get("amp_enabled"),
            "compile": load_info.get("compile", False),
            "concurrency": "one measurement process at a time; other programs on the machine are "
                           "not controlled by the tool (close them before measuring)",
        },
        "device": {"requested": requested_device, "effective": effective, "fallbacks": fallbacks},
        "cache_policy": dict(CACHE_POLICY),
        "seeds": {"base_seed": seed, "rule": SEED_RULE},
    }

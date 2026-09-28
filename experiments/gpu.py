"""GPU reference check (research.md R15; FR-016; contracts/results.md rule 7).

One narrow question: is the historical ~33 ms at 512 tokens a GPU figure? This times native Laya
on CUDA at a few lengths with Laya's own GPU defaults (autocast dtype chosen by the checkpoint and
device) and writes ``gpu_reference.json``. It is never part of the CPU results: the report may cite
it only in the note about the reference's hardware, never in rankings, curves or the cost floor.

Run it from a separate environment with a CUDA build of PyTorch (`experiments/setup_gpu.ps1`),
after the CPU measurements, so the two never compete for the machine.
"""
from __future__ import annotations

import contextlib
import io
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

from .results import Status, summarize


class GpuUnavailable(RuntimeError):
    pass


def _nvidia_smi() -> Dict[str, Any]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}
    except (OSError, subprocess.SubprocessError, IndexError):
        return {"note": "nvidia-smi not available"}


def gpu_environment() -> Dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise GpuUnavailable("CUDA is not available in this environment (torch %s, cuda build %s). "
                             "Use the .venv-gpu environment from experiments/setup_gpu.ps1."
                             % (torch.__version__, torch.version.cuda))
    props = torch.cuda.get_device_properties(0)
    return {
        "hardware_class": "gpu",
        "not_target_cpu": True,
        "device_name": props.name,
        "compute_capability": "%d.%d" % (props.major, props.minor),
        "memory_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "nvidia_smi": _nvidia_smi(),
    }


def measure_gpu(model: str, revision: Optional[str], lengths: List[int], repeats: int = 30,
                warmup: int = 5, seed: int = 0) -> Dict[str, Any]:
    import torch
    import laya
    from .inputs import build_request
    from .runner import resolve_model_revision
    from .tokens import position_limit

    env = gpu_environment()
    rev, rev_source = resolve_model_revision(model, revision)
    captured = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        agent = laya.Agent(model, device="cuda", revision=rev)
    load_seconds = time.perf_counter() - t0
    if agent.device.type != "cuda":
        raise GpuUnavailable("Laya fell back to %s: %s" % (agent.device, captured.getvalue().strip()))
    limit = position_limit(agent)
    items = []
    for L in lengths:
        cond = {"total_tokens": L, "questions": 1, "options_per_question": 2, "batch_size": 1}
        if L > limit:
            items.append({"condition": cond, "status": Status.UNSUPPORTED.value,
                          "reason": "%d exceeds positional capacity %d" % (L, limit)})
            continue
        torch.cuda.reset_peak_memory_stats()
        for w in range(warmup):
            st, q, _ = build_request(agent, L, seed + 1_000_000 + w)
            agent.predict(st[0], q, max_len=L)
        samples = []
        acc0 = None
        for r in range(repeats):
            st, q, acc = build_request(agent, L, seed + r)
            acc0 = acc0 or acc
            torch.cuda.synchronize()
            t = time.perf_counter()
            agent.predict(st[0], q, max_len=L)  # returns numpy, which already waits for the GPU
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t) * 1000.0)
        items.append({"condition": cond, "status": Status.MEASURED.value, "kind": "single",
                      "timings_ms": summarize(samples), "samples_ms": samples,
                      "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                      "token_accounting": acc0})
    return {
        "label": "GPU reference only; not a target-CPU result (FR-016)",
        "environment": env,
        "model": {"id": model, "revision": "local" if rev_source == "local" else (agent.revision or rev),
                  "revision_source": rev_source, "load_seconds": load_seconds},
        "precision": {"dtype": str(agent.dtype).replace("torch.", ""), "amp_enabled": bool(agent.amp_enabled),
                      "note": "Laya's CUDA default: autocast in the checkpoint's amp dtype on compute "
                              "capability 8 or newer"},
        "items": items,
    }

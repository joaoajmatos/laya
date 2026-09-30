"""Latency of the baseline conditions on a fixed sample, CPU (primary) and GPU (labeled) (T045, T046).

Specs: specs/002-decision-benchmark-baselines (research.md R11, R15; FR-023 to FR-025).

* One document and one question per call, on the fixed latency sample (12 dev items, three per
  workflow, ``distractor@mid``). The timed call is exactly ``runner.run``: tokenization, windowing or
  retrieval, encoding, scoring and postprocessing, and nothing else. No document representation is
  reused (constitution VI). Accounting for tokens seen happens after the timed loop.
* The clean-path guard of Phase 1 (`timing.assert_clean`) runs before every CPU call.
* GPU numbers go to ``gpu_latency/`` and are never substituted for CPU numbers. The int8 and fast-path
  variants are CPU-only and are recorded ``unsupported`` on GPU.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import data, evalrun, variants as V
from .results import Refusal, Status, summarize, write_json

SAMPLE_VARIANT = "distractor@mid"
WARMUP_DEFAULT = 1


def default_repeats(length: int) -> int:
    """Passes over the 12-item sample: 3 up to 1,024 tokens, 2 at 2,048, 1 above (12 to 36 samples)."""
    return 3 if length <= 1024 else (2 if length <= 2048 else 1)


def latency_items(fid: str, splits: Dict[str, Any], length: int, root: Optional[Path]) -> List[Dict[str, Any]]:
    """The fixed sample's items at `length`: (case, question) pairs of ``latency_sample``, `distractor@mid`."""
    wanted = {(e["case_id"], e["question_id"]) for e in splits["latency_sample"]}
    out = []
    for it in evalrun._iter_split_items(fid, "dev", root):
        if it["length"] == length and it["variant"] == SAMPLE_VARIANT and (it["case_id"], it["question_id"]) in wanted:
            out.append(it)
    return sorted(out, key=lambda i: i["item_id"])


def _load_gpu_agent(model: str, revision: Optional[str]):
    import contextlib
    import io
    import laya
    import torch
    from .gpu import GpuUnavailable
    from .runner import resolve_model_revision
    if not torch.cuda.is_available():
        raise GpuUnavailable("CUDA is not available in this environment; run from .venv-gpu (experiments/setup_gpu.ps1)")
    rev, _ = resolve_model_revision(model, revision)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = laya.Agent(model, device="cuda", revision=rev)
    if agent.device.type != "cuda":
        raise GpuUnavailable("Laya fell back to %s" % agent.device)
    return agent


def latency_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Child-side entry point: time one (condition, length) pair on the fixed sample."""
    from . import timing
    from .baselines import build_runner
    from .tokens import position_limit

    cond = spec["condition"]
    device = cond["device"]
    reason = V.unsupported_reason(cond["variant"], device)
    if reason:
        return {"status": Status.UNSUPPORTED.value, "reason": reason, "condition": cond, "device": device}
    root = Path(spec["data_root"]) if spec.get("data_root") else None
    splits = data.read_data_json("splits.json", root)
    sample = latency_items(spec["families_id"], splits, cond["length"], root)
    runnable = [it for it in sample if it["status"] == "ok"]
    if not runnable:
        why = sorted({it.get("reason") or "not runnable" for it in sample}) or ["no latency-sample items at this length"]
        return {"status": Status.UNSUPPORTED.value, "condition": cond, "device": device,
                "reason": "no runnable latency-sample item: %s" % "; ".join(why)}

    started = time.perf_counter()
    if device == "gpu":
        import torch
        agent = _load_gpu_agent(spec["model"], spec.get("revision"))
        info = {"mha_fastpath": None, "threads": None, "load_seconds": None}
        torch.cuda.reset_peak_memory_stats()
    else:
        agent, info, _ = V.load_for_variant(spec["model"], spec.get("revision"), spec.get("threads"), cond["variant"])
    runner = build_runner(cond["name"], cond["params"])
    if hasattr(runner, "use_cache"):
        runner.use_cache = False
    warmup = int(spec.get("warmup", WARMUP_DEFAULT))
    reps_spec = spec.get("repeats")
    repeats = default_repeats(cond["length"]) if reps_spec in (None, "auto") else int(reps_spec)
    cap = spec.get("time_cap")

    def sync() -> None:
        if device == "gpu":
            import torch
            torch.cuda.synchronize()

    for w in range(warmup):                               # untimed, on a sample item (kernels, allocator)
        runner.run(agent, runnable[w % len(runnable)])
    samples: List[float] = []
    metas: Dict[str, Dict[str, Any]] = {}
    clean = None
    last, capped = 0.0, False
    for _ in range(repeats):
        for it in runnable:
            if cap is not None and samples and (time.perf_counter() - started) + last > float(cap):
                capped = True
                break
            if device == "cpu":
                clean = timing.assert_clean(agent)
            sync()
            t0 = time.perf_counter()
            meta = runner.run(agent, it)
            sync()
            last = time.perf_counter() - t0
            samples.append(last * 1000.0)
            metas[it["item_id"]] = meta
        if capped:
            break
    seen = []
    for it in runnable:                                   # bookkeeping after the timed loop
        if it["item_id"] in metas:
            seen.append(runner.annotate(agent, it, metas[it["item_id"]]))
    half = len(samples) // 2
    rec = {
        "condition": cond, "device": device, "kind": "single", "call": "predict",
        "sample": [it["item_id"] for it in runnable], "sample_unsupported": [it["item_id"] for it in sample if it["status"] != "ok"],
        "repeats": {"requested": repeats, "completed": len(samples), "warmup": warmup},
        "samples_ms": samples, "timings_ms": summarize(samples) if samples else None,
        "drift_within_condition": ({"first_half_p50_ms": summarize(samples[:half])["p50"],
                                    "second_half_p50_ms": summarize(samples[half:])["p50"]} if half >= 2 else None),
        "clean_path": clean, "position_limit": position_limit(agent),
        "configured_max_len": int(agent.cfg.get("max_len", 512)),
        "tokens_seen": {"min": min((s["tokens_seen"] for s in seen), default=None),
                        "max": max((s["tokens_seen"] for s in seen), default=None)},
        "beyond_configured_max_len": any(s.get("beyond_configured_max_len") for s in seen),
        "deployable": runner.deployable, "variant": cond["variant"],
        "mha_fastpath": info.get("mha_fastpath"), "load_seconds": info.get("load_seconds"),
        "fallbacks": {"cpu_fallback_count": int(getattr(agent, "cpu_fallback_count", 0) or 0),
                      "last_fallback_reason": getattr(agent, "last_fallback_reason", None),
                      "device": str(agent.device), "dtype": str(agent.dtype).replace("torch.", ""),
                      "amp_enabled": bool(agent.amp_enabled)},
        "fixture_model": bool(agent.cfg.get("laya_sparse_fixture", False)),
    }
    if device == "gpu":
        import torch
        rec["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
        rec["label"] = "GPU latency only; never substituted for a CPU result (FR-024)"
    if capped or len(samples) < repeats * len(runnable):
        rec.update(status=Status.PARTIAL.value,
                   reason="time cap %.0f s reached after %d of %d timed calls" % (float(cap), len(samples), repeats * len(runnable)))
    else:
        rec["status"] = Status.MEASURED.value
    return rec


def latency_dir(run_path: Path, device: str) -> Path:
    """``latency/`` for CPU, ``gpu_latency/`` for GPU: the two are never mixed (contracts/results.md)."""
    return Path(run_path) / ("gpu_latency" if device == "gpu" else "latency")


def run_latency(run_path: Path, conditions: Sequence[Dict[str, Any]], model: str, families_id: str,
                revision: Optional[str] = "reviewed", threads: Optional[int] = None, data_root: Optional[Path] = None,
                repeats: Any = None, warmup: int = WARMUP_DEFAULT, time_cap: Optional[float] = None,
                require_cuda: bool = True, runner_fn: Optional[Callable] = None,
                log: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Time every condition in its own subprocess and write one file each. Never substitutes a device."""
    from .runner import run_condition

    run_fn = runner_fn or run_condition
    say = log or (lambda m: None)
    out: List[Dict[str, Any]] = []
    for cond in conditions:
        if cond["device"] == "gpu" and require_cuda and not V.unsupported_reason(cond["variant"], "gpu"):
            import torch
            if not torch.cuda.is_available():
                raise Refusal("device_mismatch", "the GPU latency command needs CUDA; run it from the .venv-gpu "
                                                 "environment (experiments/setup_gpu.ps1). CPU results are not substituted (FR-024)")
        spec = {"condition": cond, "model": model, "revision": revision, "threads": threads, "families_id": families_id,
                "data_root": str(data_root) if data_root else None, "repeats": repeats, "warmup": warmup,
                "time_cap": time_cap}
        item = run_fn("experiments.latency:latency_condition", spec, time_cap=None if time_cap is None else time_cap + 60)
        item.setdefault("condition", cond)
        item.setdefault("device", cond["device"])
        write_json(latency_dir(run_path, cond["device"]), "%s.json" % cond["condition_id"], item)
        say("%s: %s %s" % (cond["condition_id"], item.get("status"),
                           ("p50 %.0f ms" % item["timings_ms"]["p50"]) if item.get("timings_ms") else item.get("reason", "")))
        out.append(item)
    return out

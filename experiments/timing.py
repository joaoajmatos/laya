"""Clean end-to-end latency (T023, T029; research.md R5, R6; FR-009 to FR-015).

`measure_condition(spec)` runs inside a child process started by `runner.run_condition`. It loads
the agent, discards warmup requests, then times each repeat of the public call, from tokenization
through postprocessing, with `time.perf_counter` and nothing else attached:

* ``single`` and ``multi_question``: `agent.predict` (one document).
* ``batch``: `agent.predict_batch` (several documents in one forward pass).

Every repeat gets a new seed, so a new document (fresh-document workload). Requests are built
before the timer starts; building text is not part of Laya's path.

Before every timed call `assert_clean` checks that no stage wrapper, hook or profiler is active
(FR-013). Profiling lives in `profile.py` and never runs here.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .results import Status, summarize

#: Methods `profile.stage_wrappers` replaces; they must not be instance attributes when timing.
WRAPPED_METHODS = ("_encode_state", "_forward", "_decode_answers")

WARMUP_DEFAULT = 3
#: Seed offset for warmup documents, so they never coincide with a timed document.
WARMUP_SEED_OFFSET = 1_000_000


class CleanPathError(RuntimeError):
    """Timing was attempted with instrumentation attached."""


def default_repeats(total_tokens: int) -> int:
    """research.md R6: 30 at <= 512 tokens, 20 at 1,024-2,048, 10 at 4,096 and above."""
    if total_tokens <= 512:
        return 30
    if total_tokens <= 2048:
        return 20
    return 10


def condition_kind(n_questions: int, batch_size: int) -> str:
    if batch_size > 1:
        return "batch"
    return "multi_question" if n_questions > 1 else "single"


def assert_clean(agent) -> Dict[str, Any]:
    """Raise `CleanPathError` if anything would distort a clean timing."""
    import torch
    import laya.agent as agent_mod
    from laya.common import collate_items

    overrides = [m for m in WRAPPED_METHODS if m in vars(agent)]
    patched_collate = agent_mod.collate_items is not collate_items
    hooks = len(getattr(agent, "hooks", None) or [])
    profiler_on = bool(torch.autograd._profiler_enabled())
    problems = []
    if overrides:
        problems.append("instance methods wrapped: %s" % overrides)
    if patched_collate:
        problems.append("laya.agent.collate_items is patched")
    if hooks:
        problems.append("%d prediction hooks installed" % hooks)
    if profiler_on:
        problems.append("torch profiler is active")
    module_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in agent.model.modules())
    if module_hooks:
        problems.append("%d module forward hooks installed" % module_hooks)
    if problems:
        raise CleanPathError("clean timing refused: " + "; ".join(problems))
    return {"instance_overrides": [], "collate_patched": False, "prediction_hooks": 0,
            "module_hooks": 0, "profiler_active": False}


def _call(agent, kind: str, states, questions, max_len: int):
    if kind == "batch":
        return agent.predict_batch(states, questions, max_len=max_len)
    return agent.predict(states[0], questions, max_len=max_len)


def _spec_ints(spec: Dict[str, Any]) -> Dict[str, int]:
    total = int(spec["total_tokens"])
    reps = spec.get("repeats")
    return {
        "total": total,
        "nq": int(spec.get("questions", 1)),
        "opts": int(spec.get("options", 2)),
        "batch": int(spec.get("batch_size", 1)),
        "repeats": default_repeats(total) if reps in (None, "auto") else int(reps),
        "warmup": int(spec.get("warmup", WARMUP_DEFAULT)),
        "seed": int(spec.get("seed", 0)),
    }


def base_record(spec: Dict[str, Any], agent) -> Dict[str, Any]:
    from .tokens import position_limit
    s = _spec_ints(spec)
    kind = condition_kind(s["nq"], s["batch"])
    limit = position_limit(agent)
    configured = int(agent.cfg.get("max_len", 512))
    return {
        "condition": {"total_tokens": s["total"], "questions": s["nq"],
                      "options_per_question": s["opts"], "batch_size": s["batch"]},
        "kind": kind,
        "call": "predict_batch" if kind == "batch" else "predict",
        "position_limit": limit,
        "configured_max_len": configured,
        "beyond_configured_max_len": configured < s["total"] <= limit,
        "fixture_model": bool(agent.cfg.get("laya_sparse_fixture", False)),
    }


def measure_in_process(agent, spec: Dict[str, Any], load_info: Optional[Dict[str, Any]] = None,
                       started: Optional[float] = None) -> Dict[str, Any]:
    """Everything `measure_condition` does after loading. Also used directly by tests."""
    from .inputs import build_request

    started = time.perf_counter() if started is None else started
    s = _spec_ints(spec)
    rec = base_record(spec, agent)
    rec["load_seconds"] = (load_info or {}).get("load_seconds")
    rec["compile_seconds"] = None
    rec["threads"] = (load_info or {}).get("threads")
    rec["repeats"] = {"requested": s["repeats"], "completed": 0, "warmup": s["warmup"]}
    if s["total"] > rec["position_limit"]:
        rec.update(status=Status.UNSUPPORTED.value,
                   reason="%d tokens exceeds positional capacity %d; no model call made"
                          % (s["total"], rec["position_limit"]))
        return rec
    kind, cap = rec["kind"], spec.get("time_cap")

    def over_cap(next_cost: float) -> bool:
        return cap is not None and (time.perf_counter() - started) + next_cost > float(cap)

    last = 0.0
    for w in range(s["warmup"]):
        if w and over_cap(last):
            rec.update(status=Status.PARTIAL.value,
                       reason="time cap %.0f s reached during warmup (%d of %d warmup requests)"
                              % (cap, w, s["warmup"]))
            return rec
        states, questions, _ = build_request(agent, s["total"], s["seed"] + WARMUP_SEED_OFFSET + w,
                                             s["nq"], s["opts"], s["batch"])
        assert_clean(agent)
        t0 = time.perf_counter()
        _call(agent, kind, states, questions, s["total"])
        last = time.perf_counter() - t0
        if w == 0 and (load_info or {}).get("compile"):
            rec["compile_seconds"] = last  # the first call includes compilation

    samples: List[float] = []
    accounting = None
    clean = None
    for r in range(s["repeats"]):
        if (samples or last) and over_cap(last):
            break
        states, questions, acc = build_request(agent, s["total"], s["seed"] + r, s["nq"], s["opts"], s["batch"])
        if accounting is None:
            accounting = acc
        clean = assert_clean(agent)
        t0 = time.perf_counter()
        _call(agent, kind, states, questions, s["total"])
        last = time.perf_counter() - t0
        samples.append(last * 1000.0)

    rec["repeats"]["completed"] = len(samples)
    rec["samples_ms"] = samples
    rec["token_accounting"] = accounting
    rec["clean_path"] = clean
    rec["fallbacks"] = {
        "cpu_fallback_count": int(getattr(agent, "cpu_fallback_count", 0) or 0),
        "last_fallback_reason": getattr(agent, "last_fallback_reason", None),
        "device": str(agent.device),
        "dtype": str(agent.dtype).replace("torch.", ""),
        "amp_enabled": bool(agent.amp_enabled),
    }
    if samples:
        rec["timings_ms"] = summarize(samples)
        med = rec["timings_ms"]["p50"]
        rec["first_vs_median"] = {"first_ms": samples[0], "median_ms": med,
                                  "ratio": samples[0] / med if med else None}
    if len(samples) < s["repeats"]:
        rec.update(status=Status.PARTIAL.value,
                   reason="time cap %.0f s reached after %d of %d repeats"
                          % (float(cap), len(samples), s["repeats"]))
    else:
        rec["status"] = Status.MEASURED.value
    return rec


def measure_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Child-side entry point for `runner.run_condition`."""
    from .runner import load_agent
    started = time.perf_counter()
    agent, info = load_agent(spec["model"], spec.get("revision"), spec.get("threads"),
                             compile=bool(spec.get("compile", False)))
    rec = measure_in_process(agent, spec, info, started=started)
    rec["model"] = {"id": info["model"], "revision": info["revision"]}
    return rec


def compare_paths(spec: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """T029: the same fixed requests with the stage wrappers off and on, interleaved.

    Reports ``overhead_ratio = p50(on) / p50(off)`` so the report can show that the profiler's
    stage wrappers, not the clean path, carry any instrumentation cost (SC-005).
    """
    from .inputs import build_request
    from .profile import stage_wrappers

    info = None
    if agent is None:
        from .runner import load_agent
        agent, info = load_agent(spec["model"], spec.get("revision"), spec.get("threads"))
    s = _spec_ints(spec)
    kind = condition_kind(s["nq"], s["batch"])
    reqs = [build_request(agent, s["total"], s["seed"] + r, s["nq"], s["opts"], s["batch"])
            for r in range(s["repeats"])]
    warm = build_request(agent, s["total"], s["seed"] + WARMUP_SEED_OFFSET, s["nq"], s["opts"], s["batch"])
    for _ in range(max(1, s["warmup"])):
        _call(agent, kind, warm[0], warm[1], s["total"])
    off: List[float] = []
    on: List[float] = []
    for states, questions, _ in reqs:
        assert_clean(agent)
        t0 = time.perf_counter()
        _call(agent, kind, states, questions, s["total"])
        off.append((time.perf_counter() - t0) * 1000.0)
        with stage_wrappers(agent):
            t0 = time.perf_counter()
            _call(agent, kind, states, questions, s["total"])
            on.append((time.perf_counter() - t0) * 1000.0)
    so, sn = summarize(off), summarize(on)
    rec = base_record(spec, agent)
    rec.update({
        "status": Status.MEASURED.value,
        "kind": "overhead_check",
        "wrappers_off_ms": so,
        "wrappers_on_ms": sn,
        "overhead_ratio": sn["p50"] / so["p50"] if so["p50"] else None,
        "load_seconds": (info or {}).get("load_seconds"),
    })
    return rec

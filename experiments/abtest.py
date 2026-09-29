"""Instrumentation A/B test (research.md R16).

The full run showed profiled requests 15-30% *faster* than clean ones at 1,024 tokens and above,
which drift does not explain. This runs the same documents in one process, interleaving modes
request by request, so the machine's state is shared and only the instrumentation differs:

* ``clean``     no wrappers, hooks or profiler (the sweep's path)
* ``wrappers``  `profile.stage_wrappers` only
* ``labels``    `profile.label_modules` (module forward hooks with record_function labels) only
* ``profiler``  `torch.profiler` only
* ``all``       everything together (the original profile command's path)
* ``fastpath_off``    no instrumentation, PyTorch's TransformerEncoderLayer fast path disabled (R17)
* ``labels_encoder``  hooks on the encoder only, head left on the fast path (the fixed profile path)

The mode order rotates every repeat, so no mode always runs first. The intra-op thread count is
read after every call, since a profiler that changes it would explain a speed difference.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List

from .results import Status, summarize

MODES = ("clean", "wrappers", "labels", "profiler", "all", "fastpath_off", "labels_encoder")


@contextlib.contextmanager
def _mode(agent, mode: str):
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile
    from .profile import label_modules, stage_wrappers

    with contextlib.ExitStack() as stack:
        if mode == "fastpath_off":
            prev = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            stack.callback(torch.backends.mha.set_fastpath_enabled, prev)
        if mode == "labels_encoder":
            stack.enter_context(label_modules(agent))  # the head stays unhooked while the fast path is on
        if mode in ("wrappers", "all"):
            stack.enter_context(stage_wrappers(agent))
        if mode in ("labels", "all"):
            stack.enter_context(label_modules(agent))
            stack.enter_context(_hook_head(agent))  # the original labels mode also hooked the head
        if mode in ("profiler", "all"):
            stack.enter_context(torch_profile(activities=[ProfilerActivity.CPU], record_shapes=False))
        yield


@contextlib.contextmanager
def _hook_head(agent):
    """No-op hooks on the decision-head layers: reproduces the original profile path (fast path off)."""
    head = getattr(agent.model, "head", None)
    handles = [layer.register_forward_hook(lambda *a: None) for layer in (getattr(head, "layers", []) or [])]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def abtest_in_process(agent, spec: Dict[str, Any], load_info=None, started=None) -> Dict[str, Any]:
    import torch
    from .inputs import build_request
    from .timing import WARMUP_SEED_OFFSET, _call, assert_clean, base_record

    started = time.perf_counter() if started is None else started
    L = int(spec["total_tokens"])
    repeats = int(spec.get("repeats", 4))
    modes = [m for m in spec.get("modes", MODES) if m in MODES]
    cap = spec.get("time_cap")
    seed = int(spec.get("seed", 0))
    rec = base_record(dict(spec, questions=1, batch_size=1), agent)
    rec.update(kind="abtest", modes=modes, load_seconds=(load_info or {}).get("load_seconds"))
    if L > rec["position_limit"]:
        rec.update(status=Status.UNSUPPORTED.value, reason="%d exceeds positional capacity" % L)
        return rec
    reqs = [build_request(agent, L, seed + r) for r in range(repeats)]
    warm = build_request(agent, L, seed + WARMUP_SEED_OFFSET)
    t0 = time.perf_counter()
    _call(agent, "single", warm[0], warm[1], L)
    last = time.perf_counter() - t0
    samples: Dict[str, List[float]] = {m: [] for m in modes}
    threads: Dict[str, List[int]] = {m: [] for m in modes}
    order_log = []
    stopped = None
    for r, (states, questions, _) in enumerate(reqs):
        order = modes[r % len(modes):] + modes[:r % len(modes)]
        order_log.append(order)
        for m in order:
            if cap is not None and (time.perf_counter() - started) + 1.5 * last > float(cap):
                stopped = "time cap %.0f s reached at repeat %d" % (float(cap), r)
                break
            if m == "clean":
                assert_clean(agent)
            with _mode(agent, m):
                t0 = time.perf_counter()
                _call(agent, "single", states, questions, L)
                last = time.perf_counter() - t0
            samples[m].append(last * 1000.0)
            threads[m].append(torch.get_num_threads())
        if stopped:
            break
    summary = {m: summarize(v) for m, v in samples.items() if v}
    base = summary.get("clean", {}).get("p50")
    rec.update(
        samples_ms=samples, order=order_log, threads_after_call=threads,
        timings_ms=summary,
        ratio_to_clean={m: (s["p50"] / base if base else None) for m, s in summary.items()},
        repeats={"requested": repeats, "completed": min((len(v) for v in samples.values()), default=0)},
    )
    if stopped:
        rec.update(status=Status.PARTIAL.value, reason=stopped)
    else:
        rec["status"] = Status.MEASURED.value
    return rec


def abtest_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    from .runner import load_agent
    started = time.perf_counter()
    agent, info = load_agent(spec["model"], spec.get("revision"), spec.get("threads"),
                             mha_fastpath=bool(spec.get("mha_fastpath", True)))
    rec = abtest_in_process(agent, spec, info, started=started)
    rec["mha_fastpath_default"] = info["mha_fastpath"]
    rec["model"] = {"id": info["model"], "revision": info["revision"]}
    return rec

"""Kernel microbenchmarks: correctness, timing, memory and executed work (T035, research.md R8/R9).

`bench_condition(spec)` runs in its own process (`runner.run_condition`), so peak memory is per
condition. For one implementation and one shape it:

1. runs the correctness cases against `reference_attention` with the same `MaskSpec` pattern
   (absolute tolerance 1e-5 in fp32); a failing kernel is recorded ``failed`` and not timed;
2. times the kernel on the audited shape (batch 1, the model's heads and head size), with mask
   construction, index building, padding, copies and allocations inside the timed call;
3. records the analytical score-matrix size and the score elements the kernel actually computes,
   apart from measured peak memory.

`executed_work(items)` fits time against length per implementation and compares with dense.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..results import Status, summarize
from .dense import dense_attention, dense_masked_attention
from .gas import gas_attention
from .local import local_attention
from .reference import MaskSpec, reference_attention, score_matrix_bytes

TOLERANCE = 1e-5

#: impl -> (function, pattern it computes)
IMPLS: Dict[str, Tuple[Callable, str]] = {
    "dense": (dense_attention, "full"),
    "dense_masked": (dense_masked_attention, "block_local"),
    "local": (local_attention, "block_local"),
    "gas": (gas_attention, "gather"),
}

#: (case, lengths per row, sequence length). Block 16, selection 4 blocks, 2 heads, head dim 16.
CASES = [
    ("padding", [64, 55], 64),
    ("boundary_chunk", [50], 64),
    ("non_divisible_length", [70], 70),
    ("fully_masked_row", [0], 48),
    ("mixed_batch", [70, 63, 0, 9], 70),
]
CASE_BLOCK, CASE_SELECTION, CASE_HEADS, CASE_DIM = 16, 4, 2, 16

LESS_WORK_MARGIN = 0.5
SAME_WORK_MARGIN = 0.25
MIN_DENSE_EXPONENT = 1.3


def default_repeats(L: int) -> int:
    return 20 if L <= 2048 else 10


def correctness(impl: str) -> List[Dict[str, Any]]:
    fn, pattern = IMPLS[impl]
    gen = torch.Generator().manual_seed(1234)
    out = []
    for name, lengths, L in CASES:
        q, k, v = (torch.randn(len(lengths), CASE_HEADS, L, CASE_DIM, generator=gen) for _ in range(3))
        spec = MaskSpec(lengths, pattern, CASE_BLOCK, CASE_SELECTION)
        with torch.no_grad():
            ref = reference_attention(q, k, v, spec)
            got = fn(q, k, v, spec)
        finite = bool(torch.isfinite(got).all())
        err = float((ref - got).abs().max()) if finite else float("inf")
        out.append({"case": name, "max_abs_error": err, "tolerance": TOLERANCE,
                    "finite": finite, "passed": finite and err <= TOLERANCE})
    return out


def issued_score_elements(impl: str, B: int, H: int, L: int, block: int, sel_blocks: int) -> int:
    """Score entries the kernel computes (from the tensor shapes it issues), not those it keeps."""
    nb = -(-L // block)
    if impl in ("dense", "dense_masked"):
        return B * H * L * L
    if impl == "local":
        return B * H * nb * block * 3 * block
    return B * H * nb * block * sel_blocks * block


def bench_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    impl = spec["impl"]
    L, H, D = int(spec["length"]), int(spec["heads"]), int(spec["head_dim"])
    B = int(spec.get("batch", 1))
    block = int(spec.get("block") or 128)
    sel = int(spec.get("selection_blocks") or 4)
    repeats = spec.get("repeats") or "auto"
    repeats = default_repeats(L) if repeats == "auto" else int(repeats)
    warmup = int(spec.get("warmup", 2))
    if spec.get("threads"):
        torch.set_num_threads(int(spec["threads"]))
    fn, pattern = IMPLS[impl]
    rec: Dict[str, Any] = {
        "impl": impl, "pattern": pattern,
        "shape": {"batch": B, "heads": H, "length": L, "head_dim": D,
                  "block": block if impl != "dense" else None,
                  "selection_blocks": sel if impl == "gas" else None,
                  "selection_tokens": sel * block if impl == "gas" else None},
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "score_matrix_bytes_analytical": score_matrix_bytes(B, H, L, 4),
        "issued_score_elements": issued_score_elements(impl, B, H, L, block, sel),
        "timing_includes": ["mask construction", "index building", "padding", "copies", "allocations"],
    }
    checks = correctness(impl)
    rec["correctness"] = checks
    if not all(c["passed"] for c in checks):
        bad = [c["case"] for c in checks if not c["passed"]]
        rec.update(status=Status.FAILED.value, cause="correctness",
                   reason="disagrees with the reference beyond %g on: %s" % (TOLERANCE, ", ".join(bad)))
        return rec
    gen = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    q, k, v = (torch.randn(B, H, L, D, generator=gen) for _ in range(3))
    mspec = MaskSpec([L] * B, pattern, block, sel)
    cap = spec.get("time_cap")
    started = time.perf_counter()
    samples: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            fn(q, k, v, mspec)
        for _ in range(repeats):
            if cap is not None and samples and (time.perf_counter() - started) + samples[-1] / 1000 > float(cap):
                break
            t0 = time.perf_counter()
            fn(q, k, v, mspec)
            samples.append((time.perf_counter() - t0) * 1000.0)
    rec["samples_ms"] = samples
    rec["repeats"] = {"requested": repeats, "completed": len(samples), "warmup": warmup}
    if samples:
        rec["timings_ms"] = summarize(samples)
    if len(samples) < repeats:
        rec.update(status=Status.PARTIAL.value,
                   reason="time cap reached after %d of %d repeats" % (len(samples), repeats))
    else:
        rec["status"] = Status.MEASURED.value
    return rec


def _group_key(it: Dict[str, Any]) -> Tuple:
    s = it["shape"]
    return (it["impl"], s.get("block"), s.get("selection_blocks"))


def executed_work(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Time-vs-length exponent per implementation and a verdict against dense (FR-023)."""
    from ..profile import fit_exponent

    groups = defaultdict(list)
    for i, it in enumerate(items):
        if it.get("status") in ("measured", "partial") and it.get("timings_ms"):
            groups[_group_key(it)].append((it["shape"]["length"], it["timings_ms"]["p50"],
                                           it["issued_score_elements"], i))
    dense_exp = None
    for key, pts in groups.items():
        if key[0] == "dense":
            pts.sort()
            dense_exp = fit_exponent([p[0] for p in pts], [p[1] for p in pts])
    out = {"dense_exponent": dense_exp, "thresholds": {"less_work_margin": LESS_WORK_MARGIN,
                                                       "same_work_margin": SAME_WORK_MARGIN,
                                                       "min_dense_exponent": MIN_DENSE_EXPONENT},
           "groups": []}
    for key, pts in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        pts.sort()
        lengths = [p[0] for p in pts]
        exp = fit_exponent(lengths, [p[1] for p in pts])
        issued = fit_exponent(lengths, [p[2] for p in pts])
        verdict = classify(key[0], exp, dense_exp)
        out["groups"].append({"impl": key[0], "block": key[1], "selection_blocks": key[2],
                              "lengths": lengths, "time_exponent": exp,
                              "issued_elements_exponent": issued, "verdict": verdict,
                              "derived_from": ["kernels.json#%d" % p[3] for p in pts]})
    return out


def classify(impl: str, exponent: Optional[float], dense_exponent: Optional[float]) -> str:
    if impl == "dense":
        return "baseline"
    if exponent is None or dense_exponent is None or dense_exponent < MIN_DENSE_EXPONENT:
        return "not_determined"
    if exponent <= dense_exponent - LESS_WORK_MARGIN:
        return "less_work"
    if exponent >= dense_exponent - SAME_WORK_MARGIN:
        return "same_work"
    return "not_determined"

"""Quality metrics for Phase 2 (T005; research.md R10; FR-021, FR-022).

* ``accuracy`` is exact match with the gold label for every question type (clarified 2026-09-29).
* ``ece`` is classification calibration error from the *predicted answer's* probability, in ten
  equal-width bins. Entropy-based confidence is never used (constitution, technical constraints).
* ``fit_temperature`` fits one temperature by minimizing log loss; the caller passes calibration
  split data only (FR-019).
* ``paired_cluster_bootstrap`` resamples *cases*, because the questions of one case are dependent
  (FR-022). ``verdict`` turns an interval into better / equal / worse / inconclusive.

NumPy only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

N_BINS = 10
DEFAULT_BOOTSTRAP = 5000
MARGIN_PP = 2.0


def accuracy(correct: Sequence[Any]) -> Optional[float]:
    """Share of items answered exactly right; None when there are no items."""
    if len(correct) == 0:
        return None
    return float(np.mean(np.asarray(correct, dtype=float)))


def level_error(predicted: Sequence[float], gold: Sequence[float]) -> Optional[float]:
    """Mean absolute error between predicted and gold ordinal levels."""
    if len(predicted) == 0:
        return None
    return float(np.mean(np.abs(np.asarray(predicted, dtype=float) - np.asarray(gold, dtype=float))))


def ece(confidences: Sequence[float], correct: Sequence[Any], n_bins: int = N_BINS) -> Optional[float]:
    """Expected calibration error over `n_bins` equal-width bins of the predicted answer's probability.

    `confidences` must be the probability of the *predicted answer* (for ``noul`` the larger of
    P(true) and P(false); for ``score`` the most probable level's probability). A value of exactly
    1.0 falls in the last bin.
    """
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return None
    if conf.shape != hit.shape:
        raise ValueError("confidences and correct must have the same length")
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(conf[mask].mean() - hit[mask].mean())
    return float(total / conf.size)


def scale_probs(probs: Sequence[float], temperature: float) -> np.ndarray:
    """Temperature-scale a probability vector: p_i^(1/T), renormalized."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    z = np.log(p) / float(temperature)
    z -= z.max()
    q = np.exp(z)
    return q / q.sum()


def _nll(prob_vectors: List[np.ndarray], gold_idx: Sequence[int], temperature: float) -> float:
    total = 0.0
    for p, g in zip(prob_vectors, gold_idx):
        total -= math.log(max(float(scale_probs(p, temperature)[g]), 1e-12))
    return total / max(1, len(gold_idx))


def fit_temperature(prob_vectors: Sequence[Sequence[float]], gold_idx: Sequence[int],
                    lo: float = 0.25, hi: float = 4.0, iters: int = 60) -> float:
    """Temperature minimizing mean log loss (golden-section search over log T)."""
    if len(gold_idx) == 0:
        return 1.0
    vecs = [np.asarray(p, dtype=float) for p in prob_vectors]
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(vecs, gold_idx, math.exp(c)), _nll(vecs, gold_idx, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(vecs, gold_idx, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(vecs, gold_idx, math.exp(d))
    return float(math.exp((a + b) / 2))


def paired_cluster_bootstrap(a_correct: Sequence[Any], b_correct: Optional[Sequence[Any]],
                             case_ids: Sequence[Any], n_boot: int = DEFAULT_BOOTSTRAP,
                             seed: int = 0) -> Dict[str, Any]:
    """Case-clustered bootstrap of ``mean(a) - mean(b)`` (or of ``mean(a)`` when `b` is None).

    The i-th entries of all three sequences describe the same item. Cases are resampled with
    replacement, and each case brings all of its items, so questions of one case stay together.
    Returns ``diff``, a 95% percentile interval ``lo``/``hi`` (as accuracy fractions), and counts.
    """
    a = np.asarray(a_correct, dtype=float)
    b = np.zeros_like(a) if b_correct is None else np.asarray(b_correct, dtype=float)
    if a.shape != b.shape or a.shape[0] != len(case_ids):
        raise ValueError("a_correct, b_correct and case_ids must be aligned")
    if a.size == 0:
        return {"diff": None, "lo": None, "hi": None, "n_items": 0, "n_cases": 0, "n_boot": 0}
    labels, inverse = np.unique(np.asarray(list(case_ids), dtype=object).astype(str), return_inverse=True)
    n_cases = len(labels)
    diff_sum = np.bincount(inverse, weights=a - b, minlength=n_cases)
    count = np.bincount(inverse, minlength=n_cases).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_cases, size=(int(n_boot), n_cases))
    stats = diff_sum[idx].sum(axis=1) / count[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": float((a - b).mean()), "lo": float(lo), "hi": float(hi),
            "n_items": int(a.size), "n_cases": int(n_cases), "n_boot": int(n_boot)}


def verdict(lo: Optional[float], hi: Optional[float], margin_pp: float = MARGIN_PP) -> str:
    """``better`` (lower bound above 0), ``worse`` (upper bound below 0), ``equal`` (interval inside
    the margin), otherwise ``inconclusive`` (research.md R10). `lo` and `hi` are accuracy fractions."""
    if lo is None or hi is None:
        return "inconclusive"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    m = margin_pp / 100.0
    if lo >= -m and hi <= m:
        return "equal"
    return "inconclusive"

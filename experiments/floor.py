"""Cost floor: what remains if attention score/value work were free (T036, research.md R9, FR-025).

* Per length, **estimate**: ``floor_ms = total x (1 - attention_score_value share)``. The total is the
  clean p50 from ``sweep.json`` when present (the share comes from the profile), otherwise the
  profiled total. Projections, MLPs, norms, all decision-head layers and data preparation remain.
* **Analytical**: the plan's operation-count bound ``8 x (1 - f)`` for 4,096 against 512 tokens,
  with ``f`` the attention score/value share at 512: linear-layer work grows 8x from 512 to 4,096,
  so even free attention leaves 4K at least ``8 x (1 - f)`` times the 512-token latency.

The two are separate fields and never merged (constitution II).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from . import results

SV = "attention_score_value"


def build_floor(run_path: Union[str, Path]) -> Dict[str, Any]:
    run_path = Path(run_path)
    if not (run_path / "profile.json").exists():
        raise FileNotFoundError("floor needs profile.json in %s; run `python -m experiments profile` first" % run_path)
    prof = results.read_json(run_path, "profile.json")
    clean: Dict[int, Dict[str, Any]] = {}
    if (run_path / "sweep.json").exists():
        for i, it in enumerate(results.read_json(run_path, "sweep.json").get("items", [])):
            if it.get("kind") == "single" and it.get("status") == "measured" and it.get("timings_ms"):
                clean[it["condition"]["total_tokens"]] = {"p50": it["timings_ms"]["p50"], "ref": "sweep.json#%d" % i}
    items = []
    share_at: Dict[int, float] = {}
    for i, it in enumerate(prof.get("items", [])):
        L = it["condition"]["total_tokens"]
        if it.get("status") not in ("measured", "partial") or not it.get("components"):
            items.append({"length": L, "status": it.get("status", "failed"),
                          "reason": "no usable profile at this length: %s" % it.get("reason", "not measured"),
                          "derived_from": ["profile.json#%d" % i]})
            continue
        share = it["components"][SV]["share"]
        share_at[L] = share
        c = clean.get(L)
        total = c["p50"] if c else it["total_profiled_ms"]
        items.append({
            "length": L, "status": "measured", "label": "estimate",
            "total_ms": total, "total_basis": "clean p50" if c else "profiled total",
            "attention_score_value_share": share,
            "floor_ms": total * (1.0 - share), "floor_fraction": 1.0 - share,
            "drift_flagged": bool((it.get("drift") or {}).get("flagged")),
            "derived_from": ["profile.json#%d" % i] + ([c["ref"]] if c else []),
        })
    analytical: Dict[str, Any] = {"label": "analytical",
                                  "formula": "latency(4096) / latency(512) >= 8 x (1 - f512) when attention is free"}
    f = share_at.get(512)
    if f is None:
        analytical.update(status="unsupported", reason="no measured profile at 512 tokens")
    else:
        analytical.update(status="measured", f_512=f, bound_ratio_4096_vs_512=8.0 * (1.0 - f))
        t512 = clean.get(512, {}).get("p50")
        if t512:
            analytical["bound_ms_4096"] = 8.0 * (1.0 - f) * t512
            analytical["derived_from"] = [clean[512]["ref"]]
    return {"items": items, "analytical_bound": analytical}

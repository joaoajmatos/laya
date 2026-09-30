"""Token-length profile of upstream cases under Laya's tokenizer (T015; FR-004; SC-002).

A *row* is one question of one case as the model sees it: ``[CLS] head [SEP] options [SEP] state
[SEP]``. Its length is ``special + task/option + state`` tokens, reconciled by `tokens.account`.
A *case* has five rows of different lengths (the question heads differ), so a case's length is its
longest row: a case "fits within N" only when every one of its rows does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import data, tokens

FIT_LENGTHS = (512, 1024, 2048, 4096, 8192)


def stats(values: Sequence[float]) -> Dict[str, Any]:
    """min, median, p90, p99, max (linear interpolation) and the count."""
    if len(values) == 0:
        return {"n": 0, "min": None, "median": None, "p90": None, "p99": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def fit_share(values: Sequence[float], lengths: Sequence[int] = FIT_LENGTHS) -> Dict[str, Optional[float]]:
    """Share of `values` that are at most each named length."""
    if len(values) == 0:
        return {str(n): None for n in lengths}
    v = np.asarray(values, dtype=float)
    return {str(n): float((v <= n).mean()) for n in lengths}


def profile_rows(agent, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row token accounting for every (case, question). Questions whose options overflow the
    head budget (Laya then caps each option's tokens) are flagged in their row and listed under
    ``over_head_budget``, reported separately and never dropped silently."""
    from laya.common import build_sequence

    limit = tokens.position_limit(agent)
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    rows: List[Dict[str, Any]] = []
    over: List[Dict[str, Any]] = []
    for c in cases:
        for qid, qdef in c["questions"].items():
            state_text = data.serialize_state(c["state"])
            try:
                rec = tokens.account(agent, state_text, {qid: qdef}, max_len=limit)
            except ValueError as exc:           # the markers did not survive the head budget
                over.append({"case_id": c["id"], "workflow": c["workflow"], "question_id": qid,
                             "n_options": c["qmeta"][qid]["n_options"], "reason": str(exc)})
                continue
            r = rec["rows"][0]
            _, _, stats = build_sequence(agent.tok, state_text, agent._to_internal(qdef), max_len=limit,
                                         head_max_len=head_max_len, state_ids=[], return_stats=True)
            squeezed = stats["tokens_per_option"] is not None or stats["options_distinct"] < stats["options"]
            if squeezed:
                over.append({"case_id": c["id"], "workflow": c["workflow"], "question_id": qid,
                             "n_options": c["qmeta"][qid]["n_options"],
                             "reason": "options share the %d-token head budget (tokens_per_option=%s, distinct=%d of %d)"
                                       % (head_max_len, stats["tokens_per_option"], stats["options_distinct"],
                                          stats["options"])})
            rows.append({
                "case_id": c["id"], "workflow": c["workflow"], "question_id": qid,
                "question_type": qdef["type"], "n_options": c["qmeta"][qid]["n_options"],
                "row_tokens": r["row_tokens"], "state_tokens": r["state_tokens_original"],
                "task_option_tokens": r["task_option_tokens"], "special_tokens": r["special_tokens"],
                "low_confidence": c["qmeta"][qid]["low_confidence"], "over_head_budget": squeezed,
            })
    return {"rows": rows, "over_head_budget": over}


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Row-level and case-level distributions with the share fitting each named length."""
    case_max: Dict[str, int] = {}
    for r in rows:
        case_max[r["case_id"]] = max(case_max.get(r["case_id"], 0), r["row_tokens"])
    row_tokens = [r["row_tokens"] for r in rows]
    return {"rows": dict(stats(row_tokens), fit_share=fit_share(row_tokens)),
            "cases": dict(stats(list(case_max.values())), fit_share=fit_share(list(case_max.values())))}


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_wf: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_wf.setdefault(r["workflow"], []).append(r)
    return {"overall": summarize(rows), "by_workflow": {wf: summarize(v) for wf, v in sorted(by_wf.items())}}


def build_length_profile(agent, model_id: str, root: Optional[Path] = None,
                         splits: Sequence[str] = data.SPLITS) -> Dict[str, Any]:
    """Length profile of the imported upstream splits (rows are kept for the test split only)."""
    man = data.read_data_json("manifest.json", root)
    body: Dict[str, Any] = {
        "schema_version": data.SCHEMA_VERSION, "data_id": man["data_id"], "model": model_id,
        "configured_max_len": int(agent.cfg.get("max_len", 512)),
        "head_max_len": int(agent.cfg.get("head_max_len", 192)),
        "position_limit": tokens.position_limit(agent), "fit_lengths": list(FIT_LENGTHS),
        "unit": "row = one question of one case, special + question/option + state tokens",
        "by_split": {},
    }
    for s in splits:
        prof = profile_rows(agent, data.load_cases(s, root))
        entry = aggregate(prof["rows"])
        entry["over_head_budget"] = prof["over_head_budget"]
        if s == "test":
            entry["rows"] = prof["rows"]
        body["by_split"][s] = entry
    return body


def write_length_profile(agent, model_id: str, root: Optional[Path] = None) -> Path:
    body = build_length_profile(agent, model_id, root)
    return data._write_json(data.data_root(root) / "length_profile.json", body)

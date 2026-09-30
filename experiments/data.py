"""Upstream `typed-decisions` import, fingerprint, low-confidence cutoff and frozen splits.

Specs: specs/002-decision-benchmark-baselines (research.md R1, R2, R13; FR-001 to FR-005, FR-012).

* The dataset is read from parquet with ``pyarrow`` (imported lazily, so modules that only load
  prebuilt items import cleanly in an environment without it) at a pinned revision.
* A content fingerprint over the canonical rows is recorded in ``experiments/data/manifest.json``
  and checked on every later import (FR-002). Raw upstream records live only under
  ``experiments/data/cache/`` (git-ignored, FR-003).
* ``factors`` is diagnostic. It is never part of a model input.
* Splits are by case, stratified by workflow, seeded and fingerprinted (FR-012). The final split is
  built here and never scored (FR-013).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .results import Refusal

DATASET_ID = "LocalLLaMA/typed-decisions"
#: Pinned dataset revision (observed 2026-09-29).
DATASET_REVISION = "d51d993547ad8355b1c25157fbc1fea0649e8ffa"
LICENSE = "apache-2.0"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
SPLITS = ("train", "test")
EVAL_SPLITS = ("dev", "calibration", "final")

#: Files of the pinned repository that this phase reads.
DATA_FILES = tuple(["%s/%s-00000-of-00001.parquet" % (cfg, split)
                    for cfg in ("all",) + WORKFLOWS for split in SPLITS])

SPLIT_RULE = "stratified_by_workflow_30_20_50"
SPLIT_FRACTIONS = (("dev", 0.3), ("calibration", 0.2))          # the rest is final
DEFAULT_SPLIT_SEED = 2002
LOW_CONFIDENCE_QUANTILE = 25                                     # bottom quartile, computed on train (R13)
AGREEMENT_QUANTILE = 75                                          # top quartile of total_variation
LATENCY_PER_WORKFLOW = 3
VARIANT_CASES_PER_WORKFLOW = 5

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(__file__).resolve().parent / "data"

JSON_COLUMNS = ("state", "questions", "gold", "factors", "label_agreement")


class DataError(ValueError):
    """The imported data is inconsistent (counts, configs, schema)."""


# --------------------------------------------------------------------------- reading

def data_root(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else DATA_ROOT


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq          # lazy: only the parquet-reading function needs pyarrow
    return pq.read_table(str(path)).to_pylist()


def parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode the JSON columns of one upstream row."""
    out = {k: row[k] for k in row if k not in JSON_COLUMNS}
    for col in JSON_COLUMNS:
        val = row.get(col)
        out[col] = json.loads(val) if isinstance(val, str) else val
    return out


def _canonical(row: Dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """SHA-256 over the canonical parsed rows sorted by id (independent of row order and whitespace)."""
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        h.update(_canonical(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def combined_fingerprint(by_split: Dict[str, str]) -> str:
    return hashlib.sha256("|".join("%s=%s" % (k, by_split[k]) for k in sorted(by_split)).encode()).hexdigest()


def _hf_download(revision: str) -> Callable[[str], str]:
    def download(rel: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(DATASET_ID, rel, repo_type="dataset", revision=revision)
    return download


def _load_split_rows(base: Path, cfg: str, split: str) -> List[Dict[str, Any]]:
    rows = [parse_row(r) for r in _read_parquet(base / cfg / ("%s-00000-of-00001.parquet" % split))]
    for r in rows:
        if r.get("split") not in (None, split):
            raise DataError("row %s in %s/%s says split %r" % (r.get("id"), cfg, split, r.get("split")))
    return rows


def _question_confidences(rows: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(q["confidence"]) for r in rows for q in r["gold"].values() if "confidence" in q]


def _question_types(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        for q in r["questions"].values():
            counts[q["type"]] = counts.get(q["type"], 0) + 1
    return counts


def _split_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_wf: Dict[str, int] = {}
    for r in rows:
        by_wf[r["workflow"]] = by_wf.get(r["workflow"], 0) + 1
    return {"cases": len(rows), "questions": sum(len(r["questions"]) for r in rows),
            "questions_by_type": _question_types(rows), "cases_by_workflow": dict(sorted(by_wf.items()))}


def _checkpoint_block() -> Dict[str, Any]:
    from laya.revisions import PINNED_REVISIONS
    return {
        "fine_tuned": {"model": "convaiinnovations/laya-typed-decisions",
                       "revision": PINNED_REVISIONS.get("convaiinnovations/laya-typed-decisions"),
                       "configured_max_len": 1024, "head_max_len": 256},
        "base": {"model": "convaiinnovations/laya", "revision": PINNED_REVISIONS.get("convaiinnovations/laya"),
                 "configured_max_len": 512},
    }


def _write_json(path: Path, body: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def read_data_json(name: str, root: Optional[Path] = None) -> Dict[str, Any]:
    path = data_root(root) / name
    if not path.exists():
        raise FileNotFoundError("%s not found; run the earlier command that writes it" % path)
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- import (T013)

def cross_check_configs(all_rows: Sequence[Dict[str, Any]], wf_rows: Sequence[Dict[str, Any]], split: str) -> None:
    """The per-workflow configs must describe the same cases as ``all``.

    Case ids must match exactly, and every column they share must be equal. The real per-workflow
    files store the gold labels flattened (``<question>__label``, ...) instead of one JSON column,
    so those are checked against ``all``'s ``gold`` for the label (2026-09-29 spike of the pinned revision).
    """
    if sorted(r["id"] for r in wf_rows) != sorted(r["id"] for r in all_rows):
        raise DataError("per-workflow configs disagree with the 'all' config on %s case ids" % split)
    by_id = {r["id"]: r for r in all_rows}
    for r in wf_rows:
        a = by_id[r["id"]]
        for key, value in r.items():
            if "__" in key:
                qid, field = key.split("__", 1)
                if field == "label" and str(a["gold"].get(qid, {}).get("label")) != str(value):
                    raise DataError("case %s: %s differs between the per-workflow config and 'all'" % (r["id"], key))
            elif key in a and a[key] != value:
                raise DataError("case %s: column %r differs between the per-workflow config and 'all' (%s)"
                                % (r["id"], key, split))


def import_dataset(revision: str = DATASET_REVISION, root: Optional[Path] = None,
                   download: Optional[Callable[[str], str]] = None, refresh: bool = False,
                   with_checkpoints: bool = True) -> Dict[str, Any]:
    """Import the pinned dataset, verify it and write the Data Manifest.

    A cached copy is used when present (so reruns work offline); ``refresh`` re-downloads it first.
    When a manifest already exists its fingerprint must match, otherwise `Refusal`
    (``fingerprint_mismatch``, FR-002). Per-workflow configs must agree with ``all``.
    """
    root = data_root(root)
    cache = root / "cache"
    fetch = download or _hf_download(revision)
    for rel in DATA_FILES:
        target = cache / rel
        if refresh or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetch(rel), target)

    rows = {s: _load_split_rows(cache, "all", s) for s in SPLITS}
    for s in SPLITS:
        per_wf = [r for wf in WORKFLOWS for r in _load_split_rows(cache, wf, s)]
        cross_check_configs(rows[s], per_wf, s)
        ids = [r["id"] for r in rows[s]]
        if len(set(ids)) != len(ids):
            raise DataError("duplicate case ids inside the %s split" % s)

    fps = {s: fingerprint_rows(rows[s]) for s in SPLITS}
    fp = combined_fingerprint(fps)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded.get("fingerprint", {}).get("combined") != fp:
            raise Refusal("fingerprint_mismatch",
                          "imported data has fingerprint %s but the manifest recorded %s; refusing to proceed "
                          "(delete experiments/data/ to re-import a new revision)"
                          % (fp[:16], str(recorded.get("fingerprint", {}).get("combined"))[:16]))

    conf = _question_confidences(rows["train"])
    tv = [float(a["total_variation"]) for r in rows["train"] for a in r["label_agreement"].values()
          if "total_variation" in a]
    import datetime as _dt
    body = {
        "schema_version": SCHEMA_VERSION,
        "data_id": "typed-decisions@%s" % revision[:12],
        "source": DATASET_ID,
        "revision": revision,
        "license": LICENSE,
        "configs": ["all"] + list(WORKFLOWS),
        "subsets": list(WORKFLOWS),
        "splits": {s: _split_summary(rows[s]) for s in SPLITS},
        "workflows": {s: _split_summary(rows[s])["cases_by_workflow"] for s in SPLITS},
        "fingerprint": dict(fps, combined=fp),
        "low_confidence": {"cutoff": float(np.percentile(conf, LOW_CONFIDENCE_QUANTILE)), "basis": "train",
                           "quantile": LOW_CONFIDENCE_QUANTILE / 100.0, "n_questions": len(conf)},
        "agreement": {"total_variation_top_quartile_cutoff": float(np.percentile(tv, AGREEMENT_QUANTILE)) if tv else None,
                      "basis": "train"},
        "checkpoints": _checkpoint_block() if with_checkpoints else {},
        "imported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(manifest_path, body)
    return body


# --------------------------------------------------------------------------- cases

def load_cases(split: str, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Upstream cases of ``train`` or ``test`` with derived per-question metadata (``qmeta``).

    ``qmeta[qid]`` has ``type``, ``n_options``, ``low_confidence`` (bottom quartile of train
    confidence), ``argmax_agree`` and ``tv_top_quartile``. ``factors`` is kept for diagnostics only.
    """
    if split not in SPLITS:
        raise ValueError("split must be one of %s, got %r" % (SPLITS, split))
    root = data_root(root)
    man = read_data_json("manifest.json", root)
    cutoff = man["low_confidence"]["cutoff"]
    tv_cut = (man.get("agreement") or {}).get("total_variation_top_quartile_cutoff")
    cases = _load_split_rows(root / "cache", "all", split)
    for c in cases:
        c["qmeta"] = {}
        for qid, q in c["questions"].items():
            g = c["gold"][qid]
            ag = c["label_agreement"].get(qid, {})
            crit = q.get("criteria")
            c["qmeta"][qid] = {
                "type": q["type"],
                "n_options": len(crit) if crit is not None else 0,
                "low_confidence": float(g.get("confidence", 1.0)) <= cutoff,
                "argmax_agree": bool(ag.get("argmax_agree", True)),
                "tv_top_quartile": tv_cut is not None and float(ag.get("total_variation", 0.0)) >= tv_cut,
            }
    return cases


# --------------------------------------------------------------------------- splits (T014)

def _shuffled(ids: Sequence[str], seed: Any, tag: str) -> List[str]:
    rng = random.Random("%s|%s" % (seed, tag))
    out = sorted(ids)
    rng.shuffle(out)
    return out


def _split_fingerprint(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _counts(n: int) -> Dict[str, int]:
    dev = int(round(SPLIT_FRACTIONS[0][1] * n))
    cal = int(round(SPLIT_FRACTIONS[1][1] * n))
    return {"dev": dev, "calibration": cal, "final": n - dev - cal}


def build_splits(test_cases: Sequence[Dict[str, Any]], seed: int, source_fingerprint: str) -> Dict[str, Any]:
    """Pure function: the split manifest for these test cases."""
    by_wf: Dict[str, List[str]] = {}
    for c in test_cases:
        by_wf.setdefault(c["workflow"], []).append(c["id"])
    ids: Dict[str, List[str]] = {s: [] for s in EVAL_SPLITS}
    wf_of = {c["id"]: c["workflow"] for c in test_cases}
    for wf in sorted(by_wf):
        order = _shuffled(by_wf[wf], seed, wf)
        n = _counts(len(order))
        ids["dev"] += order[:n["dev"]]
        ids["calibration"] += order[n["dev"]:n["dev"] + n["calibration"]]
        ids["final"] += order[n["dev"] + n["calibration"]:]

    def per_wf(members: Sequence[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in members:
            out[wf_of[i]] = out.get(wf_of[i], 0) + 1
        return dict(sorted(out.items()))

    splits = {s: {"case_ids": sorted(ids[s]), "n_cases": len(ids[s]), "n_per_workflow": per_wf(ids[s]),
                  "fingerprint": _split_fingerprint(ids[s])} for s in EVAL_SPLITS}

    half: Dict[str, List[str]] = {}
    for s in EVAL_SPLITS:
        picked: List[str] = []
        for wf in sorted(by_wf):
            members = [i for i in splits[s]["case_ids"] if wf_of[i] == wf]
            picked += _shuffled(members, seed, "half|%s|%s" % (s, wf))[:math.ceil(len(members) / 2)]
        half[s] = sorted(picked)

    by_id = {c["id"]: c for c in test_cases}
    latency: List[Dict[str, str]] = []
    variant_cases: List[str] = []
    for wf in sorted(by_wf):
        pool = [i for i in half["dev"] if wf_of[i] == wf]
        chosen = _shuffled(pool, seed, "latency|%s" % wf)[:LATENCY_PER_WORKFLOW]
        for k, cid in enumerate(chosen):
            qids = list(by_id[cid]["questions"])
            latency.append({"case_id": cid, "question_id": qids[k % len(qids)]})
        variant_cases += _shuffled(pool, seed, "variant|%s" % wf)[:VARIANT_CASES_PER_WORKFLOW]

    return {"schema_version": SCHEMA_VERSION, "rule": SPLIT_RULE, "seed": seed,
            "source_fingerprint": source_fingerprint, "splits": splits, "half_sample": half,
            "latency_sample": latency, "variant_sample": sorted(variant_cases)}


def make_splits(seed: int = DEFAULT_SPLIT_SEED, root: Optional[Path] = None) -> Dict[str, Any]:
    """Write ``splits.json``. Refuses to overwrite a manifest that would come out different."""
    root = data_root(root)
    man = read_data_json("manifest.json", root)
    body = build_splits(load_cases("test", root), seed, man["fingerprint"]["test"])
    path = root / "splits.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created_at"} != body:
            raise Refusal("fingerprint_mismatch",
                          "splits.json already exists and the new split (seed %s) differs; the split manifest is "
                          "frozen (FR-012). Delete experiments/data/splits.json only to start a new study." % seed)
        return old
    import datetime as _dt
    body = dict(body, created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    _write_json(path, body)
    return body


def check_splits(root: Optional[Path] = None) -> Dict[str, Any]:
    """Recompute the splits from the recorded seed and compare with ``splits.json``."""
    root = data_root(root)
    old = read_data_json("splits.json", root)
    man = read_data_json("manifest.json", root)
    fresh = build_splits(load_cases("test", root), old["seed"], man["fingerprint"]["test"])
    ok = {k: v for k, v in old.items() if k != "created_at"} == fresh
    if not ok:
        raise Refusal("fingerprint_mismatch", "splits.json does not match the source data and its recorded seed")
    return {"ok": True, "fingerprints": {s: old["splits"][s]["fingerprint"] for s in EVAL_SPLITS}}


def split_cases(split: str, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Upstream test cases that belong to ``dev``, ``calibration`` or ``final``."""
    if split not in EVAL_SPLITS:
        raise ValueError("split must be one of %s, got %r" % (EVAL_SPLITS, split))
    ids = set(read_data_json("splits.json", root)["splits"][split]["case_ids"])
    return [c for c in load_cases("test", root) if c["id"] in ids]


# --------------------------------------------------------------------------- original items (T014)

def serialize_state(state: Any) -> str:
    """Exactly what `laya.common.serialize_state` produces for a dict state."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def gold_block(case: Dict[str, Any], qid: str) -> Dict[str, Any]:
    g = case["gold"][qid]
    m = case["qmeta"][qid]
    return {"label": g["label"], "probabilities": g.get("probabilities"), "confidence": g.get("confidence"),
            "low_confidence": m["low_confidence"], "argmax_agree": m["argmax_agree"],
            "tv_top_quartile": m["tv_top_quartile"]}


def item_id(case_id: str, qid: str, variant: str, length: Any, extra: str = "") -> str:
    return "|".join([case_id, qid, variant, str(length)] + ([extra] if extra else []))


def original_items(split: str, root: Optional[Path] = None,
                   cases: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One ``original`` Evaluation Item per (case, question): the unwrapped case as the native path sees it."""
    cases = list(cases) if cases is not None else split_cases(split, root)
    items = []
    for c in cases:
        for qid, qdef in c["questions"].items():
            items.append({
                "item_id": item_id(c["id"], qid, "original", "original"),
                "case_id": c["id"], "question_id": qid, "split": split, "workflow": c["workflow"],
                "question_type": qdef["type"], "n_options": c["qmeta"][qid]["n_options"],
                "variant": "original", "length": "original",
                "state_text": serialize_state(c["state"]), "question": qdef,
                "layout": [{"kind": "target", "ref_case_id": c["id"]}],
                "evidence_span": None,   # the whole state is the evidence; nothing was added or cut
                "gold": gold_block(c, qid), "option_order": None,
                "construction": {"variant": "original"}, "status": "ok",
            })
    return items


def manifest_data_block(root: Optional[Path] = None, families_id: Optional[str] = None) -> Dict[str, Any]:
    """The ``data`` block of the run manifest (data-model.md "Data Manifest")."""
    man = read_data_json("manifest.json", root)
    block = {"source": man["source"], "dataset_revision": man["revision"], "fingerprints": man["fingerprint"],
             "low_confidence_cutoff": man["low_confidence"]["cutoff"], "families_id": families_id,
             "split_fingerprints": None, "split_seed": None}
    try:
        sp = read_data_json("splits.json", root)
        block["split_fingerprints"] = {s: sp["splits"][s]["fingerprint"] for s in EVAL_SPLITS}
        block["split_seed"] = sp["seed"]
    except FileNotFoundError:
        pass
    return block

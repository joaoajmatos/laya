"""Hand audit of generated items: pre-checks, sample sheet, recording, and the gate (T030).

Specs: specs/002-decision-benchmark-baselines (research.md R12; FR-014; SC-004).

Automatic pre-checks (exact length, span integrity, markers, label and identifier leakage) run
before the sheet is written, so the researcher's time goes to what only a person can judge:
does the added context change the correct answer or make it ambiguous?

``require_audit(families_id)`` is the gate `eval` uses: no baseline comparison runs until the audit
of the *current* families has passed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import data, families
from .results import Refusal

MIN_AUDIT_ITEMS = 50
DEFAULT_AUDIT_ITEMS = 60

RUBRIC = (
    "1. Do the added records or text change the correct answer to this question for the record under review?",
    "2. Do they make the answer ambiguous (two defensible answers where the original had one)?",
    "3. Is the record under review intact, complete and clearly marked?",
)


class AuditPrecheckError(ValueError):
    """An automatic pre-check failed; the sheet is not written."""


def precheck(agent, items: Sequence[Dict[str, Any]], cases_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    """Automatic checks over supported items. Returns failure messages (empty when all pass)."""
    from laya.common import encode_text
    from . import tokens

    failures: List[str] = []
    for it in items:
        if it.get("status") != "ok":
            continue
        iid, text, span = it["item_id"], it["state_text"], it["evidence_span"]
        case = cases_by_id[it["case_id"]]
        target = data.serialize_state(case["state"])
        length = it["length"]
        if isinstance(length, int):
            row = it["accounting"]["rows"][0]
            if row["final_length"] != length or row["truncated"]:
                failures.append("%s: row length %s is not the named %s" % (iid, row["final_length"], length))
        if text[span["char_start"]:span["char_end"]] != target:
            failures.append("%s: the target record text is not intact" % iid)
        if text.count(families.MARKER_TARGET) != 1:
            failures.append("%s: expected exactly one target marker" % iid)
        ids = encode_text(agent.tok, text, add_special_tokens=False)["input_ids"]
        tgt = encode_text(agent.tok, target, add_special_tokens=False)["input_ids"]
        if ids[span["state_start"]:span["state_end"]] != tgt:
            failures.append("%s: the evidence span does not cover exactly the target tokens" % iid)
        n_ref = text.count(families.MARKER_REFERENCE)
        if n_ref != len(it["construction"]["distractor_case_ids"]):
            failures.append("%s: %d reference markers but %d recorded references" % (iid, n_ref, len(it["construction"]["distractor_case_ids"])))
        bad = families.identifier_tokens(target)
        for block in text.split("\n\n"):
            if block.startswith(families.MARKER_REFERENCE) and bad and families.identifier_tokens(block) & bad:
                failures.append("%s: a reference record repeats an identifier of the target" % iid)
            if block.startswith(families.MARKER_NEUTRAL):
                words = set(block.split("\n", 1)[1].lower().split())
                labels = {str(it["gold"]["label"]).lower()} if it["question_type"] == "choice" else set()
                if words & labels:
                    failures.append("%s: neutral padding contains the gold label word" % iid)
        if any(train.startswith("train/") is False for train in it["construction"]["distractor_case_ids"]):
            failures.append("%s: a reference does not come from the training split" % iid)
    return failures


def audit_sample(agent, split_items: Sequence[Dict[str, Any]], cases_by_id: Dict[str, Dict[str, Any]],
                 families_id: str, n: int = DEFAULT_AUDIT_ITEMS, seed: int = 0,
                 root: Optional[Path] = None) -> Dict[str, Any]:
    """Draw a stratified sample (family variant × length × workflow) from dev items and write the sheet.

    Never fewer than 50 items (FR-014). Stops on a failed pre-check.
    """
    if n < MIN_AUDIT_ITEMS:
        raise ValueError("the audit needs at least %d items (FR-014), asked for %d" % (MIN_AUDIT_ITEMS, n))
    pool = [it for it in split_items if it.get("status") == "ok" and it["split"] == "dev"
            and it["variant"] in families.CONTEXT_VARIANTS]
    if len(pool) < n:
        raise ValueError("only %d audit-eligible dev items exist; %d requested" % (len(pool), n))
    failures = precheck(agent, pool, cases_by_id)
    if failures:
        raise AuditPrecheckError("%d pre-check failures, first: %s" % (len(failures), failures[0]))
    rng = random.Random("audit|%s|%s" % (seed, families_id))
    cells: Dict[Any, List[Dict[str, Any]]] = {}
    for it in pool:
        cells.setdefault((it["variant"], it["length"], it["workflow"]), []).append(it)
    for members in cells.values():
        rng.shuffle(members)
    keys = sorted(cells, key=str)
    rng.shuffle(keys)
    chosen: List[Dict[str, Any]] = []
    while len(chosen) < n:
        progressed = False
        for k in keys:
            if cells[k] and len(chosen) < n:
                chosen.append(cells[k].pop())
                progressed = True
        if not progressed:
            break
    sheet = {"schema_version": data.SCHEMA_VERSION, "families_id": families_id, "seed": seed, "n": len(chosen),
             "rubric": list(RUBRIC), "strata": ["variant", "length", "workflow"],
             "items": [_sheet_entry(it, cases_by_id[it["case_id"]]) for it in chosen]}
    root = data.data_root(root)
    data._write_json(root / "audit_sheet.json", sheet)
    (root / "audit_sheet.md").write_text(render_sheet(sheet), encoding="utf-8")
    return sheet


def _excerpt(block: str, n: int = 160) -> str:
    body = block if len(block) <= n else block[:n] + " ..."
    return body.replace("\n", " ")


def _sheet_entry(it: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    blocks = it["state_text"].split("\n\n")
    return {"item_id": it["item_id"], "variant": it["variant"], "length": it["length"], "workflow": it["workflow"],
            "question_id": it["question_id"], "instructions": it["question"].get("instructions"),
            "options": it["question"].get("criteria"), "gold_label": it["gold"]["label"],
            "gold_probabilities": it["gold"].get("probabilities"),
            "target_record": data.serialize_state(case["state"]),
            "layout": [b["kind"] for b in it["layout"]], "position": it.get("position"),
            "n_references": len(it["construction"]["distractor_case_ids"]),
            "context_excerpts": [_excerpt(b) for b in blocks if not b.startswith(families.MARKER_TARGET)][:8]}


def render_sheet(sheet: Dict[str, Any]) -> str:
    lines = ["# Hand audit sheet", "", "Families `%s`, %d items. For each item answer three questions in the draft "
             "result file (`changes_answer`, `ambiguous`, `evidence_intact`, `note`):" % (sheet["families_id"], sheet["n"]), ""]
    lines += ["- " + r for r in sheet["rubric"]] + [""]
    for i, e in enumerate(sheet["items"], 1):
        lines += ["## %d. `%s`" % (i, e["item_id"]),
                  "- variant `%s`, length `%s`, workflow `%s`, question `%s`" % (e["variant"], e["length"], e["workflow"], e["question_id"]),
                  "- question: %s" % e["instructions"], "- options: %s" % json.dumps(e["options"], ensure_ascii=False),
                  "- gold label: **%s** (probabilities %s)" % (e["gold_label"], json.dumps(e["gold_probabilities"])),
                  "- layout: %s (%d reference records)" % (" > ".join(e["layout"]), e["n_references"]),
                  "- record under review: `%s`" % e["target_record"], "- context excerpts:"]
        lines += ["  - %s" % x for x in e["context_excerpts"]] + [""]
    return "\n".join(lines)


def record_audit(draft_path: Path, root: Optional[Path] = None) -> Dict[str, Any]:
    """Ingest the researcher's verdicts; bind them to the sheet's `families_id`; compute the shares."""
    root = data.data_root(root)
    sheet = json.loads((root / "audit_sheet.json").read_text(encoding="utf-8"))
    draft = json.loads(Path(draft_path).read_text(encoding="utf-8"))
    entries = draft["items"] if isinstance(draft, dict) else draft
    meta = draft if isinstance(draft, dict) else {}
    by_id = {e["item_id"]: e for e in entries}
    missing = [e["item_id"] for e in sheet["items"] if e["item_id"] not in by_id]
    if missing:
        raise ValueError("the draft has no verdict for %d sheet items, first: %s" % (len(missing), missing[0]))
    rows = []
    for e in sheet["items"]:
        v = by_id[e["item_id"]]
        for key in ("changes_answer", "ambiguous", "evidence_intact"):
            if not isinstance(v.get(key), bool):
                raise ValueError("%s: %r must be true or false" % (e["item_id"], key))
        rows.append({"item_id": e["item_id"], "changes_answer": v["changes_answer"], "ambiguous": v["ambiguous"],
                     "evidence_intact": v["evidence_intact"], "note": v.get("note", "")})
    n = len(rows)
    changed = sum(r["changes_answer"] for r in rows) / n
    ambiguous = sum(r["ambiguous"] for r in rows) / n
    intact = all(r["evidence_intact"] for r in rows)
    result = {"schema_version": data.SCHEMA_VERSION, "families_id": sheet["families_id"], "n_audited": n,
              "items": rows, "share_answer_changed": changed, "share_ambiguous": ambiguous,
              "all_evidence_intact": intact,
              # Who judged, and how, is part of the result: an audit by an AI assistant must never read as a human one.
              "auditor": str(meta.get("auditor", "researcher")), "method": str(meta.get("method", "")),
              "passed": bool(n >= MIN_AUDIT_ITEMS and changed == 0.0 and ambiguous == 0.0 and intact)}
    data._write_json(root / "audit_result.json", result)
    return result


def require_audit(families_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """The gate: a passing audit of exactly these families must exist (research.md R12)."""
    path = data.data_root(root) / "audit_result.json"
    if not path.exists():
        raise Refusal("audit_required", "no audit_result.json; run audit-sample, review the sheet and run audit-record")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("families_id") != families_id:
        raise Refusal("audit_stale", "the audit covers families %s, not the current %s; rebuild and audit again"
                      % (result.get("families_id"), families_id))
    if not result.get("passed"):
        raise Refusal("audit_required", "the audit of families %s did not pass (answer changes %.1f%%, ambiguous %.1f%%)"
                      % (families_id, 100 * result.get("share_answer_changed", 0), 100 * result.get("share_ambiguous", 0)))
    return result

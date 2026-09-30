"""T023: hand-audit sheet, recording and the gate (experiments/audit_items.py)."""
import json

import pytest

from experiments import audit_items as A
from experiments import data as D
from experiments import families as F
from experiments.results import Refusal

VARIANTS = ("neutral@mid", "distractor@begin", "distractor@mid", "distractor@end")
LENGTHS = (256, 512, 1024)


@pytest.fixture(scope="module")
def built(tiny_agent, tiny_upstream, tmp_path_factory):
    root = tmp_path_factory.mktemp("audit-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    meta = F.build_split(tiny_agent, "dev", lengths=LENGTHS, variants=VARIANTS, seed=0, root=root, tokenizer_name="tiny")
    items = list(F.read_items(F.items_dir(meta["families_id"], root) / "dev.jsonl.gz"))
    cases = {c["id"]: c for c in D.load_cases("test", root)}
    return {"root": root, "fid": meta["families_id"], "items": items, "cases": cases}


def test_prechecks_pass_on_good_items_and_catch_damage(tiny_agent, built):
    assert A.precheck(tiny_agent, built["items"], built["cases"]) == []
    it = dict(next(i for i in built["items"] if i["status"] == "ok" and i["variant"] == "distractor@mid"))
    broken = dict(it, state_text=it["state_text"].replace(F.MARKER_TARGET, "[X]", 1))
    assert any("target marker" in f or "intact" in f for f in A.precheck(tiny_agent, [broken], built["cases"]))
    leaky = dict(it, construction=dict(it["construction"], distractor_case_ids=["test/x"] + it["construction"]["distractor_case_ids"][1:]))
    assert any("training split" in f for f in A.precheck(tiny_agent, [leaky], built["cases"]))


def test_sample_is_stratified_seeded_and_at_least_fifty(tiny_agent, built):
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=60, seed=1, root=built["root"])
    assert sheet["n"] == 60 and sheet["families_id"] == built["fid"]
    ids = [e["item_id"] for e in sheet["items"]]
    assert len(set(ids)) == 60
    assert {e["variant"] for e in sheet["items"]} == set(VARIANTS)
    assert {e["length"] for e in sheet["items"]} == set(LENGTHS) - {256} | ({256} if any(e["length"] == 256 for e in sheet["items"]) else set())
    assert {e["workflow"] for e in sheet["items"]} >= {"agent_trace_observability", "customer_service", "invoice_processing"}
    again = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=60, seed=1, root=built["root"])
    assert [e["item_id"] for e in again["items"]] == ids                                   # seeded and recorded
    with pytest.raises(ValueError):
        A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=49, root=built["root"])


def test_sheet_lists_target_gold_layout_and_rubric(tiny_agent, built):
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=50, root=built["root"])
    e = sheet["items"][0]
    assert {"target_record", "gold_label", "layout", "instructions", "options", "context_excerpts"} <= set(e)
    assert len(sheet["rubric"]) == 3
    md = (built["root"] / "audit_sheet.md").read_text(encoding="utf-8")
    assert "record under review" in md and e["item_id"] in md and "ambiguous" in md


def test_only_dev_items_are_eligible(tiny_agent, built):
    others = [dict(i, split="calibration") for i in built["items"]]
    with pytest.raises(ValueError):
        A.audit_sample(tiny_agent, others, built["cases"], built["fid"], n=50, root=built["root"])


def _draft(sheet, tmp, overrides=None):
    rows = [{"item_id": e["item_id"], "changes_answer": False, "ambiguous": False, "evidence_intact": True,
             "note": ""} for e in sheet["items"]]
    for i, patch in (overrides or {}).items():
        rows[i].update(patch)
    path = tmp / "audit_result_draft.json"
    path.write_text(json.dumps({"items": rows}), encoding="utf-8")
    return path


def test_record_computes_shares_binds_to_families_and_sets_passed(tiny_agent, built, tmp_path):
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=60, root=built["root"])
    ok = A.record_audit(_draft(sheet, tmp_path), root=built["root"])
    assert ok["passed"] is True and ok["share_answer_changed"] == 0.0 and ok["families_id"] == built["fid"]
    assert A.require_audit(built["fid"], root=built["root"])["passed"]
    bad = A.record_audit(_draft(sheet, tmp_path, {3: {"changes_answer": True}}), root=built["root"])
    assert bad["passed"] is False and bad["share_answer_changed"] == pytest.approx(1 / 60)
    with pytest.raises(Refusal) as err:
        A.require_audit(built["fid"], root=built["root"])
    assert err.value.reason == "audit_required"
    amb = A.record_audit(_draft(sheet, tmp_path, {0: {"ambiguous": True}}), root=built["root"])
    assert amb["passed"] is False and amb["share_ambiguous"] == pytest.approx(1 / 60)


def test_missing_and_stale_audits_are_refused(tiny_agent, built, tmp_path):
    import os
    result = built["root"] / "audit_result.json"
    if result.exists():
        os.remove(result)
    with pytest.raises(Refusal) as err:
        A.require_audit(built["fid"], root=built["root"])
    assert err.value.reason == "audit_required"
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=50, root=built["root"])
    A.record_audit(_draft(sheet, tmp_path), root=built["root"])
    with pytest.raises(Refusal) as err:
        A.require_audit("0123456789abcdef", root=built["root"])
    assert err.value.reason == "audit_stale"


def test_a_draft_must_cover_every_sheet_item_with_booleans(tiny_agent, built, tmp_path):
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=50, root=built["root"])
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"items": [{"item_id": sheet["items"][0]["item_id"], "changes_answer": False,
                                            "ambiguous": False, "evidence_intact": True}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        A.record_audit(short, root=built["root"])
    rows = [{"item_id": e["item_id"], "changes_answer": "no", "ambiguous": False, "evidence_intact": True}
            for e in sheet["items"]]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        A.record_audit(bad, root=built["root"])


def test_the_auditor_and_method_are_recorded_and_reach_the_report(tiny_agent, built, tmp_path):
    from experiments import report2
    sheet = A.audit_sample(tiny_agent, built["items"], built["cases"], built["fid"], n=50, root=built["root"])
    rows = [{"item_id": e["item_id"], "changes_answer": False, "ambiguous": False, "evidence_intact": True, "note": ""}
            for e in sheet["items"]]
    path = tmp_path / "ai_draft.json"
    path.write_text(json.dumps({"auditor": "an AI assistant (Claude)", "method": "checks, not a full read", "items": rows}), encoding="utf-8")
    res = A.record_audit(path, root=built["root"])
    assert res["auditor"] == "an AI assistant (Claude)" and res["method"] == "checks, not a full read" and res["passed"]
    texts = " ".join(s["text"] for s in report2.section_audit(built["root"])["statements"])
    assert "by an AI assistant (Claude)" in texts and "not by a human reviewer" in texts and "repeated by a person" in texts
    plain = A.record_audit(_draft(sheet, tmp_path), root=built["root"])         # no metadata: the researcher
    assert plain["auditor"] == "researcher"
    assert "not by a human" not in " ".join(s["text"] for s in report2.section_audit(built["root"])["statements"])

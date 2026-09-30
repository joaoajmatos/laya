"""T022: controlled long-context families (experiments/families.py, experiments/data.py)."""
import json
import random

import pytest

from experiments import data as D
from experiments import families as F
from experiments import tokens as T
from experiments.bm25 import BM25

LENGTHS = (256, 512, 1024)
SHORT_WF = "customer_service"
LONG_WF = "security_incidents"


@pytest.fixture(scope="module")
def env(tiny_upstream, tmp_path_factory):
    root = tmp_path_factory.mktemp("families-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    return {"root": root, "test": D.load_cases("test", root), "train": D.load_cases("train", root)}


@pytest.fixture(scope="module")
def pool(env):
    return F.ReferencePool(env["train"])


def case_of(env, wf, k=0):
    return [c for c in env["test"] if c["workflow"] == wf][k]


def build(agent, env, pool, case, qid, variant, length, seed=0):
    refs = pool.ordered(case, seed)
    return F.build_item(agent, case, qid, variant, length, "dev", refs, F.neutral_words(agent), seed)


# --------------------------------------------------------------------------- exact length, span, target unchanged

@pytest.mark.parametrize("variant", F.CONTEXT_VARIANTS)
@pytest.mark.parametrize("length", LENGTHS)
def test_every_row_reaches_its_named_length_exactly(tiny_agent, env, pool, variant, length):
    for wf in (SHORT_WF, "invoice_processing"):
        case = case_of(env, wf)
        for qid in case["questions"]:
            it = build(tiny_agent, env, pool, case, qid, variant, length)
            assert it["status"] == "ok", it.get("reason")
            rec = T.account(tiny_agent, it["state_text"], {qid: it["question"]}, max_len=length, requested_total=length)
            assert rec["rows"][0]["final_length"] == length and not rec["rows"][0]["truncated"]
            assert it["accounting"]["rows"][0]["final_length"] == length


def test_a_miss_raises_instead_of_recording_a_wrong_length(tiny_agent, env, pool, monkeypatch):
    def miss(*a, **k):
        raise T.TokenAccountingError("requested 512 tokens, model saw 511")
    monkeypatch.setattr(F.tokens, "account", miss)
    case = case_of(env, SHORT_WF)
    with pytest.raises(T.TokenAccountingError):
        build(tiny_agent, env, pool, case, "action", "distractor@mid", 512)


def test_target_record_is_unchanged_byte_for_byte(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    for variant in F.CONTEXT_VARIANTS:
        it = build(tiny_agent, env, pool, case, "risk", variant, 512)
        span = it["evidence_span"]
        assert it["state_text"][span["char_start"]:span["char_end"]] == D.serialize_state(case["state"])
        assert it["state_text"].count(F.MARKER_TARGET) == 1


@pytest.mark.parametrize("variant", F.CONTEXT_VARIANTS)
def test_evidence_span_marks_exactly_the_target_tokens(tiny_agent, env, pool, variant):
    from laya.common import encode_text
    case = case_of(env, SHORT_WF, 1)
    qid = "outcome"
    it = build(tiny_agent, env, pool, case, qid, variant, 512)
    sp = it["evidence_span"]
    state_ids = encode_text(tiny_agent.tok, it["state_text"], add_special_tokens=False)["input_ids"]
    target_ids = encode_text(tiny_agent.tok, D.serialize_state(case["state"]), add_special_tokens=False)["input_ids"]
    assert state_ids[sp["state_start"]:sp["state_end"]] == target_ids
    row = tiny_agent._encode_state(it["state_text"], [qid], {qid: tiny_agent._to_internal(it["question"])},
                                   max_len=512)[0]["ids"]
    assert row[sp["row_start"]:sp["row_end"]] == target_ids                 # the row positions agree too
    assert sp["row_end"] - sp["row_start"] == sp["state_end"] - sp["state_start"]


# --------------------------------------------------------------------------- markers, layout, references

def test_markers_and_layout(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    it = build(tiny_agent, env, pool, case, "action", "distractor@mid", 1024)
    kinds = [b["kind"] for b in it["layout"]]
    n_refs = len(it["construction"]["distractor_case_ids"])
    assert kinds.count("reference") == n_refs >= 2 and kinds.count("target") == 1
    assert it["state_text"].count(F.MARKER_REFERENCE) == n_refs
    assert it["state_text"].count(F.MARKER_NEUTRAL) == kinds.count("neutral") >= 1
    assert it["construction"]["marker_constants"] == F.MARKERS
    assert it["state_text"].index(F.MARKER_TARGET) < it["state_text"].index(D.serialize_state(case["state"]))


def test_references_are_train_only_same_workflow_and_never_the_target(tiny_agent, env, pool):
    train_ids = {"train/%s" % c["id"]: c["workflow"] for c in env["train"]}
    for wf in D.WORKFLOWS:
        case = case_of(env, wf)
        it = build(tiny_agent, env, pool, case, "urgency", "distractor@mid", 1024)
        ids = it["construction"]["distractor_case_ids"]
        assert len(ids) == len(set(ids))                                   # never twice in one item
        for r in ids:
            assert r in train_ids and train_ids[r] == wf                  # train split, same workflow
            assert r.startswith("train/")                                  # a separate namespace from test ids
            assert D.serialize_state(next(c for c in env["train"] if "train/%s" % c["id"] == r)["state"])                 != D.serialize_state(case["state"])                        # never the target's own text


def test_reference_order_comes_from_the_top_ranked_group_first(env, pool):
    case = case_of(env, SHORT_WF)
    ordered = pool.ordered(case, 0)
    ranked = BM25(pool.text[case["workflow"]]).rank(D.serialize_state(case["state"]))
    top = {pool.by_workflow[case["workflow"]][i]["id"] for i in ranked[:F.REFERENCE_GROUP]}
    assert {r["case"]["id"] for r in ordered[:F.REFERENCE_GROUP]} == top
    assert len({r["case"]["id"] for r in ordered}) == len(ordered)


def test_identifier_sharing_and_duplicate_records_are_excluded():
    train = [{"id": "a", "workflow": "w", "state": {"ref": "INV-12345", "x": "alpha"}},
             {"id": "b", "workflow": "w", "state": {"ref": "INV-99999", "x": "alpha beta"}},
             {"id": "c", "workflow": "w", "state": {"ref": "INV-12345", "x": "alpha gamma"}}]
    target = {"id": "t", "workflow": "w", "state": {"ref": "INV-12345", "x": "alpha"}}
    ids = [r["case"]["id"] for r in F.ReferencePool(train).ordered(target, 0)]
    assert ids == ["b"]                                       # a repeats the text, c the identifier


# --------------------------------------------------------------------------- positions

def test_positions_begin_mid_end(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    its = {v: build(tiny_agent, env, pool, case, "action", v, 1024) for v in F.POSITION_VARIANTS}
    assert its["distractor@begin"]["position"]["start_fraction"] <= 0.05
    assert its["distractor@end"]["position"]["end_fraction"] >= 0.95
    mid = its["distractor@mid"]["position"]
    assert 0.25 <= (mid["start_fraction"] + mid["end_fraction"]) / 2 <= 0.75
    sets = [tuple(sorted(i["construction"]["distractor_case_ids"])) for i in its.values()]
    assert len(set(sets)) == 1                                # the same distractor set, only the order differs
    kinds = {v: [b["kind"] for b in i["layout"]] for v, i in its.items()}
    assert kinds["distractor@begin"][0] == "target" and kinds["distractor@begin"][-1] == "neutral"
    assert kinds["distractor@end"][-1] == "target" and kinds["distractor@end"][0] == "neutral"


def test_neutral_variant_puts_padding_around_the_target(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    it = build(tiny_agent, env, pool, case, "action", "neutral@mid", 1024)
    assert [b["kind"] for b in it["layout"]] == ["neutral", "target", "neutral"]
    assert it["construction"]["distractor_case_ids"] == []
    assert 0.3 <= (it["position"]["start_fraction"] + it["position"]["end_fraction"]) / 2 <= 0.7


def test_reference_sets_are_nested_across_lengths_and_questions(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    small = build(tiny_agent, env, pool, case, "action", "distractor@mid", 512)["construction"]["distractor_case_ids"]
    large = build(tiny_agent, env, pool, case, "action", "distractor@mid", 1024)["construction"]["distractor_case_ids"]
    other_q = build(tiny_agent, env, pool, case, "urgency", "distractor@mid", 1024)["construction"]["distractor_case_ids"]
    assert large[:len(small)] == small                                    # a longer length extends the shorter set
    n = min(len(large), len(other_q))
    assert large[:n] == other_q[:n]                                       # questions draw from one ordered sequence


# --------------------------------------------------------------------------- options, leakage

def test_choice_options_are_permuted_and_other_types_are_not(tiny_agent, env, pool):
    positions = set()
    for k in range(10):
        case = case_of(env, SHORT_WF, k)
        orig = case["questions"]["action"]
        it = build(tiny_agent, env, pool, case, "action", "distractor@mid", 512)
        assert sorted(it["question"]["criteria"]) == sorted(orig["criteria"])
        assert it["option_order"] is not None and sorted(it["option_order"]) == [0, 1, 2]
        assert list(it["question"]["criteria"]) == [list(orig["criteria"])[i] for i in it["option_order"]]
        assert it["gold"]["label"] in it["question"]["criteria"]           # labels are keys: no remapping needed
        positions.add(list(it["question"]["criteria"]).index(it["gold"]["label"]))
        same = build(tiny_agent, env, pool, case, "action", "distractor@begin", 1024)
        assert same["option_order"] == it["option_order"]                 # one order across variants and lengths
    assert len(positions) > 1                                              # the gold option moves around
    case = case_of(env, SHORT_WF)
    for qid in ("needs_review", "risk", "urgency"):
        it = build(tiny_agent, env, pool, case, qid, "distractor@mid", 512)
        assert it["option_order"] is None and it["question"] == case["questions"][qid]


def test_neutral_padding_uses_no_workflow_words_and_gold_is_independent_of_length(tiny_agent, env, pool):
    forbidden = {"refund", "urgent", "invoice", "order", "customer", "payment", "account", "support", "ticket",
                 "issue", "problem", "status", "delivery", "report", "document", "request", "update", "incident",
                 "security", "agent", "trace", "attack", "alert", "review", "risk"}
    assert not (set(F.NEUTRAL_VOCAB) & forbidden)
    from experiments import inputs
    assert "refund" in inputs.FILLER_WORDS and "urgent" in inputs.FILLER_WORDS        # why Phase 1's list is not reused
    case = case_of(env, SHORT_WF)
    golds = []
    for length in LENGTHS:
        it = build(tiny_agent, env, pool, case, "outcome", "neutral@mid", length)
        body = it["state_text"]
        pad_words = set()
        for chunk in body.split("\n\n"):
            if chunk.startswith(F.MARKER_NEUTRAL):
                pad_words |= set(chunk.split("\n", 1)[1].split())
        assert pad_words and pad_words <= set(F.NEUTRAL_VOCAB)
        golds.append(json.dumps(it["gold"], sort_keys=True))
    assert len(set(golds)) == 1                                                 # padding never touches the label


# --------------------------------------------------------------------------- unsupported, oracle, determinism

def test_target_longer_than_the_named_length_is_unsupported_not_truncated(tiny_agent, env, pool):
    case = case_of(env, LONG_WF)
    it = build(tiny_agent, env, pool, case, "action", "distractor@mid", 128)
    assert it["status"] == "unsupported" and it["reason"].startswith("target_exceeds_length")
    assert it["state_text"] is None and it["evidence_span"] is None
    ok = build(tiny_agent, env, pool, case, "action", "distractor@mid", 1024)
    assert ok["status"] == "ok"


def test_positional_capacity_is_respected(tiny_agent, env, pool):
    it = build(tiny_agent, env, pool, case_of(env, SHORT_WF), "action", "distractor@mid", 4096)
    assert it["status"] == "unsupported" and "positional capacity" in it["reason"]


def test_oracle_is_the_target_with_only_its_marker(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    it = F.build_item(tiny_agent, case, "risk", "oracle", "original", "dev", [], F.neutral_words(tiny_agent))
    assert it["state_text"] == F.MARKER_TARGET + "\n" + D.serialize_state(case["state"])
    assert it["layout"][0]["kind"] == "target" and it["construction"]["distractor_case_ids"] == []
    assert it["evidence_span"]["char_start"] == len(F.MARKER_TARGET) + 1


def test_items_are_deterministic_for_a_seed(tiny_agent, env, pool):
    case = case_of(env, SHORT_WF)
    a = build(tiny_agent, env, pool, case, "action", "distractor@mid", 512, seed=5)
    b = build(tiny_agent, env, pool, case, "action", "distractor@mid", 512, seed=5)
    c = build(tiny_agent, env, pool, case, "action", "distractor@mid", 512, seed=6)
    assert F.canonical_line(a) == F.canonical_line(b)
    assert F.canonical_line(a) != F.canonical_line(c)


# --------------------------------------------------------------------------- files and splits

def test_split_files_are_disjoint_fingerprinted_and_inherit_their_split(tiny_agent, env):
    root = env["root"]
    kw = dict(lengths=(256, 512), variants=("distractor@mid", "oracle"), seed=0, root=root, tokenizer_name="tiny")
    F.build_split(tiny_agent, "dev", **kw)
    F.build_split(tiny_agent, "calibration", **kw)
    meta = F.build_split(tiny_agent, "final", **kw)
    fid = meta["families_id"]
    seen = {}
    for split in D.EVAL_SPLITS:
        items = list(F.read_items(F.items_dir(fid, root) / ("%s.jsonl.gz" % split)))
        assert items and all(i["split"] == split for i in items)
        seen[split] = {i["case_id"] for i in items}
    assert not (seen["dev"] & seen["calibration"]) and not (seen["dev"] & seen["final"])
    assert not (seen["calibration"] & seen["final"])
    assert meta["splits"]["dev"]["n_items"] == 12 * 5 * 3         # 12 dev cases x 5 questions x (2 lengths + oracle)
    assert all(len(meta["splits"][s]["fingerprint"]) == 64 for s in D.EVAL_SPLITS)
    again = F.build_split(tiny_agent, "dev", **kw)
    assert again["splits"]["dev"]["fingerprint"] == meta["splits"]["dev"]["fingerprint"]   # reproducible
    other = F.build_split(tiny_agent, "dev", **dict(kw, seed=1))
    assert other["families_id"] != fid


def test_every_supported_item_records_an_evidence_span(tiny_agent, env):
    kw = dict(lengths=(512,), variants=("neutral@mid", "distractor@begin"), seed=0, root=env["root"], tokenizer_name="tiny")
    meta = F.build_split(tiny_agent, "dev", **kw)
    items = list(F.read_items(F.items_dir(meta["families_id"], env["root"]) / "dev.jsonl.gz"))
    ok = [i for i in items if i["status"] == "ok"]
    assert ok and all(i["evidence_span"] and i["accounting"] for i in ok)
    assert all(i["reason"] for i in items if i["status"] != "ok")


def test_tiny_room_around_the_target_never_raises_and_never_records_a_wrong_length(tiny_agent, env, pool):
    """Regression (2026-09-29): a real final-split case with 12 tokens of room broke the exact-length fit."""
    case = case_of(env, SHORT_WF)
    qid = "action"
    qdef, _ = F.permuted_question(case["questions"][qid], case["id"], qid, 0)
    fixed = F.fixed_row_tokens(tiny_agent, qdef)
    target_cost = F.count_tokens(tiny_agent, F._block(F.MARKER_TARGET, D.serialize_state(case["state"])))
    words = F.neutral_words(tiny_agent)
    outcomes = set()
    for room in range(0, 40):
        for variant in F.CONTEXT_VARIANTS:
            it = F.build_item(tiny_agent, case, qid, variant, fixed + target_cost + room, "dev",
                              pool.ordered(case, 0), words, 0)
            outcomes.add((variant, it["status"]))
            if it["status"] == "ok":
                length = fixed + target_cost + room
                assert it["accounting"]["rows"][0]["final_length"] == length
                assert it["evidence_span"]["char_end"] > it["evidence_span"]["char_start"]
            else:
                assert it["reason"] and it["state_text"] is None
    assert ("neutral@mid", "ok") in outcomes and ("neutral@mid", "unsupported") in outcomes

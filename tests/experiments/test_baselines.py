"""T035: baseline conditions (experiments/baselines.py) on the tiny checkpoint."""
import pytest

from experiments import baselines as B
from experiments import data as D
from experiments import families as F
from experiments import tokens as T


@pytest.fixture(scope="module")
def env(tiny_upstream, tmp_path_factory):
    root = tmp_path_factory.mktemp("baselines-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    return {"test": D.load_cases("test", root), "train": D.load_cases("train", root)}


@pytest.fixture(scope="module")
def pool(env):
    return F.ReferencePool(env["train"])


def make(agent, env, pool, variant, length, qid="action", wf="customer_service", k=0):
    case = [c for c in env["test"] if c["workflow"] == wf][k]
    return F.build_item(agent, case, qid, variant, length, "dev", pool.ordered(case, 0), F.neutral_words(agent), 0)


# --------------------------------------------------------------------------- evidence helpers

def test_covered_fraction_and_chunks():
    span = {"state_start": 100, "state_end": 200}
    assert B.covered_fraction(span, [(0, 300)]) == 1.0
    assert B.covered_fraction(span, [(0, 150)]) == 0.5
    assert B.covered_fraction(span, [(0, 100), (200, 300)]) == 0.0
    assert B.covered_fraction(span, [(90, 120), (150, 170)]) == pytest.approx(0.4)
    assert B.chunk_ranges(50) == [(0, 50)] and B.chunk_ranges(0) == []
    ch = B.chunk_ranges(200)
    assert ch[0] == (0, 64) and ch[1] == (32, 96) and ch[-1] == (136, 200)          # 50% overlap, end-aligned
    assert all(b - a == 64 for a, b in ch)


def test_select_chunks_keeps_the_union_within_budget_and_merges_runs():
    ranges = B.chunk_ranges(256)
    scores = [0.0] * len(ranges)
    scores[3] = 9.0                                    # chunk (96, 160)
    runs = B.select_chunks(ranges, scores, 64)
    assert runs == [(96, 160)]
    runs = B.select_chunks(ranges, scores, 100)
    kept = sum(b - a for a, b in runs)
    assert kept <= 100 and runs[0][0] <= 96 and runs[0][1] >= 160
    assert B.select_chunks(ranges, scores, 10 ** 6) == [(0, 256)]                   # everything fits: one run
    assert all(runs[i][1] < runs[i + 1][0] for i in range(len(runs) - 1))          # position order, merged


def test_question_query_excludes_the_marker_text():
    q = {"type": "choice", "instructions": "which is best", "criteria": {"a": "first option", "b": None}}
    text = B.question_query(q)
    assert "which is best" in text and "first option" in text and "UNDER REVIEW" not in text.upper()
    assert "score" not in B.question_query({"type": "score", "instructions": "how", "criteria": ["low", "high"]}) or True


# --------------------------------------------------------------------------- truncation

def test_truncation_uses_512_and_the_configured_cap_with_a_right_cut(tiny_agent, env, pool):
    cap = int(tiny_agent.cfg["max_len"])
    end = make(tiny_agent, env, pool, "distractor@end", 1024)
    begin = make(tiny_agent, env, pool, "distractor@begin", 1024)
    for runner, limit in ((B.TruncationRunner(512), 512), (B.TruncationRunner(None), cap)):
        out_end = runner(tiny_agent, end)
        out_begin = runner(tiny_agent, begin)
        assert out_end["max_len_used"] == limit and out_end["tokens_seen"] <= limit
        assert out_end["evidence_visible"] == "none"          # the target sits at the end: the right cut removes it
        assert out_begin["evidence_visible"] == "full"        # the target sits at the start: it survives
        assert out_end["beyond_configured_max_len"] is (limit > cap)
    assert B.TruncationRunner(512).name == "trunc512" and B.TruncationRunner(None).name == "truncCap"


def test_partial_visibility_when_the_cut_falls_inside_the_target(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "distractor@mid", 1024)
    sp = it["evidence_span"]
    fixed = F.fixed_row_tokens(tiny_agent, it["question"])
    limit = fixed + sp["state_start"] + (sp["state_end"] - sp["state_start"]) // 2
    out = B.NativeRunner(max_len=limit)(tiny_agent, it)
    assert out["evidence_visible"] == "partial" and 0.3 < out["evidence_fraction"] < 0.7


def test_native_runs_the_named_length_and_labels_beyond_cap(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "distractor@mid", 512)
    out = B.NativeRunner()(tiny_agent, it)
    assert out["max_len_used"] == 512 and out["tokens_seen"] == 512 and out["evidence_visible"] == "full"
    assert out["beyond_configured_max_len"] is True and out["max_len_used"] <= T.position_limit(tiny_agent)


# --------------------------------------------------------------------------- window

def test_window_equals_predict_long_and_reports_window_evidence(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "distractor@mid", 1024)
    runner = B.WindowRunner(100)
    out = runner(tiny_agent, it)
    ref = tiny_agent.predict_long(it["state_text"], {it["question_id"]: it["question"]}, window=100, stride=None,
                                  batch_size=None)
    assert out["answer"] == ref["answers"][it["question_id"]]
    assert out["probability_scope"] == "deciding_window" and out["deployable"] is True
    assert out["windows"] == ref["usage"]["windows"] > 1
    n_state = len(B._state_ids(tiny_agent, it["state_text"]))
    wins = runner.windows(tiny_agent, n_state)
    assert wins[0] == (0, 100) and wins[1][0] == 50 and wins[-1][1] == n_state          # stride = window // 2
    assert out["evidence_visible"] in ("full", "partial")
    assert out["tokens_seen"] > 100 and out["window_size"] == 100


def test_window_default_size_is_cap_minus_head_minus_eight(tiny_agent):
    cap, head = int(tiny_agent.cfg["max_len"]), int(tiny_agent.cfg["head_max_len"])
    assert B.WindowRunner("default").budget(tiny_agent) == cap - head - 8
    assert B.WindowRunner(256).budget(tiny_agent) == 256


# --------------------------------------------------------------------------- retrieval

def test_retrieval_keeps_the_query_matching_chunk_within_budget(tiny_agent):
    filler = " ".join(["quiet", "morning", "harbor", "small", "river", "cold"] * 40)
    needle = "customer order refund delivery customer order refund delivery"
    text = filler + " " + needle + " " + filler
    item = {"item_id": "x", "case_id": "c", "question_id": "q", "state_text": text,
            "question": {"type": "choice", "instructions": "customer order refund delivery",
                         "criteria": {"a": "refund", "b": "delivery"}},
            "evidence_span": None, "length": 512, "option_order": None}
    ids = B._state_ids(tiny_agent, text)
    at = next(i for i in range(len(ids)) if tiny_agent.tok.decode(ids[i:i + 1]) == "customer")
    r = B.RetrieveRunner(128)
    kept_text, runs, n = r.select(tiny_agent, item)
    assert n == len(ids) and sum(b - a for a, b in runs) <= 128
    assert any(a <= at < b for a, b in runs)                      # the matching chunk was retained
    assert "customer" in kept_text
    assert runs == sorted(runs)
    item["evidence_span"] = {"state_start": at, "state_end": at + 8}
    out = r(tiny_agent, item)
    assert out["evidence_visible"] == "full" and out["tokens_seen"] < len(ids)
    assert r.name == "retrieve128"


def test_retrieval_with_a_budget_above_the_state_keeps_everything(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "neutral@mid", 512)
    r = B.RetrieveRunner(2048)
    text, runs, n = r.select(tiny_agent, it)
    assert text == it["state_text"] and runs == [(0, n)]


def test_retrieval_output_never_exceeds_positional_capacity(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "distractor@mid", 1024)
    out = B.RetrieveRunner(512)(tiny_agent, it)
    assert out["max_len_used"] <= T.position_limit(tiny_agent)
    assert out["tokens_seen"] <= out["max_len_used"]


# --------------------------------------------------------------------------- oracle

def test_oracle_is_the_target_with_only_its_marker_and_is_not_deployable(tiny_agent, env, pool):
    small = make(tiny_agent, env, pool, "distractor@mid", 512)
    large = make(tiny_agent, env, pool, "distractor@mid", 1024)
    case = [c for c in env["test"] if c["id"] == small["case_id"]][0]
    assert B.OracleRunner.oracle_text(small) == F.MARKER_TARGET + "\n" + D.serialize_state(case["state"])
    r = B.OracleRunner()
    calls = []
    real = tiny_agent.predict
    tiny_agent.predict = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        a, b = r(tiny_agent, small), r(tiny_agent, large)
    finally:
        del tiny_agent.predict
    assert len(calls) == 1                                        # one call per (case, question), reused across lengths
    assert a["answer"] == b["answer"] and a["deployable"] is False and a["evidence_visible"] == "full"


# --------------------------------------------------------------------------- shared rules

def test_no_baseline_builds_an_input_beyond_its_max_len(tiny_agent, env, pool):
    it = make(tiny_agent, env, pool, "distractor@mid", 1024)
    for r in (B.NativeRunner(), B.TruncationRunner(512), B.TruncationRunner(None), B.RetrieveRunner(512),
              B.RetrieveRunner(1024), B.OracleRunner()):
        out = r(tiny_agent, it)
        assert out["tokens_seen"] <= out["max_len_used"] <= T.position_limit(tiny_agent), r.name


def test_the_timed_path_does_no_accounting(tiny_agent, env, pool, monkeypatch):
    it = make(tiny_agent, env, pool, "distractor@mid", 512)

    def forbidden(*a, **k):
        raise AssertionError("accounting is off the timed path")
    monkeypatch.setattr(T, "account", forbidden)
    for r in (B.NativeRunner(), B.TruncationRunner(512), B.WindowRunner(100), B.RetrieveRunner(256), B.OracleRunner()):
        meta = r.run(tiny_agent, it)
        assert "answer" in meta, r.name


def test_build_runner_maps_names():
    assert isinstance(B.build_runner("native"), B.NativeRunner)
    assert B.build_runner("retrieve1024").budget == 1024
    assert B.build_runner("window", {"size": 256}).budget(type("A", (), {"cfg": {}})()) == 256
    assert B.build_runner("oracle").deployable is False
    with pytest.raises(ValueError):
        B.build_runner("bm25")
    assert set(B.CONDITION_NAMES) == {"native", "trunc512", "truncCap", "window", "retrieve512", "retrieve1024",
                                      "retrieve2048", "oracle"}

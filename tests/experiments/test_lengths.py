"""T011: length profile (experiments/lengths.py) on the tiny checkpoint and synthetic upstream data."""
import json

import numpy as np
import pytest

from experiments import data as D
from experiments import lengths as Lg
from experiments import tokens


@pytest.fixture(scope="module")
def cases(tiny_upstream, tmp_path_factory):
    root = tmp_path_factory.mktemp("lengths-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    return root, D.load_cases("test", root)


def test_stats_match_a_hand_computation():
    s = Lg.stats([10, 20, 30, 40, 50])
    assert (s["min"], s["median"], s["max"], s["n"]) == (10.0, 30.0, 50.0, 5)
    assert s["p90"] == pytest.approx(46.0) and s["p99"] == pytest.approx(49.6)
    assert Lg.stats([])["median"] is None


def test_fit_share_is_the_share_at_or_below_each_length():
    sh = Lg.fit_share([100, 500, 512, 513, 3000], (512, 1024, 4096))
    assert sh == {"512": 0.6, "1024": 0.8, "4096": 1.0}


def test_row_tokens_reconcile_for_every_case_and_question(tiny_agent, cases):
    root, cs = cases
    prof = Lg.profile_rows(tiny_agent, cs[:6])
    assert len(prof["rows"]) == 6 * 5 and not prof["over_head_budget"]
    for r in prof["rows"]:
        assert r["row_tokens"] == r["special_tokens"] + r["task_option_tokens"] + r["state_tokens"]
    # Independent check against the tokens accounting record of the whole case.
    c = cs[0]
    rec = tokens.account(tiny_agent, D.serialize_state(c["state"]), c["questions"],
                         max_len=tokens.position_limit(tiny_agent))
    by_q = {r["question_id"]: r["row_tokens"] for r in rec["rows"]}
    got = {r["question_id"]: r["row_tokens"] for r in prof["rows"] if r["case_id"] == c["id"]}
    assert got == by_q


def test_case_length_is_its_longest_row(tiny_agent, cases):
    _, cs = cases
    prof = Lg.profile_rows(tiny_agent, cs[:4])
    summ = Lg.summarize(prof["rows"])
    longest = {}
    for r in prof["rows"]:
        longest[r["case_id"]] = max(longest.get(r["case_id"], 0), r["row_tokens"])
    assert summ["cases"]["max"] == max(longest.values())
    assert summ["cases"]["n"] == 4 and summ["rows"]["n"] == 20


def test_per_workflow_aggregates(tiny_agent, cases):
    _, cs = cases
    prof = Lg.profile_rows(tiny_agent, cs)
    agg = Lg.aggregate(prof["rows"])
    assert set(agg["by_workflow"]) == set(D.WORKFLOWS)
    long_wf = agg["by_workflow"]["security_incidents"]["rows"]["median"]
    short_wf = agg["by_workflow"]["customer_service"]["rows"]["median"]
    assert long_wf > short_wf                                   # the fixture's long workflow
    assert agg["overall"]["rows"]["n"] == len(prof["rows"])


def test_options_over_the_head_budget_are_listed_not_dropped(tiny_agent, cases):
    _, cs = cases
    c = json.loads(json.dumps(cs[0], default=str))
    # 60 long option labels overflow the fixture's 96-token head budget.
    c["questions"]["action"] = {"type": "choice", "instructions": "which",
                                "criteria": {"option number %d of many" % i: "x" for i in range(60)}}
    c["qmeta"]["action"]["n_options"] = 60
    prof = Lg.profile_rows(tiny_agent, [c])
    assert [o["question_id"] for o in prof["over_head_budget"]] == ["action"]
    assert len(prof["rows"]) == 5                                # kept in the profile, and flagged
    assert [r["question_id"] for r in prof["rows"] if r["over_head_budget"]] == ["action"]


def test_write_length_profile(tiny_agent, cases):
    root, _ = cases
    path = Lg.write_length_profile(tiny_agent, "tiny", root)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert set(body["by_split"]) == {"train", "test"}
    assert body["by_split"]["test"]["overall"]["rows"]["fit_share"]["512"] is not None
    assert "rows" in body["by_split"]["test"] and "rows" not in body["by_split"]["train"]

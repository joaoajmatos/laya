"""T013: token accounting (experiments/tokens.py)."""
import copy

import pytest

from experiments import tokens as T

Q1 = {"q": {"type": "choice", "instructions": "which option best matches the document",
            "criteria": {"red": "a red one", "blue": "a blue one"}}}
Q2 = dict(Q1, n={"type": "noul", "instructions": "is it late"})
STATE = "the customer said the order was late and asked for a refund on the delivery"


def _check_sum(row):
    assert (row["special_tokens"] + row["task_option_tokens"] + row["state_tokens_retained"]
            + row["padding_tokens"]) == row["final_length"]


def test_parts_sum_exactly(tiny_agent):
    rec = T.account(tiny_agent, STATE, Q1)
    assert rec["n_questions"] == 1 and rec["options_per_question"] == [2]
    (row,) = rec["rows"]
    _check_sum(row)
    assert row["padding_tokens"] == 0 and row["truncated"] is False
    assert row["special_tokens"] == 4 + 2  # CLS, 2 SEP, final SEP, one MASK per option
    assert row["task_option_tokens"] > 0


def test_matches_what_predict_reports(tiny_agent):
    rec = T.account(tiny_agent, STATE, Q2)
    out = tiny_agent.predict(STATE, Q2)
    unpadded = sum(r["final_length"] - r["padding_tokens"] for r in rec["rows"])
    assert out["usage"]["input_tokens"] == unpadded


def test_truncation_flag(tiny_agent):
    long_state = " ".join(["the delivery was late"] * 200)
    rec = T.account(tiny_agent, long_state, Q1, max_len=64)
    (row,) = rec["rows"]
    assert row["truncated"] is True
    assert row["state_tokens_retained"] < row["state_tokens_original"]
    assert row["final_length"] == 64
    _check_sum(row)
    rec = T.account(tiny_agent, STATE, Q1, max_len=10_000)
    assert rec["rows"][0]["truncated"] is False


def test_batch_of_different_lengths_has_padding(tiny_agent):
    rec = T.account(tiny_agent, [STATE, STATE + " " + STATE], Q1, batch=True)
    short, long_ = rec["rows"]
    assert short["padding_tokens"] > 0 and long_["padding_tokens"] == 0
    for r in rec["rows"]:
        _check_sum(r)
    assert rec["n_states"] == 2


def test_multi_question_request_has_one_entry_per_row(tiny_agent):
    rec = T.account(tiny_agent, STATE, Q2)
    assert rec["n_questions"] == 2 and len(rec["rows"]) == 2
    assert rec["options_per_question"] == [2, 2]
    assert {r["question_id"] for r in rec["rows"]} == {"q", "n"}
    for r in rec["rows"]:
        _check_sum(r)


def test_question_and_special_tokens_count_against_position_limit(tiny_agent):
    rec = T.account(tiny_agent, STATE, Q1)
    row = rec["rows"][0]
    assert rec["position_limit"] == 2048
    assert row["final_length"] == row["special_tokens"] + row["task_option_tokens"] + row["state_tokens_retained"]
    bad = copy.deepcopy(rec)
    bad["position_limit"] = row["final_length"] - 1
    with pytest.raises(T.TokenAccountingError):
        T.validate_record(bad)


def test_record_that_does_not_sum_is_rejected(tiny_agent):
    rec = T.account(tiny_agent, STATE, Q1)
    bad = copy.deepcopy(rec)
    bad["rows"][0]["state_tokens_retained"] += 1
    with pytest.raises(T.TokenAccountingError):
        T.validate_record(bad)
    bad = copy.deepcopy(rec)
    bad["requested_total"] = rec["final_length"] + 1
    with pytest.raises(T.TokenAccountingError):
        T.validate_record(bad)
    bad = copy.deepcopy(rec)
    bad["rows"][0]["truncated"] = True
    with pytest.raises(T.TokenAccountingError):
        T.validate_record(bad)

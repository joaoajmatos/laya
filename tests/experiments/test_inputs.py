"""T014: exact-length synthetic requests (experiments/inputs.py)."""
import pytest

from experiments import inputs as I
from experiments import tokens as T


@pytest.mark.parametrize("total", [128, 512, 1500])
@pytest.mark.parametrize("nq,opts,nstates", [(1, 2, 1), (2, 2, 1), (5, 8, 1), (10, 2, 1), (1, 2, 4), (2, 8, 4)])
def test_every_row_hits_the_target(tiny_agent, total, nq, opts, nstates):
    states, questions, acc = I.build_request(tiny_agent, total, seed=3, n_questions=nq,
                                             options_per_question=opts, n_states=nstates)
    assert len(states) == nstates and len(questions) == nq
    assert acc["max_len"] == total and acc["requested_total"] == total
    assert len(acc["rows"]) == nq * nstates
    assert acc["options_per_question"] == [opts] * nq
    for row in acc["rows"]:
        assert row["final_length"] == total
        assert row["padding_tokens"] == 0 and row["truncated"] is False


def test_model_sees_exactly_the_target(tiny_agent):
    states, questions, acc = I.build_request(tiny_agent, 300, seed=1, n_questions=3, n_states=2)
    out = tiny_agent.predict_batch(states, questions, max_len=acc["max_len"])
    assert sum(o["usage"]["input_tokens"] for o in out) == 300 * 3 * 2
    single, q1, acc1 = I.build_request(tiny_agent, 300, seed=1)
    assert tiny_agent.predict(single[0], q1, max_len=acc1["max_len"])["usage"]["input_tokens"] == 300


def test_seeds_change_the_document(tiny_agent):
    a, _, _ = I.build_request(tiny_agent, 256, seed=1)
    b, _, _ = I.build_request(tiny_agent, 256, seed=2)
    c, _, _ = I.build_request(tiny_agent, 256, seed=1)
    assert a != b and a == c
    many, _, _ = I.build_request(tiny_agent, 256, seed=1, n_states=4)
    assert len(set(many)) == 4


def test_target_below_overhead_raises(tiny_agent):
    with pytest.raises(ValueError, match="too small"):
        I.build_request(tiny_agent, 10, seed=0, n_questions=1, options_per_question=8)


def test_a_row_that_misses_the_target_fails(tiny_agent, monkeypatch):
    real = I._filler_state
    monkeypatch.setattr(I, "_filler_state", lambda tok, budget, rng, one: real(tok, budget - 1, rng, one))
    with pytest.raises(T.TokenAccountingError):
        I.build_request(tiny_agent, 256, seed=0)


def test_question_heads_share_one_length(tiny_agent):
    qs = I.build_questions(tiny_agent, 10, 8)
    lengths = I._prefix_lengths(tiny_agent, qs, tiny_agent.cfg["head_max_len"])
    assert len(set(lengths)) == 1

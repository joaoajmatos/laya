"""T021: BM25 (experiments/bm25.py)."""
import math

import pytest

from experiments import bm25


def test_tokenization_is_lowercase_words():
    assert bm25.tokenize("The Order, #42 is LATE!") == ["the", "order", "42", "is", "late"]


def test_scores_match_a_hand_computation():
    docs = ["red apple", "green apple apple", "blue sky"]
    idx = bm25.BM25(docs)
    n = 3
    idf_apple = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    avg = (2 + 3 + 2) / 3

    def score(f, length):
        return idf_apple * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * length / avg))

    got = idx.scores("apple")
    assert got[0] == pytest.approx(score(1, 2))
    assert got[1] == pytest.approx(score(2, 3))
    assert got[2] == 0.0


def test_rank_orders_by_score_and_breaks_ties_by_index():
    docs = ["cat dog", "cat dog", "bird", "cat"]
    order = bm25.BM25(docs).rank("cat dog")
    assert order[:2] == [0, 1]                 # equal scores keep index order
    assert order.index(3) < order.index(2)     # a partial match beats no match
    assert bm25.BM25(docs).rank("cat dog") == order          # deterministic


def test_empty_query_scores_zero_and_helper_matches():
    assert bm25.BM25(["a b", "c"]).scores("") == [0.0, 0.0]
    assert bm25.rank("apple", ["apple pie", "kiwi"])[0] > 0.0
    assert bm25.BM25([]).rank("x") == []

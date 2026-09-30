"""A small pure-Python BM25 (T025; research.md R6, R9).

Shared by distractor selection (`families.py`) and the lexical retrieval baseline (`baselines.py`).
Lowercase word tokens, k1 = 1.5, b = 0.75. Ties break by document index, so results are
deterministic. No dependency.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

K1 = 1.5
B = 0.75
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class BM25:
    """BM25 index over a fixed list of documents."""

    def __init__(self, documents: Sequence[str], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.n = len(self.docs)
        self.avg = (sum(self.lengths) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf: Dict[str, float] = {t: math.log(1.0 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        """BM25 score of every document for `query` (repeated query words count once)."""
        terms = set(tokenize(query))
        out = []
        for c, length in zip(self.docs, self.lengths):
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * (length / self.avg if self.avg else 0.0))
            for t in terms:
                f = c.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out

    def rank(self, query: str) -> List[int]:
        """Document indices, best first; equal scores keep index order."""
        sc = self.scores(query)
        return sorted(range(self.n), key=lambda i: (-sc[i], i))


def rank(query: str, documents: Sequence[str]) -> List[float]:
    """Scores of `documents` for `query` (convenience for one-off use)."""
    return BM25(documents).scores(query)

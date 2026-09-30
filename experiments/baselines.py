"""Baseline conditions as transformations in front of the unchanged `Agent` (T040, T041).

Specs: specs/002-decision-benchmark-baselines (research.md R7, R8, R9; FR-008, FR-016, FR-018, FR-020).

Every runner has two layers so quality and latency share one implementation:

* ``run(agent, item)`` is the *timed path*: everything a deployed system would do for one fresh
  document and one question (windowing, retrieval, `predict`), and nothing else. Latency runs call
  only this (constitution VI: selection is inside the timed call).
* ``annotate(agent, item, meta)`` computes tokens seen and evidence visibility *off the timed path*.

``__call__`` is ``run`` then ``annotate``, used by quality runs.

Conditions (research.md R8): ``native`` (the item's named length), ``trunc512`` and ``truncCap``
(right cut to 512 tokens and to the configured cap), ``window`` (`Agent.predict_long`),
``retrieve<B>`` (BM25 chunks up to B retained tokens), and ``oracle`` (the target record with only
its marker; a diagnostic control that is never ranked as deployable, FR-020).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import bm25, tokens

PROB_SCOPE_FULL = "full_input"
PROB_SCOPE_WINDOW = "deciding_window"

CONDITION_NAMES = ("native", "trunc512", "truncCap", "window", "retrieve512", "retrieve1024", "retrieve2048", "oracle")
RETRIEVE_BUDGETS = (512, 1024, 2048)
WINDOW_SIZES = (256, 512, "default")
CHUNK_TOKENS = 64
CHUNK_STRIDE = 32
#: Windowed calls above this many tokens run one window per forward pass (memory, Phase 1 R9/R11).
WINDOW_BATCH_LIMIT = 2048


# --------------------------------------------------------------------------- evidence

def evidence_fraction(span: Optional[Dict[str, int]], retained_state_tokens: int, state_tokens: int) -> float:
    """Share of the evidence tokens that reach the model when the state is cut to its first
    `retained_state_tokens`. With no recorded span the whole state is the evidence."""
    if span is None:
        return 1.0 if state_tokens == 0 else min(1.0, retained_state_tokens / state_tokens)
    lo, hi = int(span["state_start"]), int(span["state_end"])
    if hi <= lo:
        return 1.0
    return max(0, min(hi, retained_state_tokens) - lo) / float(hi - lo)


def visibility(fraction: float) -> str:
    """``full``, ``partial`` or ``none`` (FR-018)."""
    if fraction >= 1.0 - 1e-9:
        return "full"
    return "none" if fraction <= 1e-9 else "partial"


def covered_fraction(span: Optional[Dict[str, int]], kept: Sequence[Tuple[int, int]]) -> float:
    """Share of the evidence span inside the kept token ranges ``[(start, end), ...]``."""
    if span is None:
        return 1.0
    lo, hi = int(span["state_start"]), int(span["state_end"])
    if hi <= lo:
        return 1.0
    covered = 0
    for a, b in sorted(kept):
        covered += max(0, min(hi, b) - max(lo, a))
    return min(1.0, covered / float(hi - lo))


def _state_ids(agent, text: str) -> List[int]:
    from laya.common import encode_text
    return encode_text(agent.tok, text.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]


# --------------------------------------------------------------------------- runners

class Runner:
    """Base: `run` is the timed path, `annotate` the off-path bookkeeping."""

    name = "runner"
    deployable = True
    probability_scope = PROB_SCOPE_FULL

    def run(self, agent, item: Dict[str, Any]) -> Dict[str, Any]:            # pragma: no cover - abstract
        raise NotImplementedError

    def annotate(self, agent, item: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, agent, item: Dict[str, Any]) -> Dict[str, Any]:
        meta = self.run(agent, item)
        out = {"answer": meta["answer"], "probability_scope": self.probability_scope, "deployable": self.deployable}
        out.update(self.annotate(agent, item, meta))
        return out


class NativeRunner(Runner):
    """Native Laya: the item's state and question through `Agent.predict` at `max_len`.

    ``max_len`` is the named length for a constructed item, or the checkpoint's configured cap for
    an ``original`` item (or `max_len` when set, as for the truncation baselines).
    """

    name = "native"

    def __init__(self, max_len: Optional[int] = None):
        self.max_len = max_len

    def max_len_for(self, agent, item: Dict[str, Any]) -> int:
        if self.max_len is not None:
            return int(self.max_len)
        length = item.get("length")
        if isinstance(length, int):
            return length
        return int(agent.cfg.get("max_len", 512))

    def run(self, agent, item):
        qid = item["question_id"]
        max_len = self.max_len_for(agent, item)
        res = agent.predict(item["state_text"], {qid: item["question"]}, max_len=max_len)
        return {"answer": res["answers"][qid], "max_len": max_len, "input_tokens": (res.get("usage") or {}).get("input_tokens")}

    def annotate(self, agent, item, meta):
        qid = item["question_id"]
        acc = tokens.account(agent, item["state_text"], {qid: item["question"]}, max_len=meta["max_len"])
        row = acc["rows"][0]
        frac = evidence_fraction(item.get("evidence_span"), row["state_tokens_retained"], row["state_tokens_original"])
        return {"tokens_seen": int(row["row_tokens"]), "evidence_fraction": frac, "evidence_visible": visibility(frac),
                "max_len_used": meta["max_len"],
                "beyond_configured_max_len": meta["max_len"] > int(agent.cfg.get("max_len", 512))}


class TruncationRunner(NativeRunner):
    """Right cut to `limit` total tokens: 512, or the configured cap when `limit` is None."""

    def __init__(self, limit: Optional[int]):
        super().__init__(max_len=limit)
        self.limit = limit
        self.name = "trunc512" if limit == 512 else "truncCap"

    def max_len_for(self, agent, item):
        return int(self.limit if self.limit is not None else agent.cfg.get("max_len", 512))


class WindowRunner(Runner):
    """The existing windowed long-document path, `Agent.predict_long` (stride = window // 2)."""

    name = "window"
    probability_scope = PROB_SCOPE_WINDOW

    def __init__(self, size: Any = "default", stride: Optional[int] = None):
        self.size = None if size in (None, "default") else int(size)
        self.stride = stride

    def budget(self, agent) -> int:
        cap, head = int(agent.cfg.get("max_len", 512)), int(agent.cfg.get("head_max_len", 192))
        return self.size if self.size else max(64, cap - head - 8)

    def run(self, agent, item):
        qid = item["question_id"]
        n_hint = len(item["state_text"]) // 3
        res = agent.predict_long(item["state_text"], {qid: item["question"]}, window=self.size, stride=self.stride,
                                 batch_size=1 if n_hint > WINDOW_BATCH_LIMIT else None)
        return {"answer": res["answers"][qid], "usage": res.get("usage") or {}}

    def windows(self, agent, n_state: int) -> List[Tuple[int, int]]:
        """The (start, end) token ranges `predict_long` scans, replicated from its loop."""
        budget = self.budget(agent)
        step = self.stride if self.stride else max(1, budget // 2)
        out, i = [], 0
        while i < n_state:
            out.append((i, min(n_state, i + budget)))
            if i + budget >= n_state:
                break
            i += step
        return out

    def annotate(self, agent, item, meta):
        n_state = len(_state_ids(agent, item["state_text"]))
        wins = self.windows(agent, n_state)
        span = item.get("evidence_span")
        best = max((covered_fraction(span, [w]) for w in wins), default=0.0)
        ans = meta["answer"]
        deciding = ans.get("window")
        dec_frac = covered_fraction(span, [(deciding["token_start"], deciding["token_end"])]) if deciding else None
        return {"tokens_seen": int((meta["usage"] or {}).get("input_tokens") or 0), "evidence_fraction": best,
                "evidence_visible": visibility(best), "windows": len(wins), "window_size": self.budget(agent),
                "deciding_window_fraction": dec_frac, "deciding_window": deciding,
                "max_len_used": int(agent.cfg.get("max_len", 512)), "beyond_configured_max_len": False}


def question_query(qdef: Dict[str, Any]) -> str:
    """The retrieval query: the question's instructions plus its option texts. Never the marker text."""
    crit = qdef.get("criteria")
    parts = [str(qdef.get("instructions", ""))]
    if isinstance(crit, dict):
        for k, v in crit.items():
            parts += [str(k), "" if v is None else str(v)]
    elif isinstance(crit, (list, tuple)):
        parts += [str(c) for c in crit]
    return " ".join(p for p in parts if p)


def chunk_ranges(n: int, size: int = CHUNK_TOKENS, stride: int = CHUNK_STRIDE) -> List[Tuple[int, int]]:
    """Fixed chunks of `size` tokens with 50% overlap; the last chunk is aligned to the end."""
    if n <= size:
        return [(0, n)] if n else []
    out = list(range(0, n - size + 1, stride))
    if out[-1] + size < n:
        out.append(n - size)
    return [(s, s + size) for s in out]


def select_chunks(ranges: Sequence[Tuple[int, int]], scores: Sequence[float], budget: int) -> List[Tuple[int, int]]:
    """Best-scoring chunks whose *union* stays within `budget` tokens; returns merged runs in position order."""
    order = sorted(range(len(ranges)), key=lambda i: (-scores[i], i))
    kept: set = set()
    for i in order:
        a, b = ranges[i]
        add = set(range(a, b)) - kept
        if len(kept) + len(add) <= budget:
            kept |= add
    runs: List[Tuple[int, int]] = []
    for t in sorted(kept):
        if runs and runs[-1][1] == t:
            runs[-1] = (runs[-1][0], t + 1)
        else:
            runs.append((t, t + 1))
    return runs


class RetrieveRunner(Runner):
    """Lexical retrieval, then native Laya on the retained text (research.md R9).

    The state is cut into 64-token chunks with 50% overlap without using record boundaries; chunks are
    ranked by BM25 against the question's instructions and option texts; the best ones are kept up to
    `budget` tokens, in original order. Only the budget is tuned.
    """

    deployable = True

    def __init__(self, budget: int):
        self.budget = int(budget)
        self.name = "retrieve%d" % self.budget

    def select(self, agent, item) -> Tuple[str, List[Tuple[int, int]], int]:
        ids = _state_ids(agent, item["state_text"])
        if len(ids) <= self.budget:
            return item["state_text"], [(0, len(ids))], len(ids)
        ranges = chunk_ranges(len(ids))
        texts = [agent.tok.decode(ids[a:b]) for a, b in ranges]
        scores = bm25.BM25(texts).scores(question_query(item["question"]))
        runs = select_chunks(ranges, scores, self.budget)
        text = "\n".join(agent.tok.decode(ids[a:b]) for a, b in runs)
        return text, runs, len(ids)

    def run(self, agent, item):
        qid = item["question_id"]
        text, runs, n_state = self.select(agent, item)
        fixed = _fixed_tokens(agent, item["question"])
        max_len = min(tokens.position_limit(agent), fixed + self.budget + 32)
        res = agent.predict(text, {qid: item["question"]}, max_len=max_len)
        return {"answer": res["answers"][qid], "runs": runs, "n_state": n_state, "max_len": max_len,
                "input_tokens": (res.get("usage") or {}).get("input_tokens")}

    def annotate(self, agent, item, meta):
        frac = covered_fraction(item.get("evidence_span"), meta["runs"])
        return {"tokens_seen": int(meta.get("input_tokens") or 0), "evidence_fraction": frac,
                "evidence_visible": visibility(frac), "retained_ranges": len(meta["runs"]),
                "max_len_used": meta["max_len"],
                "beyond_configured_max_len": meta["max_len"] > int(agent.cfg.get("max_len", 512))}


def _fixed_tokens(agent, qdef: Dict[str, Any]) -> int:
    from .families import fixed_row_tokens
    return fixed_row_tokens(agent, qdef)


class OracleRunner(NativeRunner):
    """The target record with only its marker, then native Laya: a diagnostic control (FR-020).

    It is also the framing control (research.md R4). The input depends only on the case and
    question, so its result is computed once per (case, question) and reused across lengths.
    """

    name = "oracle"
    deployable = False

    def __init__(self):
        super().__init__(max_len=None)
        self._cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @staticmethod
    def oracle_text(item: Dict[str, Any]) -> str:
        from .families import MARKER_TARGET
        sp = item["evidence_span"]
        return MARKER_TARGET + "\n" + item["state_text"][sp["char_start"]:sp["char_end"]]

    def run(self, agent, item):
        key = (item["case_id"], item["question_id"], str(item.get("option_order")))
        if key not in self._cache:
            qid = item["question_id"]
            cap = int(agent.cfg.get("max_len", 512))
            text = self.oracle_text(item)
            res = agent.predict(text, {qid: item["question"]}, max_len=cap)
            self._cache[key] = {"answer": res["answers"][qid], "max_len": cap, "text": text,
                                "input_tokens": (res.get("usage") or {}).get("input_tokens")}
        return dict(self._cache[key])

    def annotate(self, agent, item, meta):
        return {"tokens_seen": int(meta.get("input_tokens") or 0), "evidence_fraction": 1.0,
                "evidence_visible": "full", "max_len_used": meta["max_len"], "beyond_configured_max_len": False}


def build_runner(name: str, params: Optional[Dict[str, Any]] = None) -> Runner:
    """The runner for a condition name (`params` carries the tuned window size)."""
    params = params or {}
    if name == "native":
        return NativeRunner()
    if name == "trunc512":
        return TruncationRunner(512)
    if name == "truncCap":
        return TruncationRunner(None)
    if name == "window":
        return WindowRunner(params.get("size", "default"), params.get("stride"))
    if name.startswith("retrieve"):
        return RetrieveRunner(int(name[len("retrieve"):]))
    if name == "oracle":
        return OracleRunner()
    raise ValueError("unknown condition %r; choose from %s" % (name, list(CONDITION_NAMES)))

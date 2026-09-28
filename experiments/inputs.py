"""Synthetic requests with an exact total token count (T017, research.md R4).

`build_request(agent, total_tokens, seed, ...)` returns states and questions such that every
question row the model sees is exactly `total_tokens` long, special, question and option tokens
included. The caller passes ``max_len=total_tokens`` to `predict`/`predict_batch`
(``accounting["max_len"]``), so nothing is cut and nothing is padded.

* Questions are built from one template whose only varying part is a single-token word, and all
  questions share the same single-token option labels, so every question has the same head length.
* Each state is seeded English-word filler. Its length is found by binary search on the word
  count, then topped up one single-token word at a time until it fills the budget exactly.
* Every row is verified by `tokens.account`; a miss raises instead of recording a wrong length.

A different seed gives a different document, so timed repeats never reuse a state (fresh-document
workload, constitution VI).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from . import tokens

#: Ordinary English words. Real text shape for the tokenizer, not a repeated token.
FILLER_WORDS = (
    "the a an of to in and or is are was were be been it this that these those with for on at by "
    "from as not no yes about after again against all also always among another any around away "
    "back because before below between both but came can come could day did different do does "
    "down each end even every few find first for found get give go good great had has have he "
    "help her here high him his home house how into just keep kind know land large last left "
    "life light line little long look made make man many may me men might more most mother move "
    "much must name near need never new next night now number off often old once one only open "
    "other our out over own page paper part people place play point put read right room said "
    "same saw say school see seem sentence set she should show side small so some something "
    "sometimes song soon sound spell start state still stop story study such take tell than "
    "their them then there they thing think three through time together too took tree try turn "
    "two under until up us use very walk want water way we well went what when where which while "
    "white who why will word work world would write year you young customer order request refund "
    "delivery account payment invoice support ticket issue problem status update urgent normal "
    "late early closed report document market harbor river mountain city village morning evening "
    "winter summer quiet loud quick slow bright dark warm cold north south east west red blue green"
).split()

#: Candidates for the varying question word and for option labels.
_SLOT_CANDIDATES = (
    "red blue green yellow north south east west river mountain city village market harbor "
    "morning evening winter summer quiet loud small large quick slow bright dark warm cold "
    "customer order refund delivery account payment invoice ticket issue status report document "
    "the a of to in and or is it this that with for on at by from as not no yes new old high low"
).split()

QUESTION_TEMPLATE = "which option best matches the document for the {slot} request"


def _encode(tok, text: str) -> List[int]:
    from laya.common import encode_text
    return encode_text(tok, text, add_special_tokens=False)["input_ids"]


def single_token_words(tok, candidates=_SLOT_CANDIDATES) -> List[str]:
    """Words that add exactly one token, alone and after another word."""
    base = len(_encode(tok, "the"))
    out = []
    for w in dict.fromkeys(candidates):
        if len(_encode(tok, w)) == 1 and len(_encode(tok, "the " + w)) == base + 1 \
                and w != tok.mask_token:
            out.append(w)
    return out


def build_questions(agent, n_questions: int, options_per_question: int) -> Dict[str, Dict[str, Any]]:
    """`n_questions` choice questions with `options_per_question` options and equal head length."""
    if n_questions < 1 or options_per_question < 1:
        raise ValueError("need at least one question and one option")
    words = single_token_words(agent.tok)
    if len(words) < max(n_questions, options_per_question):
        raise ValueError("tokenizer has only %d usable single-token words; need %d"
                         % (len(words), max(n_questions, options_per_question)))
    labels = words[:options_per_question]
    slots = words[:n_questions]
    return {
        "q%d" % i: {
            "type": "choice",
            "instructions": QUESTION_TEMPLATE.format(slot=slots[i]),
            "criteria": list(labels),
        }
        for i in range(n_questions)
    }


def _prefix_lengths(agent, questions, head_max_len) -> List[int]:
    from laya.common import build_sequence
    out = []
    for qid, qdef in questions.items():
        agent._check_question(qid, qdef)
        ids, _ = build_sequence(agent.tok, "", agent._to_internal(qdef), max_len=10 ** 9,
                                head_max_len=head_max_len, state_ids=[])
        out.append(len(ids))  # prefix plus the final [SEP]
    return out


def _filler_state(tok, budget: int, rng: random.Random, one_token: List[str]) -> str:
    """Seeded filler whose tokenization is exactly `budget` tokens."""
    if budget < 1:
        raise ValueError("state budget must be at least 1 token")
    words = [rng.choice(FILLER_WORDS) for _ in range(budget + 16)]

    def ntok(n: int) -> int:
        return len(_encode(tok, " ".join(words[:n]))) if n else 0

    lo, hi = 0, len(words)
    while ntok(hi) < budget:  # every word gives at least one token, so this ends quickly
        words.extend(rng.choice(FILLER_WORDS) for _ in range(budget))
        hi = len(words)
    while lo < hi:  # largest n with ntok(n) <= budget
        mid = (lo + hi + 1) // 2
        if ntok(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    text_words = words[:lo]
    count = ntok(lo)
    for _ in range(budget * 2 + 8):
        if count == budget:
            return " ".join(text_words)
        if count < budget:
            text_words.append(rng.choice(one_token))
        else:
            text_words.pop()
        count = len(_encode(tok, " ".join(text_words))) if text_words else 0
    raise ValueError("could not reach exactly %d state tokens" % budget)


def build_request(agent, total_tokens: int, seed: int, n_questions: int = 1,
                  options_per_question: int = 2, n_states: int = 1
                  ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """States, questions and the verified token accounting for one request.

    Every question row of every state is exactly `total_tokens` long. Use
    ``accounting["max_len"]`` as the per-call `max_len`.
    """
    total_tokens = int(total_tokens)
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    questions = build_questions(agent, n_questions, options_per_question)
    prefixes = _prefix_lengths(agent, questions, head_max_len)
    if len(set(prefixes)) != 1:
        raise ValueError("question heads differ in length %s; rows could not share one length" % prefixes)
    overhead = prefixes[0]
    budget = total_tokens - overhead
    if budget < 1:
        raise ValueError(
            "total_tokens=%d is too small: the question, %d option(s) and special tokens already "
            "take %d tokens, leaving no room for a state" % (total_tokens, options_per_question, overhead))
    one_token = single_token_words(agent.tok, FILLER_WORDS) or single_token_words(agent.tok)
    states = []
    for i in range(n_states):
        rng = random.Random("laya-filler:%d:%d" % (seed, i))
        states.append(_filler_state(agent.tok, budget, rng, one_token))
    accounting = tokens.account(agent, states if n_states > 1 else states[0], questions,
                                max_len=total_tokens, head_max_len=head_max_len,
                                requested_total=total_tokens, batch=n_states > 1)
    for row in accounting["rows"]:
        if row["padding_tokens"] or row["truncated"]:
            raise tokens.TokenAccountingError("built request has padding or truncation: %r" % row)
    accounting["seed"] = seed
    return states, questions, accounting

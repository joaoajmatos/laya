"""`golden tokenizer`: records of Laya's real sequence builder on a fixed input list (laya:005, implements raya:004).

Every record comes from `laya.common.build_sequence` via `Agent._encode_state` (the path `predict` takes). Nothing in
`laya/` is reimplemented or changed. Fields marked "derived" are computed here from the builder's output and its
documented budget; the builder itself returns neither position ids nor truncation flags.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import common

DEFAULT_TOKENIZER_OUT = common.DEFAULT_OUT / "tokenizer"

DOC = "the customer asked for a refund on a late delivery and the support ticket is still open"
Q_CHOICE = {"type": "choice", "instructions": "which option best matches the document",
            "criteria": {"continue": "proceed with the request", "review": "open a review", "stop": "halt the request"}}
Q_SCORE = {"type": "score", "instructions": "how risky is the request",
           "criteria": ["zero risk", "low risk", "high risk"]}
Q_NOUL = {"type": "noul", "instructions": "this document needs review",
          "criteria": {"false": "no review is needed", "true": "a review is needed"}}
Q_NOUL_LABELS = {"type": "noul", "instructions": "is the order late", "labels": {"false": "on time", "true": "late"}}
Q_MARKERS = {"type": "choice", "instructions": "which [MASK] [SEP] option is best [CLS]",
             "criteria": {"a": "first [MASK] option", "b": "second [SEP] option"}}
Q_MANY = {"type": "choice", "instructions": "pick one of many options",
          "criteria": {"opt%d" % i: "description of option number %d with some extra words" % i for i in range(12)}}

CONVERSATION = [{"role": "user", "content": "my order is late, where is it"},
                {"role": "agent", "content": "let me check the delivery status for you"},
                {"role": "user", "content": "please open a refund request for the late order"}] * 6


def _exact_state(builder_prefix_len: int, max_len: int, extra: int = 0) -> str:
    """`the` is one token; `max_len - builder_prefix_len + extra` of them fills the sequence exactly (+ `extra`)."""
    return " ".join(["the"] * (max_len - builder_prefix_len + extra))


def input_list(agent) -> List[Dict[str, Any]]:
    """The fixed inputs. `exact` ones are sized from the builder's own prefix length so they hit `max_len` exactly."""
    from laya.common import build_sequence
    max_len = 96
    prefix, _ = build_sequence(agent.tok, "", agent._to_internal(Q_CHOICE), max_len, 64)
    plen = len(prefix)                      # head part including the closing [SEP] of the sequence
    base = dict(max_len=max_len, head_max_len=64)
    return [
        dict(name="empty_document", description="empty document, one question", state="", questions={"q": Q_CHOICE}, **base),
        dict(name="primary_one_question", description="one document, one question (primary workload)",
             state=DOC, questions={"q": Q_CHOICE}, **base),
        dict(name="several_questions", description="one document, choice + score + noul + noul with labels",
             state=DOC, questions={"action": Q_CHOICE, "risk": Q_SCORE, "needs_review": Q_NOUL, "late": Q_NOUL_LABELS},
             **base),
        dict(name="state_dict", description="state given as a dict (serialized as JSON)",
             state={"task": "refund", "customer": "ana", "status": "late", "amount": 7}, questions={"q": Q_CHOICE}, **base),
        dict(name="state_conversation_list", description="conversation list state; truncated from the left",
             state=CONVERSATION, questions={"q": Q_CHOICE}, **base),
        dict(name="unicode", description="non-ASCII text, accents, CJK, emoji, combining marks",
             state="Naïve café — 東京の配達が遅れた 😀 Ünïcödé é ß ﬁ", questions={"q": Q_CHOICE}, **base),
        dict(name="marker_like_strings", description="[MASK]/[CLS]/[SEP]/[PAD]/[UNK] in the document and the question",
             state="before [MASK] middle [CLS] x [SEP] y [PAD] z [UNK] after", questions={"q": Q_MARKERS}, **base),
        dict(name="long_document", description="document far longer than max_len (right-truncated)",
             state=(DOC + " ") * 30, questions={"q": Q_CHOICE}, **base),
        dict(name="exact_max_len", description="document that makes the sequence exactly max_len",
             state=_exact_state(plen, max_len), questions={"q": Q_CHOICE}, **base),
        dict(name="max_len_plus_one", description="one token more than exact_max_len (truncated by one)",
             state=_exact_state(plen, max_len, 1), questions={"q": Q_CHOICE}, **base),
        dict(name="options_over_head_budget", description="many options with a small head_max_len (per-option cap)",
             state=DOC, questions={"q": Q_MANY}, max_len=96, head_max_len=48),
    ]


def build_record(agent, spec: Dict[str, Any]) -> Dict[str, Any]:
    """One record: the builder's output for every question of one input, plus derived truncation fields."""
    from laya.common import QTYPES, build_sequence, encode_text, serialize_state
    tok = agent.tok
    internal = {qid: agent._to_internal(q) for qid, q in spec["questions"].items()}
    ids = list(spec["questions"])
    state = spec["state"]
    truncate_left = isinstance(state, list)
    items = agent._encode_state(state, ids, internal, spec["max_len"], spec["head_max_len"])
    state_text = serialize_state(state)
    state_ids = encode_text(tok, state_text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    out_items = []
    for qid, item in zip(ids, items):
        head, _ = build_sequence(tok, "", internal[qid], spec["max_len"], spec["head_max_len"], truncate_left=truncate_left)
        head_len = len(head) - 1                         # ids before the state, closing [SEP] of the question included
        room = max(0, spec["max_len"] - head_len - 1)    # the builder's own budget for the state
        kept = min(len(state_ids), room)
        # The builder cuts the sequence a second time at max_len; the markers beyond it are dropped.
        clipped = head_len + kept + 1 > spec["max_len"]
        seq = item["ids"]
        state_part = seq[head_len:head_len + kept]
        expect = state_ids[len(state_ids) - kept:] if truncate_left and kept else state_ids[:kept]
        if state_part != expect:
            raise RuntimeError("record %s/%s: derived state span does not match the builder output" % (spec["name"], qid))
        out_items.append({
            "qid": qid, "qtype": item["qtype"], "qtype_name": internal[qid]["t"], "input_ids": seq,
            "position_ids": list(range(len(seq))),            # derived: the encoder uses arange(len)
            "markers": item["markers"], "length": len(seq),
            "options_stats": item["options"],
            "head_length": head_len,                          # derived
            "state_tokens_total": len(state_ids), "state_tokens_kept": kept,
            "state_truncated": kept < len(state_ids), "state_budget": room,
            "sequence_clipped": clipped, "truncate_left": truncate_left,
        })
    if len(ids) == 1:
        direct = build_sequence(tok, state, internal[ids[0]], spec["max_len"], spec["head_max_len"],
                                truncate_left=truncate_left, return_stats=True)
        if list(direct[0]) != out_items[0]["input_ids"] or list(direct[1]) != out_items[0]["markers"]:
            raise RuntimeError("record %s: build_sequence and Agent._encode_state disagree" % spec["name"])
    return {"name": spec["name"], "description": spec["description"], "state": state, "state_serialized": state_text,
            "questions": spec["questions"], "internal_questions": internal, "max_len": spec["max_len"],
            "head_max_len": spec["head_max_len"], "truncate_left": truncate_left, "items": out_items}


def _tokenizer_json(checkpoint: str) -> Path:
    root = Path(checkpoint)
    for cand in (root / "tokenizer" / "tokenizer.json", root / "tokenizer.json"):
        if cand.is_file():
            return cand
    raise FileNotFoundError("no tokenizer.json under %s" % checkpoint)


def export_tokenizer(out: Path = DEFAULT_TOKENIZER_OUT, checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Write records/, the tokenizer.json used, and meta.json. Fixture tokenizer unless `checkpoint` is given."""
    import laya
    out = Path(out)
    common.make_deterministic()
    if checkpoint:
        agent = laya.Agent(checkpoint, device="cpu")
        return _write(out, agent, _tokenizer_json(checkpoint), {"model": "checkpoint",
                                                                 "checkpoint_dir_name": Path(checkpoint).name})
    with common.FixtureAgent(64) as fx:
        return _write(out, fx.agent, _tokenizer_json(fx.path), {"model": "fixture", "fixture_hidden": 64})


def _write(out: Path, agent, tokenizer_json: Path, source: Dict[str, Any]) -> Dict[str, Any]:
    tok = agent.tok
    specs = input_list(agent)
    names = []
    for spec in specs:
        common.write_json(out / "records" / ("%s.json" % spec["name"]), build_record(agent, spec))
        names.append(spec["name"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_json, out / "tokenizer.json")
    meta = common.base_meta(
        "tokenizer-golden", records=sorted(names), tokenizer_file="tokenizer.json",
        special_tokens={"pad": tok.pad_token_id, "unk": tok.unk_token_id, "cls": tok.cls_token_id,
                        "sep": tok.sep_token_id, "mask": tok.mask_token_id},
        mask_token=tok.mask_token, vocab_size=len(tok), builder="laya/common.py:build_sequence via Agent._encode_state",
        option_token_cap=48, **source)
    common.write_json(out / "meta.json", meta)
    return meta

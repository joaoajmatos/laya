"""Token accounting (T016, data-model.md "Token Accounting Record", research.md R7).

For every question row of a request, split the tensor the model sees into special tokens,
question/option tokens, retained state tokens and padding, and check that they add up exactly.

The rows come from `Agent._encode_state`, the same code the public `predict` path runs, so the
record describes what the model actually received. The parts are counted independently of that
code: the state is tokenized in full on its own, and the question prefix is rebuilt by
`build_sequence` with an empty state. A mismatch means the accounting (or the input builder) is
wrong, and `TokenAccountingError` stops the run instead of recording a wrong length.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

State = Union[str, dict, list]


class TokenAccountingError(ValueError):
    """A token accounting record does not reconcile."""


def position_limit(agent) -> int:
    """Positional capacity of the loaded encoder (`max_position_embeddings`)."""
    return int(agent.model.encoder.config.max_position_embeddings)


def _internal_questions(agent, questions: Dict[str, Dict[str, Any]]):
    ids = list(questions)
    for qid in ids:
        agent._check_question(qid, questions[qid])
    return ids, {qid: agent._to_internal(questions[qid]) for qid in ids}


def _row_parts(agent, state: State, q: Dict[str, Any], max_len: int, head_max_len: int,
               state_len: int) -> Dict[str, int]:
    """Independent count of one row's parts, before padding."""
    from laya.common import build_sequence

    # The question prefix alone: [CLS] head [SEP] ([MASK] option)* [SEP], then the final [SEP].
    ids0, markers0 = build_sequence(agent.tok, state, q, max_len=10 ** 9, head_max_len=head_max_len,
                                    state_ids=[])
    prefix_len = len(ids0) - 1
    k = len(markers0)
    special = 4 + k                        # CLS, SEP after head, SEP after options, final SEP, one MASK per option
    task_option = prefix_len - 3 - k       # head tokens + option text tokens
    room = max(0, max_len - prefix_len - 1)
    retained = min(state_len, room)
    return {"special_tokens": special, "task_option_tokens": task_option,
            "state_tokens_retained": retained, "options": k}


def account(agent, state: Union[State, Sequence[State]], questions: Dict[str, Dict[str, Any]],
            max_len: Optional[int] = None, head_max_len: Optional[int] = None,
            requested_total: Optional[int] = None, batch: bool = False) -> Dict[str, Any]:
    """Token Accounting Record for one request.

    `state` is one state, or a list of states when `batch=True` (the `predict_batch` path, all
    states collated into one tensor). Every (state, question) pair is one row. Raises
    `TokenAccountingError` when the record does not reconcile.
    """
    from laya.common import collate_items, encode_text, serialize_state

    states = list(state) if batch else [state]
    max_len = int(agent.cfg.get("max_len", 512) if max_len is None else max_len)
    head_max_len = int(agent.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len)
    ids, internal = _internal_questions(agent, questions)
    limit = position_limit(agent)

    encoded, rows = [], []
    for si, st in enumerate(states):
        # What the model receives: the same function predict() uses.
        items = agent._encode_state(st, ids, internal, max_len=max_len, head_max_len=head_max_len)
        encoded.append(items)
        # The full state, tokenized independently of the sequence builder.
        full = encode_text(agent.tok, serialize_state(st).replace(agent.tok.mask_token, " "),
                           add_special_tokens=False)["input_ids"]
        for qid, item in zip(ids, items):
            parts = _row_parts(agent, st, internal[qid], max_len, head_max_len, len(full))
            rows.append({
                "state_index": si,
                "question_id": qid,
                "state_tokens_original": len(full),
                "task_option_tokens": parts["task_option_tokens"],
                "special_tokens": parts["special_tokens"],
                "state_tokens_retained": parts["state_tokens_retained"],
                "row_tokens": len(item["ids"]),
                "options": parts["options"],
            })

    collated = collate_items(encoded, agent.tok.pad_token_id)
    tensor_len = int(collated["input_ids"].shape[1])
    att = collated["attention_mask"]
    for r, row in enumerate(rows):
        row["padding_tokens"] = int((att[r] == 0).sum())
        row["final_length"] = tensor_len
        row["truncated"] = row["state_tokens_retained"] < row["state_tokens_original"]

    record = {
        "requested_total": requested_total,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "n_states": len(states),
        "n_questions": len(ids),
        "options_per_question": [rows[i]["options"] for i in range(len(ids))],
        "position_limit": limit,
        "final_length": tensor_len,
        "truncated": any(r["truncated"] for r in rows),
        "padding_tokens": sum(r["padding_tokens"] for r in rows),
        "rows": rows,
    }
    validate_record(record)
    return record


def validate_record(record: Dict[str, Any]) -> None:
    """Check the reconciliation rules; raise `TokenAccountingError` on the first violation."""
    rows = record.get("rows") or []
    if not rows:
        raise TokenAccountingError("record has no rows")
    if record.get("n_questions") is not None and len(record.get("options_per_question", [])) != record["n_questions"]:
        raise TokenAccountingError("options_per_question needs one entry per question")
    finals = {r["final_length"] for r in rows}
    if len(finals) != 1:
        raise TokenAccountingError("rows report different final lengths %s" % sorted(finals))
    for i, r in enumerate(rows):
        parts = (r["special_tokens"] + r["task_option_tokens"] + r["state_tokens_retained"]
                 + r["padding_tokens"])
        if parts != r["final_length"]:
            raise TokenAccountingError(
                "row %d: special %d + task/option %d + state %d + padding %d = %d, but the model saw %d"
                % (i, r["special_tokens"], r["task_option_tokens"], r["state_tokens_retained"],
                   r["padding_tokens"], parts, r["final_length"]))
        if "row_tokens" in r and r["row_tokens"] + r["padding_tokens"] != r["final_length"]:
            raise TokenAccountingError("row %d: row tokens + padding != final length" % i)
        if r["final_length"] > record["position_limit"]:
            raise TokenAccountingError("row %d: final length %d exceeds position limit %d"
                                       % (i, r["final_length"], record["position_limit"]))
        if r["truncated"] != (r["state_tokens_retained"] < r["state_tokens_original"]):
            raise TokenAccountingError("row %d: truncated flag disagrees with the counts" % i)
        req = record.get("requested_total")
        if req is not None and r["final_length"] != req:
            raise TokenAccountingError("row %d: requested %d tokens, model saw %d"
                                       % (i, req, r["final_length"]))

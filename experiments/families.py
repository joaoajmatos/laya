"""Controlled long-context evaluation items built from upstream cases (T026 to T029).

Specs: specs/002-decision-benchmark-baselines (research.md R3 to R6; FR-009 to FR-011, FR-015).

An item is one *question row*: (case, question, variant, length). Its state is a plain-text
document of marked records around the *unchanged* target record::

    [BACKGROUND NOTE]              neutral filler (absorbs the exact-length adjustment)
    [REFERENCE RECORD, NOT UNDER REVIEW]   a near-matching upstream *training* record
    [RECORD UNDER REVIEW]          the target record, byte-identical to the native serialization
    ...

The finished model row has exactly the named number of tokens, checked by `tokens.account`. The
target's evidence span (token range in the state and in the row) is recorded from the tokenizer's
character offsets, so it is exact.

Variants (data-model.md maps them to the spec's three families):

* ``neutral@mid``        neutral padding only, target in the middle (neutral padding family)
* ``distractor@mid``     near-matching reference records, target in the middle
* ``distractor@begin``   the same records, target first          (position family)
* ``distractor@end``     the same records, target last           (position family)
* ``oracle``             the target record with only its marker (framing control, R4)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import bm25, data, tokens
from .inputs import single_token_words

RULE_VERSION = "1"
NEUTRAL_VOCAB_VERSION = "1"

MARKER_TARGET = "[RECORD UNDER REVIEW]"
MARKER_REFERENCE = "[REFERENCE RECORD, NOT UNDER REVIEW]"
MARKER_NEUTRAL = "[BACKGROUND NOTE]"
MARKERS = {"target": MARKER_TARGET, "reference": MARKER_REFERENCE, "neutral": MARKER_NEUTRAL}

DEFAULT_LENGTHS = (512, 1024, 2048, 4096, 8192)
POSITION_VARIANTS = ("distractor@begin", "distractor@mid", "distractor@end")
CONTEXT_VARIANTS = ("neutral@mid",) + POSITION_VARIANTS
DEFAULT_VARIANTS = CONTEXT_VARIANTS + ("oracle",)

#: Reference records are drawn in seeded groups of this many, best-ranked group first (R6).
REFERENCE_GROUP = 20
#: A neutral block is at least this many filler words (plus its marker).
MIN_FILLER = 4
#: Tokens kept back from the greedy fill so the exact-length step always has room.
SAFETY_MARGIN = 3

#: Generic words with no workflow terms (customers, orders, invoices, incidents, agents, tickets...).
NEUTRAL_VOCAB = tuple(
    ("red blue green yellow north south east west river mountain city village market harbor morning "
     "evening winter summer quiet loud small large quick slow bright dark warm cold garden window "
     "bridge forest valley island ocean desert meadow cloud storm breeze thunder rain snow frost "
     "sunset sunrise shadow lantern candle basket ribbon marble copper silver golden crimson violet "
     "orchard vineyard pasture canyon glacier lagoon reef cliff pebble harvest festival journey "
     "compass anchor sail voyage lighthouse cottage chimney doorway staircase courtyard fountain "
     "meadowlark heron sparrow falcon willow maple cedar birch clover thistle fern moss lichen "
     "amber ivory scarlet teal indigo plum honey ginger pepper saffron walnut almond cherry apple "
     "pear peach melon lemon orange breeze whisper echo rhythm melody harmony chorus lullaby "
     "the a of to in and or is it this that with for on at by from as not no yes new old high low").split())


class FamilyError(ValueError):
    """An item cannot be built as specified (never recorded as a wrong length)."""


# --------------------------------------------------------------------------- tokenizer helpers

def _ids(agent, text: str) -> List[int]:
    from laya.common import encode_text
    return encode_text(agent.tok, text.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]


def count_tokens(agent, text: str) -> int:
    """Token count of `text`, cached on the agent (reference records are counted once per pool, not per item)."""
    cache = agent.__dict__.setdefault("_families_token_cache", {})
    n = cache.get(text)
    if n is None:
        n = cache[text] = len(_ids(agent, text))
    return n


def fixed_row_tokens(agent, qdef: Dict[str, Any], head_max_len: Optional[int] = None) -> int:
    """Tokens of a row with an empty state: ``[CLS] head [SEP] options [SEP]`` plus the final ``[SEP]``."""
    from laya.common import build_sequence
    head = int(agent.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len)
    ids, _ = build_sequence(agent.tok, "", agent._to_internal(qdef), max_len=10 ** 9, head_max_len=head, state_ids=[])
    return len(ids)


def neutral_words(agent) -> List[str]:
    """Neutral words that add exactly one token, alone and after another word (as Phase 1 `inputs`)."""
    words = single_token_words(agent.tok, NEUTRAL_VOCAB)
    if len(words) < 8:
        raise FamilyError("tokenizer has only %d usable neutral single-token words" % len(words))
    return words


def filler(words: Sequence[str], n: int, rng: random.Random) -> str:
    """`n` seeded neutral words (each one token)."""
    return " ".join(rng.choice(words) for _ in range(max(0, n)))


# --------------------------------------------------------------------------- reference records

_IDENT = re.compile(r"[A-Za-z0-9_-]*\d{3,}[A-Za-z0-9_-]*")


def identifier_tokens(text: str) -> set:
    """Identifier-like strings (three or more digits), which must never repeat between records."""
    return set(_IDENT.findall(text))


class ReferencePool:
    """Training records of one workflow, ranked per target by BM25 similarity (research.md R6)."""

    def __init__(self, train_cases: Sequence[Dict[str, Any]]):
        self.by_workflow: Dict[str, List[Dict[str, Any]]] = {}
        for c in sorted(train_cases, key=lambda c: c["id"]):
            self.by_workflow.setdefault(c["workflow"], []).append(c)
        self.text = {wf: [data.serialize_state(c["state"]) for c in cs] for wf, cs in self.by_workflow.items()}
        self.index = {wf: bm25.BM25(t) for wf, t in self.text.items()}

    def ordered(self, target: Dict[str, Any], seed: Any) -> List[Dict[str, Any]]:
        """Reference records for `target`: near matches first, in seeded groups of 20, without repeats.

        A record is skipped when it repeats the target's text or one of its identifier-like strings.
        The order depends only on the case and the seed, so every length and variant of a case draws
        from the same sequence (nested sets, and one distractor set for all three positions).
        """
        wf = target["workflow"]
        target_text = data.serialize_state(target["state"])
        ranked = self.index[wf].rank(target_text)
        bad = identifier_tokens(target_text)
        keep = []
        for i in ranked:
            text = self.text[wf][i]
            if text == target_text or (bad and identifier_tokens(text) & bad):
                continue
            keep.append(i)
        rng = random.Random("%s|refs|%s" % (seed, target["id"]))
        order: List[int] = []
        for g in range(0, len(keep), REFERENCE_GROUP):
            group = keep[g:g + REFERENCE_GROUP]
            rng.shuffle(group)
            order += group
        return [{"case": self.by_workflow[wf][i], "text": self.text[wf][i]} for i in order]


# --------------------------------------------------------------------------- options

def permuted_question(qdef: Dict[str, Any], case_id: str, qid: str, seed: Any) -> Tuple[Dict[str, Any], Optional[List[int]]]:
    """`choice` option order permuted with a seed (FR-015); other types are never permuted.

    Labels are the option keys, so the gold label needs no remapping. The permutation depends on the
    case, question and seed only, so every variant and length of a question sees the same order.
    """
    if qdef["type"] != "choice":
        return qdef, None
    keys = list(qdef["criteria"])
    rng = random.Random("%s|options|%s|%s" % (seed, case_id, qid))
    order = list(range(len(keys)))
    rng.shuffle(order)
    out = dict(qdef)
    out["criteria"] = {keys[i]: qdef["criteria"][keys[i]] for i in order}
    return out, order


# --------------------------------------------------------------------------- one item

def _block(marker: str, text: str) -> str:
    return "%s\n%s" % (marker, text)


def _compose(blocks: Sequence[Tuple[str, str, str]]) -> Tuple[str, Dict[str, int]]:
    """Join (kind, marker, text) blocks; return the text and the target record's character range."""
    parts, pos, span = [], 0, {}
    for i, (kind, marker, text) in enumerate(blocks):
        head = marker + "\n"
        if kind == "target":
            span = {"char_start": pos + len(head), "char_end": pos + len(head) + len(text)}
        block = head + text
        parts.append(block)
        pos += len(block) + (2 if i < len(blocks) - 1 else 0)
    return "\n\n".join(parts), span


def _token_span(agent, text: str, char_start: int, char_end: int) -> Tuple[int, int, int]:
    """(state_start, state_end, n_tokens): tokens overlapping the character range, from tokenizer offsets."""
    from laya.common import encode_text
    if agent.tok.mask_token in text:
        raise FamilyError("the built state contains the mask token %r; character offsets would shift" % agent.tok.mask_token)
    enc = encode_text(agent.tok, text, add_special_tokens=False,
                      return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    start = next(i for i, (s, e) in enumerate(offs) if e > char_start)
    end = next((i for i, (s, e) in enumerate(offs) if s >= char_end), len(offs))
    return start, end, len(enc["input_ids"])


def _item_seed(seed: Any, *parts: Any) -> str:
    return "|".join(str(p) for p in (seed,) + parts)


def build_item(agent, case: Dict[str, Any], qid: str, variant: str, length: Any, split: str,
               refs: Sequence[Dict[str, Any]], words: Sequence[str], seed: Any = 0,
               rule_version: str = RULE_VERSION) -> Dict[str, Any]:
    """Build one Evaluation Item (data-model.md). `length` is an int, or ``"original"`` for ``oracle``."""
    qdef, order = permuted_question(case["questions"][qid], case["id"], qid, seed)
    target_text = data.serialize_state(case["state"])
    item = {
        "item_id": data.item_id(case["id"], qid, variant, length),
        "case_id": case["id"], "question_id": qid, "split": split, "workflow": case["workflow"],
        "question_type": qdef["type"], "n_options": case["qmeta"][qid]["n_options"],
        "variant": variant, "length": length, "question": qdef,
        "gold": data.gold_block(case, qid), "option_order": order,
        "construction": {"seed": seed, "rule_version": rule_version, "marker_constants": MARKERS,
                         "distractor_case_ids": [], "neutral_vocab_version": NEUTRAL_VOCAB_VERSION},
    }
    if variant == "oracle":
        blocks = [("target", MARKER_TARGET, target_text)]
        text, span = _compose(blocks)
        s, e, n = _token_span(agent, text, span["char_start"], span["char_end"])
        item.update(state_text=text, layout=[{"kind": "target", "ref_case_id": case["id"], "tokens": n}],
                    evidence_span={"state_start": s, "state_end": e, **span}, status="ok")
        return item

    if variant not in CONTEXT_VARIANTS:
        raise FamilyError("unknown variant %r" % variant)
    limit = tokens.position_limit(agent)
    if int(length) > limit:
        return _unsupported(item, "%d exceeds positional capacity %d" % (length, limit))
    fixed = fixed_row_tokens(agent, qdef)
    budget = int(length) - fixed                      # state tokens the row can hold
    target_block = _block(MARKER_TARGET, target_text)
    t_cost = count_tokens(agent, target_block)
    m_bg = count_tokens(agent, MARKER_NEUTRAL + "\n") + 1
    room = budget - t_cost
    if t_cost > budget:
        return _unsupported(item, "target_exceeds_length: the target row needs %d tokens at least, %d named"
                            % (t_cost + fixed, length))
    if room < m_bg + MIN_FILLER:
        return _unsupported(item, "no_room_for_context: %d tokens are left around the target" % room)

    rng = random.Random(_item_seed(seed, case["id"], qid, variant, length))
    picked: List[Dict[str, Any]] = []
    if variant != "neutral@mid":
        used = 0
        for ref in refs:
            cost = count_tokens(agent, _block(MARKER_REFERENCE, ref["text"])) + 1
            if used + cost + m_bg + MIN_FILLER + SAFETY_MARGIN > room:
                break
            picked.append(ref)
            used += cost
    position = {"neutral@mid": "mid", "distractor@mid": "mid", "distractor@begin": "begin",
                "distractor@end": "end"}[variant]

    # Two marked background blocks need their two markers plus at least MIN_FILLER words between them.
    two_blocks = variant == "neutral@mid" and room >= 2 * m_bg + MIN_FILLER

    def compose(n_fill: int) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
        ref_blocks = [("reference", MARKER_REFERENCE, r["text"]) for r in picked]
        if variant == "neutral@mid" and two_blocks:
            first = n_fill // 2
            second = n_fill - first
            blocks = []
            if first >= 1:
                blocks.append(("neutral", MARKER_NEUTRAL, filler(words, first, random.Random(rng.random()))))
            blocks.append(("target", MARKER_TARGET, target_text))
            blocks.append(("neutral", MARKER_NEUTRAL, filler(words, second, random.Random(rng.random()))))
        elif variant == "neutral@mid":          # too little room for two marked blocks: one block after the target
            blocks = [("target", MARKER_TARGET, target_text),
                      ("neutral", MARKER_NEUTRAL, filler(words, n_fill, random.Random(rng.random())))]
        else:
            k = {"begin": 0, "mid": len(ref_blocks) // 2, "end": len(ref_blocks)}[position]
            blocks = ref_blocks[:k] + [("target", MARKER_TARGET, target_text)] + ref_blocks[k:]
            pad = ("neutral", MARKER_NEUTRAL, filler(words, n_fill, random.Random(rng.random())))
            blocks = [pad] + blocks if position == "end" else blocks + [pad]
        text, span = _compose(blocks)
        return text, span, [(b[0], b[1]) for b in blocks]

    # Fit exactly: start from an estimate and correct by the measured difference (each filler word is one token).
    n_fill = max(MIN_FILLER, room - sum(count_tokens(agent, _block(MARKER_REFERENCE, r["text"])) + 1 for r in picked)
                 - (2 * m_bg if two_blocks else m_bg) - SAFETY_MARGIN)
    text = None
    for _ in range(8):
        text, span, kinds = compose(n_fill)
        s, e, n_state = _token_span(agent, text, span["char_start"], span["char_end"])
        diff = budget - n_state
        if diff == 0:
            break
        n_fill += diff
        if n_fill < MIN_FILLER:
            picked = picked[:-1] if picked else picked
            if not picked and variant != "neutral@mid":
                return _unsupported(item, "no_room_for_context: the exact length cannot be reached")
            n_fill = MIN_FILLER + SAFETY_MARGIN
    else:
        # The exact length cannot be reached with whole marked blocks: keep the item's place, say why, never a wrong length.
        return _unsupported(item, "length_not_reachable: %s tokens cannot be reached exactly around this record" % length)

    layout = [{"kind": kind} for kind, _ in kinds]
    for entry, r in zip([l for l in layout if l["kind"] == "reference"], picked):
        entry["ref_case_id"] = "train/%s" % r["case"]["id"]
    for entry in layout:
        if entry["kind"] == "target":
            entry["ref_case_id"] = case["id"]
    row_start = fixed - 1 + s
    item["construction"]["distractor_case_ids"] = ["train/%s" % r["case"]["id"] for r in picked]
    item.update(state_text=text, layout=layout, status="ok",
                evidence_span={"state_start": s, "state_end": e, "row_start": row_start,
                               "row_end": row_start + (e - s), **span},
                position={"requested": position, "start_fraction": s / float(n_state),
                          "end_fraction": e / float(n_state)})
    item["accounting"] = tokens.account(agent, text, {qid: qdef}, max_len=int(length), requested_total=int(length))
    return item


def _unsupported(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    item.update(status="unsupported", reason=reason, state_text=None, layout=[], evidence_span=None)
    return item


# --------------------------------------------------------------------------- item sets and files

def families_id(agent_name: str, fingerprint: str, seed: Any, lengths: Sequence[int], variants: Sequence[str],
                rule_version: str = RULE_VERSION) -> str:
    """Fingerprint of everything that determines the items (rule version, seeds, markers, vocabulary)."""
    body = json.dumps({"rule": rule_version, "vocab": NEUTRAL_VOCAB_VERSION, "seed": seed, "lengths": list(lengths),
                       "variants": list(variants), "markers": MARKERS, "tokenizer": agent_name,
                       "data": fingerprint, "reference_group": REFERENCE_GROUP, "min_filler": MIN_FILLER},
                      sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def iter_items(agent, cases: Sequence[Dict[str, Any]], pool: ReferencePool, split: str,
               lengths: Sequence[int] = DEFAULT_LENGTHS, variants: Sequence[str] = DEFAULT_VARIANTS,
               seed: Any = 0, rule_version: str = RULE_VERSION) -> Iterator[Dict[str, Any]]:
    words = neutral_words(agent)
    for case in cases:
        refs = pool.ordered(case, seed) if any(v != "oracle" and v != "neutral@mid" for v in variants) else []
        for qid in case["questions"]:
            for variant in variants:
                if variant == "oracle":
                    yield build_item(agent, case, qid, "oracle", "original", split, refs, words, seed, rule_version)
                    continue
                for length in lengths:
                    yield build_item(agent, case, qid, variant, int(length), split, refs, words, seed, rule_version)


def canonical_line(item: Dict[str, Any]) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_items(path: Path, items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Write gzip JSON lines; returns the count and the fingerprint of the uncompressed content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as f:
        for it in items:
            line = canonical_line(it)
            h.update(line.encode("utf-8"))
            h.update(b"\n")
            f.write(line + "\n")
            n += 1
    return {"n_items": n, "fingerprint": h.hexdigest()}


def read_items(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def items_dir(fid: str, root: Optional[Path] = None) -> Path:
    return data.data_root(root) / "items" / fid


def build_split(agent, split: str, lengths: Sequence[int] = DEFAULT_LENGTHS,
                variants: Sequence[str] = DEFAULT_VARIANTS, seed: Any = 0, rule_version: str = RULE_VERSION,
                root: Optional[Path] = None, tokenizer_name: str = "", cases: Optional[Sequence[Dict[str, Any]]] = None,
                progress=None) -> Dict[str, Any]:
    """Build and write one split's items, and record its fingerprint in ``families.json``.

    ``final`` is built and fingerprinted like the others and never scored (FR-013).
    """
    man = data.read_data_json("manifest.json", root)
    fid = families_id(tokenizer_name, man["fingerprint"]["combined"], seed, lengths, variants, rule_version)
    pool = ReferencePool(data.load_cases("train", root))
    cases = list(cases) if cases is not None else data.split_cases(split, root)
    out_dir = items_dir(fid, root)
    counter = {"n": 0}

    def gen():
        for it in iter_items(agent, cases, pool, split, lengths, variants, seed, rule_version):
            counter["n"] += 1
            if progress and counter["n"] % 500 == 0:
                progress(counter["n"])
            yield it

    info = write_items(out_dir / ("%s.jsonl.gz" % split), gen())
    meta_path = out_dir / "families.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "schema_version": data.SCHEMA_VERSION, "families_id": fid, "rule_version": rule_version, "seed": seed,
        "lengths": list(lengths), "variants": list(variants), "markers": MARKERS,
        "neutral_vocab_version": NEUTRAL_VOCAB_VERSION, "tokenizer": tokenizer_name, "splits": {}}
    meta["splits"][split] = dict(info, n_cases=len(cases))
    data._write_json(meta_path, meta)
    return meta


def current_families_id(root: Optional[Path] = None, explicit: Optional[str] = None) -> str:
    """The families to use: the named one, or the most recently built."""
    base = data.data_root(root) / "items"
    if explicit:
        if not (base / explicit / "families.json").exists():
            raise FileNotFoundError("no families %s under %s" % (explicit, base))
        return explicit
    metas = sorted(base.glob("*/families.json"), key=lambda p: p.stat().st_mtime) if base.exists() else []
    if not metas:
        raise FileNotFoundError("no families built yet; run the families command")
    return metas[-1].parent.name

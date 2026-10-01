# Feature Specification: Tokenizer Goldens (laya:005)

**implements raya:004** (`raya/specs/004-tokenizer-goldens/spec.md`). Cross-reference as `laya:005` / `raya:004`.

**Feature Branch**: `raya/golden-generators` (local)
**Created**: 2026-10-01
**Status**: Implemented (see tasks.md)
**Depends on**: `laya:004` (`experiments/golden/` module). **Blocks**: raya 005.

**Input**: Add `python -m experiments golden tokenizer` which calls Laya's real sequence builder on a fixed list of
inputs and saves ids, positions, marker positions and truncation flags.

## User Scenarios & Testing

### User Story 1 - Sequence-Builder Goldens (P1)

`python -m experiments golden tokenizer --out artifacts/raya-golden/tokenizer/` writes one JSON record per fixed input:
document text, questions (types choice/score/noul), state, `max_len`, resulting `input_ids`, position ids, marker positions,
truncation flags and builder stats.

**Acceptance**:
1. Records exist for: empty document; one document plus one question (primary workload); one document plus several questions;
   with state (dict and conversation list); unicode; text containing marker-like strings (`[MASK]`, `[CLS]`, `[SEP]`);
   a document far longer than `max_len`; a document exactly `max_len`.
2. Repeated runs are byte-identical.
3. `tokenizer.json` used and `meta.json` (laya-sparse commit) are saved beside the records.

## Requirements

- **FR-001**: Call Laya's `build_sequence` (`laya/common.py`) and `Agent._encode_state` flow (`laya/agent.py`) via an `Agent` built on the fixture; never reimplement.
- **FR-002**: Use the fixture checkpoint's tokenizer. A real `tokenizer.json` only via `--checkpoint <dir>`, never in offline tests.
- **FR-003**: No Laya behavior is changed (no file under `laya/` is modified). Accidental-looking behavior is listed in `findings.md`.
- **FR-004**: Offline pytest `tests/experiments/test_golden_tokenizer.py` asserts byte-identical re-runs and coverage of every listed input.
- **FR-005**: Records capture stats (`options`, `options_distinct`, `tokens_per_option`), `truncate_left`, `head_max_len`, `max_len`.

## Success Criteria

- **SC-001**: All listed inputs covered, test passes offline. **SC-002**: No file under `laya/` modified.

## Clarifications (resolved by default, 2026-10-01)

- Q: "position ids". A: `build_sequence` returns no positions; Laya's model derives them (ModernBERT uses `arange(len)` internally). Records carry `position_ids = list(range(len(input_ids)))` and the marker positions are the builder's own; labeled as derived.
- Q: Truncation flags. A: Derived by comparing the untruncated lengths with the output: `state_tokens_total`, `state_tokens_kept`, `state_truncated` (bool), `truncate_left`, and `sequence_clipped` (the final `ids[:max_len]`).
- Q: Which flow produces a record? A: For each input the record has the per-question builder output obtained through `Agent._encode_state` (so state serialization, shared state tokenization and `truncate_left` for list states are exercised as in `predict`), and for the primary single-question case also a direct `build_sequence` call that must match.
- Q: Small `max_len`/`head_max_len` for the fixture. A: Each case sets its own values (e.g. 128) so truncation is exercised on a tiny model; the exact-`max_len` document is constructed by search so the sequence is exactly `max_len` without clipping.
- Q: Tokenizer file. A: `tokenizer.json` copied byte-for-byte from the fixture checkpoint's tokenizer directory next to the records, plus `special_tokens_map.json`/`tokenizer_config.json` are not required (ids of specials are in `meta.json`).

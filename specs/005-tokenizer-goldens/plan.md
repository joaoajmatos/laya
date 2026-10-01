# Implementation Plan: Tokenizer Goldens (laya:005, implements raya:004)

**Spec**: [spec.md](spec.md). Extends the `experiments/golden/` module of `laya:004`.

- `experiments/golden/tokenizer.py`: fixed input list (11 inputs: the 8 required plus a labelled-noul `several_questions`, a dict state, a
  conversation-list state, `max_len_plus_one` and `options_over_head_budget`), each run through `Agent._to_internal` + `Agent._encode_state` with
  a per-input `max_len` / `head_max_len`; single-question records are also checked against a direct `build_sequence` call.
- Output: `records/<name>.json`, `tokenizer.json` (byte copy of the one used), `meta.json` (commit SHA, special-token ids, vocab size).
- Derived fields (documented in the spec): `position_ids = range(len)`, `head_length`, `state_tokens_total/kept`, `state_truncated`, `sequence_clipped`.
  The derived state span is verified against the builder's output on every record or generation fails.
- The deterministic fixture tokenizer of `laya:004` is used; a real `tokenizer.json` only with `--checkpoint`.
- Behavior that looks accidental is in [findings.md](findings.md); nothing under `laya/` changed.

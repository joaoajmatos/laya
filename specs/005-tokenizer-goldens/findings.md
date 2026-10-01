# Findings: builder behavior that looks accidental (laya:005, FR-003)

Recorded, not changed. Raya must match Laya, so these are part of the contract unless the researcher decides otherwise.

1. **Only `[MASK]` is sanitized.** `build_sequence` replaces the literal mask token with a space in the state, instructions and options,
   but `[CLS]`, `[SEP]`, `[PAD]` and `[UNK]` typed in user text tokenize to the real special-token ids (record `marker_like_strings`:
   `[SEP]` and `[CLS]` appear inside the question head, `[PAD]` and `[UNK]` inside the state). A document can therefore inject separator or pad ids,
   and a `[PAD]` id inside the sequence is not masked out (the attention mask is all ones).
2. **No position ids come from the builder.** The model derives them; the records carry `range(len)` as a derived field.
3. **Truncation is silent.** `truncate_left=False` drops the end of the state, `True` (list states only) the start; no flag is returned.
   The records add `state_truncated` / `sequence_clipped` as derived fields. `Agent._encode_state` only raises if option markers are lost.
4. **The fixture vocabulary has no CJK or emoji pieces**, so the `unicode` record is mostly `[UNK]`; real merges need `--checkpoint <real dir>`.

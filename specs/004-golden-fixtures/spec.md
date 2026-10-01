# Feature Specification: Golden Fixtures (laya:004)

**implements raya:002** (`raya/specs/002-golden-fixtures/spec.md`). Cross-reference as `laya:004` / `raya:002`.

**Feature Branch**: `raya/golden-generators` (local)
**Created**: 2026-10-01
**Status**: Implemented (see tasks.md)
**Depends on**: nothing. **Blocks**: `laya:005`, raya 003, 006 and optionally 007.

> Numbering: `laya:003` is reserved for ContractNLI (human-owned timing), so this is the next free number.
> Not to be confused with `raya:004` (tokenizer goldens), which is implemented by `laya:005`.

**Input**: Export from Laya's tiny fixture model the contract Raya's Rust tests verify against: state_dict
inventory, per-layer activations, final logits/probabilities and kernel reference outputs. Offline, CPU only,
deterministic. No timing of any kind is recorded or claimed.

## User Scenarios & Testing

The user is the Raya implementer, who needs ground truth that localizes a divergence to a layer, a tensor or a kernel.

### User Story 1 - Weight Inventory (P1)

`python -m experiments golden weights-inventory` writes `inventory.json` listing every `state_dict` key with shape and
dtype for the fixture model. Behind `--checkpoint <dir>` it does the same for a real local checkpoint; without the flag no
real checkpoint is touched.

**Independent Test**: compare the file to `model.state_dict()`.

### User Story 2 - Forward-Pass Goldens (P1)

`python -m experiments golden export` writes one directory per case (`short`, `medium`, `padded`, `fastpath_off`) holding
`tensors.safetensors` (input ids, attention mask, marker positions/mask, qtype, embedding output, each encoder layer
output, final encoder output, `type_emb` addition, each decision-head layer output, gathered marker states, scorer
logits, act-head features, act logits, probabilities, and the weights) and `meta.json`.

**Acceptance**:
1. Each case directory holds every tensor above and `meta.json`.
2. The `fastpath_off` case matches the fast-path-on outputs within 1e-6, or export fails.
3. Two exports are byte-identical.

### User Story 3 - Kernel Goldens (P2)

`python -m experiments golden kernels` writes, per length (256, 1000, 2048; 16 heads, head dim 64), random q/k/v, the
reference output (`experiments/kernels/reference.py`), the PyTorch kernel output and the max abs diff, for `block_local`
(block 128) and `gather` (gather-attend-scatter). A diff above 1e-5 fails the command.

## Requirements

- **FR-001**: Module `experiments/golden/`, invoked as `python -m experiments golden <sub>`.
- **FR-002**: Subcommands `weights-inventory`, `export`, `kernels` (`tokenizer` is added by `laya:005`).
- **FR-003**: Runs offline on the tiny fixture from `tests/experiments/conftest.py` (`build_fixture_checkpoint`).
- **FR-004**: Default output `artifacts/raya-golden/` (git-ignored); only tiny-model cases (< 5 MB total) are vendorable.
- **FR-005**: Offline pytest `tests/experiments/test_golden.py` covers file presence, shapes, fast-path equality and byte-identical re-runs.
- **FR-006**: Decision-head goldens include `type_emb` addition, marker gathering, `scorer` logits and `act_head` logits,
  following `DecisionModel.forward` in `laya/common.py` (the source of truth).
- **FR-007**: Determinism: fixed seeds, fp32, `eval()`, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms`
  where supported, `torch.no_grad()`. Tensor files carry no timestamps; `meta.json` is written with sorted keys.
- **FR-008**: `meta.json` records the laya-sparse commit SHA, a dirty flag, model config, seed, torch/transformers/safetensors
  versions, threads, case parameters. Nothing machine- or time-dependent that would break byte identity.
- **FR-009**: No file under `laya/` is modified.

## Success Criteria

- **SC-001**: Pytest passes offline. **SC-002**: Two exports produce identical bytes. **SC-003**: Vendorable set < 5 MB.

## Clarifications (resolved by default, 2026-10-01)

- Q: Which "native-layer reference" for kernels? `laya/layers/attention.py` is empty (AGENTS.md). A: `experiments/kernels/reference.py:reference_attention`, the repo's definition of kernel semantics.
- Q: Spec says "block 128, +/-64". A: The repo's `block_local` pattern is a block-granular window (query block i sees blocks i-1, i, i+1, i.e. up to +/-128..255 keys depending on position). Export that pattern unchanged and record `pattern`, `block`, and the effective allowed mask description in `meta.json`; the `+/-64` wording is flagged as not matching the Python kernel (raya 007 must follow the Python semantics).
- Q: Gather-attend-scatter selection. A: `MaskSpec(pattern="gather", selection_blocks=4)` (default), recorded in meta.
- Q: Where does the fixture model come from at run time? A: Built into a temp directory by `tests/experiments/conftest.py:build_fixture_checkpoint` (seeded 1234), so the generator needs no committed weights. The fixture hidden=64 gives 1 head decision head, which never takes the fast path; the `fastpath_off` case therefore uses the hidden=128 variant (2 heads) so on/off are genuinely different code paths.
- Q: Padded case. A: batch-of-one with the input right-padded with `[PAD]` to a longer length and `attention_mask` zeros on the pad.
- Q: Fixture-model timings. A: Not recorded at all.
- Q: Case contents are produced how? A: A hand-run forward through `model.encoder`, `type_emb`, `head.layers`, `scorer`, `act_head` mirroring `DecisionModel.forward`; the final `logits, act_logits` are asserted equal to a real `model(...)` call.

---

description: "Task list for Decision Benchmark and Practical Baselines (Research Phase 2)"
---

# Tasks: Decision Benchmark and Practical Baselines (Research Phase 2)

**Input**: Design documents from `/specs/002-decision-benchmark-baselines/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cli.md](contracts/cli.md), [contracts/results.md](contracts/results.md)

**Tests**: Included. The plan specifies offline unit tests (quickstart step 0 depends on them). Test tasks come first within each story and should fail before the implementation they cover.

**Organization**: Grouped by user story. Paths are relative to the repository root. `laya/` is not edited by any task (plan.md, constitution I and IV); T066 verifies that. Tasks marked **RUN** execute real measurements on the measuring machine and can take hours; **MANUAL** tasks need the researcher.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 to US4, matching spec.md
- Result statuses are exactly: `measured`, `unsupported`, `failed`, `partial`. Evidence visibility values are exactly: `full`, `partial`, `none`
- Phase 1 modules (`results`, `manifest`, `runner`, `timing`, `tokens`, `inputs`, `cli`) are reused and extended, not copied. Fixtures `tiny_checkpoint`, `tiny_agent` and `fastpath_agent` already exist in `tests/experiments/conftest.py`

---

## Phase 1: Setup

**Purpose**: Repository housekeeping and the synthetic offline dataset

- [X] T001 Add `experiments/data/` to `.gitignore` so the dataset cache, derived items and split files are never committed (FR-003, FR-031). `experiments/results/` is already ignored
- [X] T002 Add `pyarrow>=14` to the `experiments` optional dependency group in `pyproject.toml` (next to `pytest>=7`) without changing `dependencies` or `[tool.setuptools] packages`
- [X] T003 Add a session-scoped fixture `tiny_upstream` to `tests/experiments/conftest.py` that writes a small synthetic dataset in the upstream schema to a temp directory as parquet (through `pyarrow`), laid out like the pinned repository: `all/train-00000-of-00001.parquet`, `all/test-00000-of-00001.parquet`, and the four per-workflow configs (`agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents`). Columns: `id` (string), `workflow`, `split`, `state` (JSON string of a dict), `questions` (JSON string, five questions: two `choice`, one `noul`, two ordinal `score`, each with `type`, `instructions`, `criteria`), `gold` (JSON string, per question `label`, `confidence`, `probabilities`, plus `noul` or `score`), `factors`, `label_agreement` (per question `argmax_agree`, `argmax_majority`, `total_variation`), `n_questions` (int64). 10 test cases and 12 train cases per workflow, varied state text and confidences (seeded). States use only words the tiny tokenizer fixture can encode, and one workflow's states are long enough to test `target_exceeds_length`. The fixture also exposes a `download` function pointing at this directory so tests never touch the network

---

## Phase 2: Foundational (blocks all user stories)

**Purpose**: Metrics, append-only result helpers, the final-split guard, and manifest extension

**CRITICAL**: No user story starts until this phase is complete

- [X] T004 [P] Write `tests/experiments/test_metrics.py` for `experiments/metrics.py`, against hand-computed values: exact-match accuracy (predicted label equals gold `label` for every type including ordinal `score`); mean absolute level error for `score`; classification ECE from the predicted answer's probability (ten equal-width bins; for `noul` the predicted answer is the more probable of true/false; for `score` the most probable level), and a test that entropy-based confidence is never used; temperature fit minimizes log loss on a small set and returns 1.0 for already-calibrated data; case-clustered paired bootstrap (resample cases, not questions, 5,000 resamples, seeded) recovers a known paired difference, is reproducible for a fixed seed and gives a wider interval when questions of one case are perfectly correlated than when treated independently; the verdict rule returns `better` (lower bound above 0), `worse` (upper bound below 0), `equal` (whole interval within ±2 points) or `inconclusive`
- [X] T005 Implement `experiments/metrics.py` per T004 and research.md R10: `accuracy`, `level_error`, `ece` (10 equal-width bins, predicted-answer probability), `fit_temperature(logits_or_probs, labels)`, `apply_temperature`, `paired_cluster_bootstrap(a_correct, b_correct, case_ids, n_boot=5000, seed)` returning difference and 95% interval, and `verdict(interval, margin_pp=2)`. NumPy only. Depends on T004
- [X] T006 [P] Add tests to `tests/experiments/test_results.py` for the Phase 2 helpers: `append_jsonl` appends one line per call and never rewrites earlier lines; `read_jsonl` skips a truncated last line from an interrupted write; `condition_id(name, params, variant, device, length)` is stable, formatted `<name>[.<params>].<variant>.<device>.L<length>`, and different for different parameters; `assert_no_final(items)` raises `SplitLocked` (reason `split_locked`, naming FR-013) for any item with `split: final`
- [X] T007 Extend `experiments/results.py` per T006 with `append_jsonl`, `read_jsonl`, `condition_id` and `assert_no_final`, plus the `SplitLocked` error and the refusal reasons of contracts/cli.md (`fingerprint_mismatch`, `split_locked`, `audit_required`, `audit_stale`, `reference_missing`, `device_mismatch`). Depends on T006
- [X] T008 [P] Add tests to `tests/experiments/test_manifest.py`: the run manifest gains a `data` block (dataset revision, fingerprints, split fingerprints, `families_id`), a `checkpoints` block (model id, pinned revision, `max_len`, `head_max_len`, positional capacity), and `variant` (`none`, `fastpath_off`, `int8_encoder`, `int8_all_nofast`); a missing `data` block is allowed for Phase 1 commands and marks the run `phase: 1`
- [X] T009 Extend `experiments/manifest.py` per T008 without changing Phase 1 fields or their meaning, and record `pyarrow`, `torch` quantization API version and `mha` fast-path setting. Bump nothing in `schema_version` for existing files; Phase 2 files start at `schema_version` 1. Depends on T008

**Checkpoint**: `pytest tests/experiments/test_metrics.py tests/experiments/test_results.py tests/experiments/test_manifest.py` passes

---

## Phase 3: User Story 1 - Upstream Data Intake and Length Profile (Priority: P1) MVP

**Goal**: Import the pinned upstream dataset with a manifest and fingerprint, create the frozen splits, profile token lengths under Laya's tokenizer, and run both native checkpoints on original-length dev cases to decide which is a valid quality reference.

**Independent Test**: On the tiny fixture (offline) and on the real pinned data (`slow`): counts, splits, workflows and question types match the source; a tampered cache is refused; token counts reconcile for every case; both checkpoints produce accuracy, calibration and per-workflow and per-type results; low-confidence cases are reported separately.

**Note on ordering**: the split manifest is built here, not in US2, because the solvability check needs the dev split. US2 verifies its properties and builds items on it.

### Tests for User Story 1

- [X] T010 [P] [US1] Write `tests/experiments/test_data.py` on `tiny_upstream`: import records source, revision, license, subsets, splits, case counts, question counts by type, and a content fingerprint (data-model.md "Data Manifest"); the fingerprint is deterministic and independent of row order; a modified cached row makes `import_dataset` raise `fingerprint_mismatch` (FR-002); per-workflow configs are cross-checked against `all` and a disagreement raises; the low-confidence cutoff equals the 25th percentile of the **train** split's question confidences and is stored with `basis: "train"` and `n_questions`; a question is low-confidence when its gold `confidence` is at or below the cutoff, applied to test questions (R13); `label_agreement.argmax_agree` false and top-quartile `total_variation` are exposed as extra strata; `factors` is never included in the model-input state; no raw record is written outside `experiments/data/`. Splits: `make_splits(seed)` gives per workflow the first 30% / next 20% / last 50% of the seeded shuffle of test case ids (scaled to the fixture's 10 cases per workflow), all disjoint and covering the test split, identical on repeat, `final` fingerprinted; a second call with a different result refuses to overwrite; `half_sample` per split is stratified by workflow and a subset of the split; `latency_sample` is 12 dev items (three per workflow) in the full-size case and `variant_sample` is 20 dev cases
- [X] T011 [P] [US1] Write `tests/experiments/test_lengths.py` on the tiny fixture and `tiny_upstream`: per case and question, `row_tokens == special_tokens + task_option_tokens + state_tokens` (reconciled with `tokens.account`); per-workflow `min`, `median`, `p90`, `p99`, `max` match a hand computation; `fit_share` for 512, 1,024, 2,048, 4,096, 8,192 is the share of rows with `row_tokens <= N`; questions whose options exceed `head_max_len` are listed under `over_head_budget` and never dropped
- [X] T012 [P] [US1] Write core cases in `tests/experiments/test_evalrun.py`: `run_items(agent, items, condition)` on original-variant items makes one `predict` call per question row, and returns `predicted`, `probabilities`, `prob_predicted`, `correct` (exact match with gold), `level_error` (ordinal only), `tokens_seen`; a `choice` gold label maps back correctly when options were permuted; `solvability()` on hand-made predictions sets `solves` true only when accuracy exceeds the per-question-type majority accuracy computed from the **train** split with a paired case-clustered interval that excludes zero, and sets `reference_checkpoint` to `none` when neither checkpoint solves (FR-006); the report strata include workflow, question type, `low_confidence`, `argmax_agree` false and top-`total_variation` quartile

### Implementation for User Story 1

- [X] T013 [US1] Implement import in `experiments/data.py`: `import_dataset(revision, cache_dir, download=None)` downloads the `all` and per-workflow parquet files of `LocalLLaMA/typed-decisions` at the pinned dataset revision `d51d993547ad8355b1c25157fbc1fea0649e8ffa` with `huggingface_hub`, reads them with `pyarrow`, parses JSON columns, checks case ids and counts across configs, computes the fingerprint (SHA-256 over canonical JSON, sorted keys, no whitespace, every row sorted by `id`, per split then combined; research.md R1), computes the low-confidence cutoff from train (R13), records the checkpoint revisions (`laya.revisions.PINNED_REVISIONS`) and `max_len`/`head_max_len` when available, and writes `experiments/data/manifest.json` and `experiments/data/cache/`. Import `pyarrow` lazily inside the parquet-reading function only, so modules that merely load prebuilt items (for example GPU latency from `.venv-gpu`, which has no `pyarrow`) import cleanly. Refuses with `fingerprint_mismatch` when a recorded fingerprint differs. `load_cases(split)` returns Upstream Case records with `n_options` and `low_confidence` per question. Depends on T007, T010
- [X] T014 [US1] Implement splits in `experiments/data.py`: `make_splits(seed)` writes `experiments/data/splits.json` per research.md R2 (`rule: stratified_by_workflow_30_20_50`), with `case_ids`, `n_cases`, `n_per_workflow` and `fingerprint` for `dev`, `calibration` and `final`, plus `half_sample` per split (research.md R11), `latency_sample` and `variant_sample`; refuses to overwrite a different existing manifest; `check_splits()` re-verifies against the source. Also add `original_items(split)`, which builds one `original` variant Evaluation Item per (case, question) directly from the Upstream Case (the unwrapped state exactly as the native path serializes it, `length: "original"`, `status: ok`), so US1 does not depend on `families.py`; T029 reuses it. Depends on T013
- [X] T015 [P] [US1] Implement `experiments/lengths.py`: per case and question `row_tokens` split into `state_tokens`, `task_option_tokens`, `special_tokens` under the target checkpoint's tokenizer using `tokens.account` at `max_len` large enough to avoid truncation; aggregates per workflow, per split and overall; `fit_share` for 512, 1,024, 2,048, 4,096, 8,192; `over_head_budget`. Writes `experiments/data/length_profile.json`. Depends on T013
- [X] T016 [US1] Implement the core runner in `experiments/evalrun.py`: `run_items(agent, items, condition)` for the `native` condition on `original` items from `data.original_items` (one `predict` call per question row, `max_len` = the checkpoint's configured cap), computing `predicted`, `probabilities`, `prob_predicted` (the predicted answer's probability), `correct`, `level_error`, `tokens_seen`; appends each result to `quality/<condition_id>/predictions.jsonl` with `append_jsonl` as it finishes and skips items already present (resume); records `status` (`measured`, `unsupported`, `failed`) and `reason`. CPU only: raises `device_mismatch` if the agent is not on CPU. Quality results carry `device: "cpu"`. Depends on T005, T007, T012
- [X] T017 [US1] Implement `solvability(models, split)` in `experiments/evalrun.py`: run the fine-tuned (`convaiinnovations/laya-typed-decisions`) and base (`convaiinnovations/laya`) checkpoints on original-length dev items (each checkpoint at its configured cap; FR-007), compute per-checkpoint accuracy per question type and workflow, `mean_abs_level_error`, `ece_raw`, the strata of T012, the per-question-type majority baseline from train, `paired_vs_majority` (case-clustered interval) and `solves` (research.md R14), and write `solvability.json` with `reference_checkpoint` (`none` when no checkpoint solves; later comparisons are then labeled as lacking a valid reference). Depends on T014, T016
- [X] T018 [US1] Register `data-import`, `length-profile`, `splits` and `solvability` in `experiments/cli.py` per contracts/cli.md: `data-import` (`--revision`, `--refresh`; exits non-zero with `fingerprint_mismatch`), `length-profile` (`--model`), `splits` (`--seed`, `--check`), `solvability` (`--models`, `--split dev`). Each appends its command line to `commands.txt` and prints the output path. Depends on T013, T014, T015, T017
- [X] T019 [P] [US1] Add `slow` integration tests to `tests/experiments/test_real_checkpoint.py` on the real data and checkpoints: 1,200 train and 400 test cases, 2,000 test questions, 100 test cases per workflow; re-import reproduces the fingerprint; the dev split has 120 cases (30 per workflow); both checkpoints run on 10 original-length dev rows on CPU with `device: cpu`, and the fine-tuned checkpoint's `max_len` is 1,024 and `head_max_len` 256
- [X] T020 [US1] **RUN**: on the measuring machine execute `python -m experiments data-import`, `length-profile`, `splits` and `solvability --run-id p2-dev` on the real data. Confirm the manifest counts, that the length profile's original-length min, median and max are near the 2026-09-29 spike values (124, 308 and 597 tokens; a large difference is a finding to record in research.md), and read `solvability.json` for the reference decision (SC-002). Depends on T018, T019

**Checkpoint**: User Story 1 works alone. `data-import`, `length-profile`, `splits` and `solvability` run on the fixture and on real data; T010 to T012 pass.

---

## Phase 4: User Story 2 - Controlled Long-Context Families and Frozen Splits (Priority: P2)

**Goal**: Build evaluation items at 512, 1,024, 2,048, 4,096 and 8,192 total tokens with neutral, near-matching-distractor and position families, verify them, hand-audit them, and leave the final split built but unscored.

**Independent Test**: Generate the families on the fixture and on real data. Every item's row length is exactly its named length, every item records its evidence span, no case is in two splits, distractors come only from the training split, and a hand audit of at least 50 items shows 0% answer changes.

### Tests for User Story 2

- [X] T021 [P] [US2] Write `tests/experiments/test_bm25.py` for `experiments/bm25.py`: ranking on a tiny corpus matches a hand computation of BM25 (k1 1.5, b 0.75); ties break by index so results are deterministic; an empty query returns zero scores; tokenization is lowercase word tokens
- [X] T022 [P] [US2] Write `tests/experiments/test_families.py` on `tiny_agent` and `tiny_upstream`, covering: **exact length** (every item at each supported length has row length equal to its named length by `tokens.account`; a miss raises instead of recording a wrong length, FR-009, SC-003); **target unchanged** (the target record text in `state_text` equals `serialize_state(state)` byte for byte); **evidence span** (the tokens at `evidence_span` decode to the target record, `row_start`/`row_end` and `state_start`/`state_end` agree, and 100% of items have one, FR-010); **markers** (the target is introduced by `[RECORD UNDER REVIEW]`, references by `[REFERENCE RECORD, NOT UNDER REVIEW]`, neutral text by `[BACKGROUND NOTE]`; FR-011); **train-only** (`distractor_case_ids` are all train-split ids of the same workflow, never a test case, never the target; each reference record used at most once per item); **near-matching** (references are drawn from the top 20 by BM25 similarity to the target); **positions** (begin: target starts within the first 5% of state tokens; end: ends within the last 5%; mid: centered; the three position variants share the same distractor set and differ only in order); **shared context and adjustment** (all questions of one case share the same context records; the adjustment segment of neutral filler is at the very start of the state, is at most about 150 tokens, and never lies between the target and its marker); **counterbalancing** (`choice` option order is permuted with a seed and the gold label is remapped, `option_order` recorded; `noul` and `score` are never permuted; FR-015); **leakage** (neutral padding uses a vocabulary with no workflow terms and `inputs.FILLER_WORDS` is not used; an identifier or value string of the target never appears in a reference record; gold-label frequency does not depend on padding length or position); **unsupported** (a row whose original length exceeds the named length gets `status: unsupported` with `reason: target_exceeds_length`, never truncated); **determinism** (same seed and rule version give identical items and `families_id`); **split inheritance** (every variant of a case carries its split and no case is in two split files)
- [X] T023 [P] [US2] Write `tests/experiments/test_audit_items.py`: pre-checks (exact length, span integrity, marker presence, label leakage, identifier reuse) run before the sheet and stop on a failure; `audit-sample` draws at least 50 items (default 60), stratified across family, length and workflow, from the dev split only, seeded and recorded; the sheet lists target record, gold, layout with markers and the three rubric questions of research.md R12; `audit-record` computes `share_answer_changed` and `share_ambiguous`, binds the result to the current `families_id`, and sets `passed` true only at 0% answer changes and 0% ambiguous; a result bound to an older `families_id` is `audit_stale`; a missing result is `audit_required`
- [X] T024 [P] [US2] Add final-lock cases to `tests/experiments/test_evalrun.py`: `families` accepts `--split final` and writes items and a fingerprint; `assert_no_final` on a results directory containing a `final` item raises `split_locked`. The `eval`/`latency` refusals are tested in T037 and the `final_scored_items` count in T052, since those commands come later

### Implementation for User Story 2

- [X] T025 [P] [US2] Implement `experiments/bm25.py`: a small pure-Python BM25 (k1 1.5, b 0.75) with lowercase word tokenization, deterministic ordering, `rank(query, documents)` returning scores; no dependency. Add `bm25.py` to the plan.md source tree. Depends on T021
- [X] T026 [US2] Implement in `experiments/families.py` the constants and neutral text: the fixed marker strings, the neutral-padding vocabulary (a filtered word list containing no workflow terms; `NEUTRAL_VOCAB_VERSION`) and a seeded generator that produces neutral paragraphs to an exact token budget by binary search plus single-token top-up, reusing the approach of `inputs._filler_state` but not its word list. Depends on T022
- [X] T027 [US2] Implement distractor selection in `experiments/families.py`: reference records come only from the upstream **train** split, same workflow as the target, ranked by `bm25.rank` over the serialized states and drawn by seeded sampling without replacement from the top 20; reject records that share an identifier or value string with the target; never reuse a record within an item. Depends on T025, T026
- [X] T028 [US2] Implement layout and exact length in `experiments/families.py`: given (case, question, variant, length, seed) place the target record (unchanged) with its marker, reference records or neutral paragraphs around it per position (begin within the first 5%, mid centered, end within the last 5%), then absorb the per-question head difference with an adjustment segment at the very start of the state, sized by tokenizing until `tokens.account` reports exactly the named length (a miss raises). Record `evidence_span` `{state_start, state_end, row_start, row_end}`, `layout` segments (`kind`: `target`, `reference`, `neutral`, `adjust`), `accounting`, and `status: ok`, or `status: unsupported` with `reason: target_exceeds_length`. Depends on T027
- [X] T029 [US2] Implement item generation in `experiments/families.py`: `build_items(split, lengths, variants, seed, rule_version)` for `neutral@mid`, `distractor@mid`, `distractor@begin`, `distractor@end`, plus `original` (the unwrapped case as the native path sees it) and `oracle` (the target record with only its marker; the framing control, research.md R4). Permute `choice` option order with a seed and remap the gold label (`option_order`), never `noul` or `score`. Write `experiments/data/items/<families_id>/<split>.jsonl` and `families.json` (rule version, seeds, per-split fingerprints); `families_id` is the fingerprint of rule version and seeds. Every item follows data-model.md "Evaluation Item" with `construction` `{seed, rule_version, marker_constants, distractor_case_ids, neutral_vocab_version}`. Depends on T014, T028
- [X] T030 [P] [US2] Implement `experiments/audit_items.py`: pre-checks per T023; `audit_sample(n=60, seed)` (never below 50) stratified by family, length and workflow from the dev split, writing `audit_sheet.json` and `audit_sheet.md`; `record_audit(path)` writes `audit_result.json` with per-item `changes_answer`, `ambiguous`, `evidence_intact`, `note`, `share_answer_changed`, `share_ambiguous`, the audited `families_id` and `passed` (true only at 0% changed and 0% ambiguous); `require_audit(families_id)` raises `audit_required` or `audit_stale`. Depends on T023, T029
- [X] T031 [US2] Register `families`, `audit-sample` and `audit-record` in `experiments/cli.py` per contracts/cli.md (`--split dev|calibration|final`, `--lengths` default `512,1024,2048,4096,8192`, `--variants`, `--seed`, `--rule-version`, `--n`, `--from`). `families --split final` builds and fingerprints only. Depends on T029, T030
- [X] T032 [P] [US2] Add `slow` tests to `tests/experiments/test_real_checkpoint.py`: on the real tokenizer and real dev cases, every `distractor@mid` item at 512, 2,048 and 8,192 reaches its exact length, cases whose longest row exceeds 512 are `unsupported` at 512 and nowhere else, and the 8,192-token item fits the checkpoint's positional capacity
- [X] T033 [US2] **RUN**: `python -m experiments families --split dev`, `--split calibration` and `--split final`, then `audit-sample`. Check `families.json` fingerprints, that 100% of items record an evidence span and that no case is in two split files (SC-003). Depends on T031, T032
- [X] T034 [US2] **MANUAL**: the researcher reviews at least 50 items in `audit_sheet.md`, writes `audit_result_draft.json`, and runs `python -m experiments audit-record --from audit_result_draft.json`. If any item changes the answer or is ambiguous, fix the construction rule in `experiments/families.py`, bump `rule_version`, rerun T033 and audit a fresh sample until `share_answer_changed` is 0% (SC-004). Record the audit outcome in research.md. Depends on T033

**Checkpoint**: User Stories 1 and 2 work. Families are built for all three splits, audited with `passed: true`, and the final split is fingerprinted and unopened.

---

## Phase 5: User Story 3 - Native and Baseline Quality-Cost Curves on CPU and GPU (Priority: P3)

**Goal**: Run native Laya and the simple baselines on the dev split at every length, with the optimized-system variants measured apart, and collect quality, calibration, latency and memory on CPU plus latency on GPU.

**Independent Test**: Run every baseline on the dev split at every supported length on the fixture. Each result row carries quality, calibration, latency, memory and evidence-visibility records, and unsupported or failed conditions appear with reasons. A rerun resumes without redoing finished items.

### Tests for User Story 3

- [X] T035 [P] [US3] Write `tests/experiments/test_baselines.py` on `tiny_agent`: `trunc512` and `truncCap` set `max_len` to 512 and to the configured cap and use the default right cut, and evidence visibility is `full`, `partial` or `none` from the retained state tokens (target at the end of a long item gives `none`); `window` results equal `agent.predict_long` with the tuned window, stride = window // 2 and `batch_size=1` above 2,048 tokens, carry `probability_scope: "deciding_window"`, and visibility is `full` when any window holds the whole evidence span and `partial` for the best partial coverage; `retrieve<B>` cuts the state into 64-token chunks with 50% overlap, ranks them by BM25 against the question's instructions plus option texts and never the marker text, keeps the best chunks up to B retained tokens (B in {512, 1,024, 2,048}), restores original order, never exceeds B, and maps visibility back to state positions; `oracle` input equals the target record with only its marker (a diagnostic control, marked `deployable: false`, FR-020); no baseline builds an input longer than its `max_len` or beyond positional capacity, and inputs above the configured cap are labeled `beyond_configured_max_len`; retrieval and window splitting happen inside the timed call
- [X] T036 [P] [US3] Write `tests/experiments/test_variants.py`: `fastpath_off` sets `torch.backends.mha` fast-path off in the process and restores it; on `fastpath_agent` (whose head takes the fast path) outputs under `fastpath_off` agree with native within 1e-4 on probabilities; `int8_encoder` replaces only encoder `torch.nn.Linear` layers with `qint8` dynamic quantization and leaves the head and its fast path native; `int8_all_nofast` quantizes all Linear layers and requires the fast path off (with it on, the run is refused with the recorded cause, matching the `'function' object has no attribute 'device'` failure of research.md R14); quantized runs on GPU are `unsupported` with that reason; variants are never merged into baseline rankings (`variant` field set)
- [X] T037 [P] [US3] Extend `tests/experiments/test_evalrun.py`: resume skips items already in `predictions.jsonl` and produces identical final files; a child that fails or times out records the affected condition `failed` with `cause` and the run continues (FR-025); a failed or unsupported item is never replaced by another length, device or configuration; aggregates compute accuracy over `measured` items only and show the denominator; `eval` refuses without a passing audit for the current `families_id`; `--tune` chooses window size from {256, 512, default 760} and retrieval budget from {512, 1,024, 2,048} using **dev only**, ties broken by lower cost, and writes them to `conditions.json` with `tuned_on: "dev"` before any calibration-split run; the tier rule (research.md R11) runs 512, 1,024, 2,048 on all cases of a split and 4,096, 8,192 on the recorded `half_sample`; quality runs make one forward pass per question row (no batching of rows, FR-023); the `--split final` refusal of `eval` and `latency` (`split_locked`, naming FR-013) is checked here; `condition_id` is stable across reruns
- [X] T038 [P] [US3] Write `tests/experiments/test_calibration_summary.py`: temperatures are fitted per (question type, condition) on calibration-split results only and applied to dev results of the same condition, and fitting on dev data raises; raw and scaled ECE are both reported; `window` scaled ECE is labeled non-comparable; `eval-summary` produces per-condition accuracy per question type, workflow and stratum, `mean_abs_level_error`, counts by `status` and by `evidence_visible`, and paired differences against `native` at matching length, `trunc512`, `truncCap` and `oracle` with case-clustered intervals and a verdict; underpowered comparisons are labeled `inconclusive` (FR-022); items with `evidence_visible` `full` and the rest are reported separately
- [X] T039 [P] [US3] Write `tests/experiments/test_latency.py`: the latency sample is the fixed 12 dev items and uses the clean path (`timing.assert_clean`, no hooks or profiler); result fields follow data-model.md "Cost Result" (`device`, `sample`, `repeats`, `warmup`, `p50_ms`, `p95_ms`, `peak_rss_bytes`, `peak_commit_bytes`, `status`, `reason`, `drift`, `fallbacks`); a GPU request without CUDA gives `device_mismatch`; `int8_*` on GPU is `unsupported`; GPU files are written under `gpu_latency/` and never mixed into `latency/`; CPU latency includes tokenization, selection or windowing, encoding, scoring and postprocessing (constitution VI)

### Implementation for User Story 3

- [X] T040 [US3] Implement `native`, `trunc512`, `truncCap`, `window` and `oracle` in `experiments/baselines.py` as transformations in front of the unchanged `Agent`: each returns the call settings (`max_len`, state, or a `predict_long` call) plus the record of tokens seen and evidence visibility (`full`, `partial` or `none`, with `evidence_fraction`) per research.md R7 and R8, and labels `beyond_configured_max_len` (FR-008). Depends on T035, T029
- [X] T041 [US3] Implement `retrieve<B>` in `experiments/baselines.py` per research.md R9: 64-token chunks with 50% overlap, BM25 over the question's instructions plus option texts (never the marker text), best chunks up to B tokens in original order, visibility mapped back to state positions, selection time inside the timed call. Chunk size and overlap are fixed, only B is tuned. Depends on T025, T040
- [X] T042 [P] [US3] Implement `experiments/variants.py`: `fastpath_off` (context manager over `torch.backends.mha`, recorded as `runtime.variant`), `int8_encoder` and `int8_all_nofast` via `torch.ao.quantization.quantize_dynamic(..., {torch.nn.Linear}, dtype=torch.qint8, inplace=True)` applied to the loaded model outside `laya/`, with the fast-path guard of T036, the torch version recorded, and the deprecation notice suppressed but noted in the manifest. Depends on T036
- [X] T043 [US3] Implement the full quality run in `experiments/evalrun.py`: conditions × variants × lengths on `dev` or `calibration`, each condition in its own subprocess through `runner.run_condition` with a `--time-cap` (default 900 s), resume from `predictions.jsonl`, the tier rule and half-sample of research.md R11, the variant sample (20 dev cases at 512, 2,048 and 8,192) for optimized variants, refusal of `--split final` (`split_locked`) and of a missing or stale audit, and `--tune` writing `conditions.json`. Windowed calls use `batch_size=1` above 2,048 tokens, and peak memory is recorded per condition (Phase 1 saw a 10.9 GB peak for one native 8,192-token request). Item `status` is `measured`, or `unsupported`/`failed` with `reason`; `evidence_visible` is one of `full`, `partial`, `none`; `probability_scope` is `deciding_window` for `window`. Depends on T037, T040, T041, T042, T016, T030
- [X] T044 [US3] Implement `calibrate` and `eval-summary` in `experiments/evalrun.py` and `experiments/metrics.py`: fit temperatures on calibration-split predictions only (`calibration.json`), apply them to dev results, compute the summaries and paired comparisons of T038 with `metrics.paired_cluster_bootstrap` and `metrics.verdict`, and write `quality/<condition_id>/summary.json` and `summary.json` containing every (condition, length) cell, with statuses. Depends on T038, T043
- [X] T045 [US3] Implement CPU latency in `experiments/latency.py` on top of `timing.measure_in_process`: for each condition run the fixed latency sample (12 dev `distractor@mid` items, three per workflow) one document and one question per call, warmup 1, repeats by `timing.default_repeats` capped by `--time-cap`, drift canary as in Phase 1, peak memory from `runner`, and windowing and retrieval inside the timed call. Writes `latency/<condition_id>.json`. Depends on T039, T040, T041
- [X] T046 [US3] Implement GPU latency in `experiments/latency.py`: the same sample on CUDA from the `.venv-gpu` environment (`experiments/gpu.py` helpers), Laya's native CUDA precision defaults, `peak_gpu_bytes`, written to `gpu_latency/<condition_id>.json`; `int8_*` and `fastpath_off` variants recorded `unsupported` with the reason; never substitutes for a CPU result and no CPU cost figure reads these files (FR-024). Depends on T045
- [X] T047 [US3] Register `eval`, `calibrate`, `latency` and `eval-summary` in `experiments/cli.py` per contracts/cli.md (`--conditions`, `--variants`, `--lengths`, `--split`, `--tune`, `--time-cap`, `--resume`, `--device cpu|gpu`, `--repeats`, `--warmup`, canary options). Depends on T043, T044, T045, T046
- [X] T048 [P] [US3] Add `slow` tests to `tests/experiments/test_real_checkpoint.py`: on the real fine-tuned checkpoint, `native` runs an 8,192-token item within positional capacity and labeled `beyond_configured_max_len`; `fastpath_off` predictions agree with native within 1e-4; both int8 variants run, and their probability shift from native is recorded (the spike saw about 0.02) and is reported, not asserted small
- [ ] T049 [US3] **RUN**: tune on dev, then quality runs, in this order, with `--run-id p2-dev`: `eval --split dev --tune`; `eval --split dev` (all conditions, tier rule); `eval --split calibration`; `calibrate`. Expect this to take many hours (estimated, research.md R11); it resumes after interruption. Depends on T034, T047, T048
- [ ] T050 [US3] **RUN**: `latency --device cpu` for all conditions and lengths, then the variant runs `eval --split dev --variants fastpath_off,int8_encoder,int8_all_nofast`, then `latency --variants ...`. Do not run other heavy work during latency runs (drift, Phase 1 R14). Depends on T049
- [ ] T051 [US3] **RUN**: from the GPU environment, `.venv-gpu\Scripts\python.exe -m experiments latency --device gpu --run-id p2-dev`, then `eval-summary`. Confirm every (condition, length) cell is measured, unsupported, failed or partial with a reason (SC-005, SC-006). Depends on T050

**Checkpoint**: User Stories 1 to 3 work. Every baseline has a result at every length on the dev split, either measured or marked with a reason, with CPU and GPU latency reported apart.

---

## Phase 6: User Story 4 - Phase 2 Report and Frozen Evaluation Plan (Priority: P4)

**Goal**: One report with quality-against-cost curves for every baseline at every length on CPU and GPU, the native-versus-truncation verdicts, the Phase 3 evidence table, and the frozen final-test protocol.

**Independent Test**: Every curve point and claim traces to a result file, every statement is labeled measured, estimated or hypothesized, and the frozen plan exists with `final_scored_items: 0`.

### Tests for User Story 4

- [X] T052 [P] [US4] Write `tests/experiments/test_evalplan.py`: the required-case computation on synthetic paired differences with a known intra-case correlation matches a hand or closed-form check within simulation tolerance; when the available final cases (200) cannot resolve the -2 point margin at 80% power the plan sets `resolvable: false` and states the cases needed, never lowering the margin (FR-030); the plan records metrics, `margin_pp` 2, comparisons, the verdict rule, pilot differences and a fingerprint; a frozen plan is not overwritten without `--new-version` and a reason; `freeze-plan` fails when any result names a final-split item and reports `final_scored_items: 0` on a clean directory (SC-008)
- [X] T053 [P] [US4] Write `tests/experiments/test_report2.py`: every statement carries `[measured]`, `[estimated]` or `[hypothesized]`; each curve point links to its result file; CPU curves never read `gpu_latency/` and GPU appears only in GPU-labeled tables (contracts/results.md rules 4 and 7 analogues); the native-versus-truncation verdicts at 2K, 4K and 8K follow `metrics.verdict` on the paired intervals; the -2 point statement names a baseline (or says none) with its lower CPU cost; oracle is labeled a diagnostic control and never ranked; optimized variants are in their own table, with the quality change against native stated including when it is zero; the report lists limits (synthetic and short upstream data, constructed long contexts, test split already public, fine-tuned checkpoint trained on the distractor pool, no realistic long documents, decoder baseline deferred) and reproduce commands for every result; `final_scored_items` is 0

### Implementation for User Story 4

- [X] T054 [US4] Implement `experiments/evalplan.py`: derive the intra-case correlation and pilot paired differences from the dev results, compute the required number of final cases for the -2 point margin at 80% power with a 95% interval by a seeded case-clustered simulation (research.md R16), and write `evaluation_plan.json` and `evaluation_plan.md` with `frozen_at`, `fingerprint`, `metrics`, `margin_pp`, `comparisons`, `pilot`, `required_cases`, `available_cases`, `power`, `resolvable` and `final_scored_items`. Depends on T044, T052
- [X] T055 [US4] Implement report sections in `experiments/report2.py`: data manifest and length profile (share of cases within 512, 1,024, 2,048, 4,096, 8,192 per workflow), solvability and the reference decision, the hand-audit result, low-confidence and agreement strata, and the over-head-budget cases (questions whose options exceed `head_max_len`) reported separately. Depends on T044, T053
- [X] T056 [US4] Implement report curves and verdicts in `experiments/report2.py`: accuracy and calibration against CPU latency, and against GPU latency, per baseline and length with paired case-clustered intervals; the native-versus-`trunc512`/`truncCap` verdict at 2K, 4K and 8K (`better`, `equal`, `worse`, `inconclusive`); whether any baseline reaches native matching-length accuracy within -2 percentage points at lower CPU cost (FR-028); optimized-variant table; the Phase 3 evidence table (sparse attention, selection, compression, none: supported, weakened or open, each row linked to a result); labels, limits and reproduce commands. Windowed probabilities are marked non-comparable. Depends on T055
- [X] T057 [US4] Register `freeze-plan`, `phase2-report` and `phase2-all` in `experiments/cli.py` per contracts/cli.md; `phase2-all --dry-run` prints the ordered plan and the estimated wall time (labeled estimated) and the real run stops at the audit gate and at any failure. Depends on T054, T056
- [ ] T058 [US4] **RUN**: `python -m experiments freeze-plan --run-id p2-dev`, then `phase2-report --run-id p2-dev`. Read `report2.md` against the raw results, confirm `final_scored_items: 0` and that a SC-009 decision on Phase 3 directions can be made from the report alone. Depends on T051, T057

**Checkpoint**: All four user stories work. The report and the frozen plan exist.

---

## Phase 7: Polish and Cross-Cutting

- [X] T059 [P] Update the Phase 2 block of `docs/research-plan.md` and the README status line with the measured findings and a pointer to `report2.md` (edit only the Phase 2 text; do not touch unrelated existing changes in the working tree)
- [X] T060 [P] Record in `specs/002-decision-benchmark-baselines/research.md` any assumption a run contradicted (row lengths, quantization behavior, timing estimates, half-sample sizes), following the R16 practice of Phase 1
- [X] T061 Run the full offline suite `python -m pytest tests/experiments -m "not slow"` and the `slow` suite on the measuring machine; fix failures, and report any check that could not be run
- [X] T062 Run `git diff --stat -- laya/` and confirm it is empty (plan; constitution I, IV); confirm `git status` shows nothing under `experiments/data/` or `experiments/results/` (FR-003, FR-031)
- [ ] T063 Run `quickstart.md` end to end and tick its success-check table (SC-001 to SC-009), including SC-001: delete `experiments/data/` and reproduce the manifest, length profile, splits and items with matching fingerprints

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none
- **Foundational (Phase 2)**: after Setup; blocks all user stories
- **US1**: after Foundational. Owns data import, splits, length profile and solvability
- **US2**: after Foundational and US1 (needs imported data, splits and the tokenizer path)
- **US3**: after US2 (needs audited families) and US1 (needs the core runner and the reference decision)
- **US4**: after US3 results
- **Polish**: after the stories

### Within Foundational

T004, T006, T008 in parallel; T005 after T004; T007 after T006; T009 after T008.

### Within Each Story

Tests before implementation. Library modules before CLI registration. Edits to the same file run in order: `experiments/data.py` (T013, T014), `experiments/families.py` (T026, T027, T028, T029), `experiments/evalrun.py` (T016, T017, then T043, T044), `experiments/baselines.py` (T040, T041), `experiments/latency.py` (T045, T046), `experiments/report2.py` (T055, T056) and `experiments/cli.py` registration tasks (T018, T031, T047, T057). RUN and MANUAL tasks follow their story's implementation and tests.

### Parallel Opportunities

- US1: T010, T011, T012 together; T015 with T014; T019 with T018
- US2: T021 to T024 together; T025 with T026; T030 with T029 once T023 exists; T032 with T031
- US3: T035 to T039 together; T042 with T040 and T041; T048 with T047
- US4: T052 and T053 together
- Polish: T059 and T060 together

## Parallel Example: User Story 3 tests

```text
Task: "Write tests/experiments/test_baselines.py (T035)"
Task: "Write tests/experiments/test_variants.py (T036)"
Task: "Extend tests/experiments/test_evalrun.py (T037)"
Task: "Write tests/experiments/test_calibration_summary.py (T038)"
Task: "Write tests/experiments/test_latency.py (T039)"
```

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phases 1 and 2. 2. Phase 3 (US1). 3. Stop and validate: `data-import`, `length-profile`, `splits` and `solvability` on the real data (T020). This alone answers how long Laya's real inputs are, and which checkpoint is a valid quality reference.

### Incremental Delivery

Add US2 for controlled families and the hand audit (a gate before any comparison), then US3 for the quality-cost curves (the central measurement), then US4 to turn them into the Phase 3 decision and the frozen final-test plan. Each increment produces usable files without the next one.

## Notes

- No task edits `laya/`. If a needed probe cannot be reached from outside the library, stop and record the finding (plan.md contingency).
- The final split is built and fingerprinted, never scored, in every task (FR-013, SC-008).
- GPU-scored quality with a CPU parity subset replaced the optional GPU quality check (Phase 8).
- Timings from the fixture only exercise the tooling; only runs on the pinned checkpoints on the measuring machine are reported.
- Compute-time figures in research.md R11 are estimates. Correct them from the first measured runs (T049) and record the correction (T060).
- Commit after each task or logical group. Do not commit anything under `experiments/data/` or `experiments/results/`.

## Status notes (2026-09-29)

- **Built and tested:** all tooling (T001 to T057, T059, T060, T062). `python -m pytest tests/experiments -m "not slow"` and `-m slow` pass.
- **Measured on the real data (T020, T033):** the dataset import, splits, length profile and solvability run (run `p2-dev`); families for dev
  (12,600 items), calibration (8,400) and final (21,000, built and fingerprinted, never scored). SC-003 holds on them: every supported item has
  its exact length and evidence span, and no case is in two splits.
- **Waiting on the researcher (T034):** the hand audit. `experiments/data/audit_sheet.md` has the 60 items; the automatic pre-checks passed on all
  12,600 dev items. `eval` and `latency` refuse to run until `audit-record` reports a passing audit of the current families.
- **Not run (T049 to T051, T058, T070):** the quality, latency and GPU runs and the final report. They are gated by T034. Scoring on CPU would take about
  45 hours for native alone and 134 hours for all architectural baselines on dev (`phase2-all --dry-run`, an estimate), so quality is now scored on the GPU
  (about 2.5 hours, an estimate) with a CPU parity subset (Phase 8, research.md R20); stage runs with `--lengths`, `--conditions` and `--max-cases`. A one-case smoke run of the whole pipeline on the real checkpoint (scratch run
  `p2-smoke`, no audit gate, not a result) completed for every condition at 2,048 and 8,192 tokens.
- **T063** (quickstart end to end) is validated through step 2 (families and audit sheet); steps 3 and 4 wait for the runs above.
- Two bugs found by running on real data are fixed and covered by regression tests: `neutral@mid` with under 18 tokens of room (now one block, or
  `unsupported` with a reason; it never aborts a build), and Phase 2 command defaults leaking into Phase 1 commands.

## Phase 8: Change request 2026-09-30 - GPU-scored quality with a CPU parity check

**Why**: scoring the grid on CPU is about 134 hours (45 for native alone); on the GPU about 2.5 hours (estimates). Decision recorded in spec.md
Clarifications (Session 2026-09-30), FR-024 and SC-006; design in research.md R20. T049 to T051 below supersede their CPU-quality wording: quality
runs use `--device gpu`, and T067 adds the CPU parity subset.

- [X] T064 Change spec.md (FR-024, SC-006, Edge Cases, Clarifications), plan.md, research.md (R20), contracts/cli.md, contracts/results.md and quickstart.md for GPU-scored quality
- [X] T065 [P] Give `evalrun.run_items`, `eval_condition`, `run_eval` and `tune_window` a scoring device (`cpu` or `gpu`): records carry `device`, a device mismatch is refused, GPU conditions without CUDA are refused before anything runs, variants are `unsupported` on GPU (`experiments/evalrun.py`)
- [X] T066 [P] Add `summary.parity` (identical-prediction share, accuracy on each device, largest probability difference, pass at 98% and 0.5 points) and put it in `summary.json` (`experiments/summary.py`)
- [X] T067 Make the report device-aware: headline quality from the scoring device, CPU cost joined through each cell's CPU twin, a parity section that states passed, failed or unverified (`experiments/report2.py`); add `eval --device`, `eval --sample variant` and the GPU steps of `phase2-all` (`experiments/cli.py`)
- [X] T068 Tests: device labels and refusals in `tests/experiments/test_evalrun.py`, parity in `tests/experiments/test_calibration_summary.py`, GPU-scored reports in `tests/experiments/test_report2.py`
- [X] T069 Smoke-check on the real checkpoint (scratch run `p2-smoke`, one dev case, 12 conditions at 2,048 and 8,192 tokens): GPU scoring completes in 158 s, and CPU and GPU predictions agree on all 60 compared items (largest probability difference 0.0045)
- [ ] T070 **RUN**: after the audit (T034), score quality on the GPU (`eval --device gpu`: tune, dev, calibration), then the CPU parity subset (`eval --device cpu --sample variant --conditions native,truncCap,retrieve1024`), then `eval-summary` and read the `parity` block. If it fails, GPU-scored quality stays labeled unverified in the report

- **T034 (2026-09-30):** the audit was performed by an AI assistant at the researcher's explicit request, not by a human, and recorded as such
  (`auditor` and `method` in `experiments/data/audit_result.json`; the report says so and recommends a human repeat it). Result: 60 items, 0% answer
  changes, 0% ambiguous, evidence intact; passed for families `1d99f7eeadeff18e`. The remaining items were checked, not read one by one.

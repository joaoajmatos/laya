# Contract: Command-Line Interface (Phase 2 additions)

Entry point: `python -m experiments <command> [options]`, run from the repository root. The Phase 1 commands and common
options (`--run-id`, `--model`, `--revision`, `--threads`, `--seed`, `--mha-fastpath`) keep their meaning; see
[specs/001-cpu-path-audit/contracts/cli.md](../../001-cpu-path-audit/contracts/cli.md). Exit code 0 means the command completed,
including when conditions are recorded `unsupported` or `failed`. A non-zero exit means the tool itself could not run
(bad arguments, fingerprint mismatch, a locked split, a missing prerequisite).

Data commands write to `experiments/data/`. Evaluation commands write to `experiments/results/<run_id>/`. Both are git-ignored.
Every command appends its command line to `commands.txt` of its output directory.

## Data commands

### `data-import`
Download the pinned dataset, verify it, and write the Data Manifest. Options: `--data-revision` (default the pinned sha; `--revision`
keeps its common meaning of a checkpoint revision), `--refresh` (re-download and re-verify), `--data-root`. Refuses when the computed fingerprint differs from a recorded one (FR-002).
Computes the low-confidence cutoff from the training split. Needs network on the first run only.

### `length-profile`
Token-length profile of the imported cases under the target checkpoint's tokenizer (`--model`, default the pinned
fine-tuned checkpoint). Writes `length_profile.json`. Needs `data-import` first.

### `splits`
Create the split manifest: `--split-seed` (default 2002, recorded). Refuses to overwrite an existing manifest unless
the result is identical. Writes the half samples, latency sample and variant sample. `--check` verifies the
existing manifest against the source without writing.

### `families`
Build evaluation items for `--split dev|calibration|final`, `--lengths` (default `512,1024,2048,4096,8192`),
`--variants` (default all four, plus `original` and `oracle`). Each item is verified for exact length and span integrity.
Writes items and `families.json`. `final` is allowed here (items and fingerprint only) and is never scored.
Options: `--seed`, `--rule-version`.

### `audit-sample`
Write the hand-audit sheet: `--n` (default 60, at least 50), `--seed`. Runs the automatic pre-checks first and stops on a failure.

### `audit-record`
Ingest the researcher's verdicts (`--from audit_result_draft.json`) into `audit_result.json` and compute the shares.
The result is bound to the current `families_id`.

## Evaluation commands

### `solvability`
Run the fine-tuned and base checkpoints on original-length dev rows (CPU). Options: `--models` (default both),
`--split dev`. Writes `solvability.json` and `quality/` results for the `original` variant (User Story 1).

### `eval`
Quality run of one or more conditions. Options: `--conditions native,trunc512,truncCap,window,retrieve512,retrieve1024,retrieve2048,oracle`
(default all), `--variants none,fastpath_off,int8_encoder,int8_all_nofast` (default `none`), `--lengths`, `--split dev|calibration`,
`--model`, `--tune` (choose the window size on dev only, write it to `conditions.json` and stop; each retrieval budget runs as its own
condition and `eval-summary` selects the best on dev), `--time-cap` (seconds one measuring process may run before it is relaunched to continue, default 1800; the run resumes
where it stopped), `--max-cases N` (a pilot on the first N cases; results say which cases they cover), `--resume` (always on),
`--device cpu|gpu` (the device that scores quality; default `cpu`; `gpu` needs `.venv-gpu` and is refused with `device_mismatch` when CUDA is missing;
variants are `unsupported` on `gpu`), `--sample all|variant` (`variant` is the fixed 20-case dev sample at 512, 2,048 and 8,192 tokens, `distractor@mid`
only: the CPU parity subset, run as `--device cpu --sample variant --conditions native,truncCap,retrieve1024`). Each condition runs in its own subprocess. Refuses when `audit_result.json` is missing, failed or bound to a
different `families_id`, and refuses `--split final` (FR-013). Failures and unsupported items are written as results.

### `calibrate`
Fit one temperature per (question type, condition) on the calibration-split results and write `calibration.json`;
`eval-summary` applies it to the dev results. Needs calibration-split runs of the same conditions.

### `latency`
Latency sample for the same conditions. Options: `--conditions`, `--variants`, `--lengths`, `--device cpu|gpu` (default `cpu`),
`--repeats auto|N`, `--warmup` (default 1), `--time-cap`, and the Phase 1 canary options. `gpu` runs from `.venv-gpu` and
writes `gpu_`-prefixed files. `int8_*` variants on `gpu` are recorded `unsupported`. Never substitutes one device for the other (FR-024, FR-025).

### `eval-summary`
Assemble per-condition summaries: accuracy, level error, raw and scaled ECE, strata, evidence visibility, and paired
case-clustered differences against named references (native at matching length, `trunc512`, `truncCap`, `oracle`), and the CPU/GPU `parity` block
(identical-prediction share, accuracy on each device, largest probability difference, pass flag; FR-024). Writes `summary.json`.

## Plan and report commands

### `freeze-plan`
Derive the final-test protocol from the dev pilot and write `evaluation_plan.json` and `.md`. Fails if any result
names a final-split item. Refuses to overwrite a frozen plan; a new version needs `--new-version` and a stated reason.

### `phase2-report`
Write `report2.json` and `report2.md`: the data manifest and length profile, solvability and reference, hand-audit result,
quality-cost curves per baseline and length on CPU and on GPU, the native-versus-truncation verdicts at 2K, 4K, 8K, the -2 point baseline
question, the Phase 3 evidence table, the limits, and reproduce commands. Every statement is labeled `[measured]`,
`[estimated]` or `[hypothesized]`. Cites GPU files only in GPU-labeled tables.

### `phase2-all`
Run the data, families, solvability, evaluation, latency, summary and report commands in order on the dev and calibration splits,
stopping at the audit gate (which needs the researcher) and at any failure. `--dry-run` prints the plan and the estimated wall time.

## Exit and refusal reasons

`fingerprint_mismatch`, `split_locked` (a `final` scoring request), `audit_required`, `audit_stale` (families changed after the audit),
`reference_missing` (report needs a solvability result), `device_mismatch` (GPU command without CUDA, CPU command on a GPU device).

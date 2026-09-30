# Quickstart: Validating Phase 2 End to End

Run from the repository root on the measuring machine (Windows, `.venv`). Command details are in [contracts/cli.md](contracts/cli.md),
file shapes in [data-model.md](data-model.md).

## Prerequisites

- Phase 1 environment set up (`experiments/setup_windows.ps1`), and `pip install -e ".[experiments]"` for `pyarrow` and `pytest`.
- Network for the first `data-import` and checkpoint download. Later runs work offline with `HF_HUB_OFFLINE=1`.
- For GPU latency: the `.venv-gpu` environment from `experiments/setup_gpu.ps1`.
- Free disk for the dataset cache and items (well under 1 GB). No credentials are needed.

## 0. Offline checks

```bash
python -m pytest tests/experiments -m "not slow"
```

Expect all tests to pass without network: fingerprint refusal, disjoint splits, exact lengths, evidence spans, train-only distractors,
baseline behavior, metrics, the final-test lock and resume.

## 1. Data intake and length profile (User Story 1)

```bash
python -m experiments data-import
python -m experiments length-profile
```

Expect: `experiments/data/manifest.json` with 1,200 train and 400 test cases, 2,000 test questions, 100 test cases per workflow, a fingerprint and the
low-confidence cutoff. A second `data-import --refresh` matches. Editing the cached parquet makes it refuse with `fingerprint_mismatch`.
`length_profile.json` gives min, median, p90, p99, max and fit shares per workflow (the 2026-09-29 spike saw 124 / 308 / 383 / 597 for
min / median / p90 / max at original length).

```bash
python -m experiments splits
python -m experiments solvability --run-id p2-dev
```

Expect: 120 / 80 / 200 cases (30 / 20 / 50 per workflow), a split manifest that reproduces on a rerun, and `solvability.json` naming the reference
checkpoint with a majority-baseline comparison.

## 2. Families and hand audit (User Story 2)

```bash
python -m experiments families --split dev
python -m experiments families --split calibration
python -m experiments families --split final
python -m experiments audit-sample
```

Expect: every item reaches its named length (`accounting` reconciles), every item records its evidence span, no case appears in two splits, and
distractors come only from training cases. `families --split final` builds and fingerprints items, and scores nothing. Open `audit_sheet.md`,
review at least 50 items, write your verdicts, then:

```bash
python -m experiments audit-record --from audit_result_draft.json
```

Expect `share_answer_changed` of 0%. Otherwise fix the rule, rerun `families` and `audit-sample`. `eval` refuses until the audit passes for
the current `families_id` (`audit_required` or `audit_stale`).

## 3. Baselines, quality and cost (User Story 3)

Quality is scored on the GPU (about 2.5 hours for the whole dev grid, an estimate), from the GPU environment. Tune first, on dev only, then run the
splits:

```bash
.venv-gpu\Scripts\python.exe -m experiments eval --device gpu --split dev --tune --run-id p2-dev
.venv-gpu\Scripts\python.exe -m experiments eval --device gpu --split dev --run-id p2-dev
.venv-gpu\Scripts\python.exe -m experiments eval --device gpu --split calibration --run-id p2-dev
python -m experiments calibrate --run-id p2-dev
```

The CPU parity subset (the same items on CPU, so GPU-scored quality can be called CPU-equivalent), CPU latency and the optimized variants:

```bash
python -m experiments eval --device cpu --sample variant --conditions native,truncCap,retrieve1024 --run-id p2-dev
python -m experiments latency --device cpu --run-id p2-dev
python -m experiments eval --variants fastpath_off,int8_encoder,int8_all_nofast --run-id p2-dev
python -m experiments latency --variants fastpath_off,int8_encoder,int8_all_nofast --run-id p2-dev
```

GPU latency, from the GPU environment:

```bash
.venv-gpu\Scripts\python.exe -m experiments latency --device gpu --run-id p2-dev
```

Expect: a result for every condition and length, either measured or `unsupported`, `failed` or `partial` with a reason (SC-005). Every measured row has
accuracy, calibration error, CPU p50 and p95, peak memory, evidence visibility and, where it ran, GPU latency (SC-006). An interrupted `eval` resumes with the same
command. `eval --split final` exits with `split_locked`.

## 4. Summary, plan and report (User Story 4)

```bash
python -m experiments eval-summary --run-id p2-dev
python -m experiments freeze-plan --run-id p2-dev
python -m experiments phase2-report --run-id p2-dev
```

Expect: `report2.md` with quality-against-CPU-latency and against-GPU-latency curves per baseline and length, the native-versus-truncation verdict
(better, equal, worse or inconclusive, with intervals) at 2K, 4K and 8K, the answer to the -2 point question, the Phase 3 evidence table, the limits and
reproduce commands. `evaluation_plan.json` states the required and available final-test cases. `final_scored_items` is 0 (SC-008).

## Whole phase in one go

```bash
python -m experiments phase2-all --dry-run
```

Shows the ordered plan and the estimated wall time. It stops at the audit gate, which needs you.

## Success checks

| Criterion | How to check |
|---|---|
| SC-001 | Delete `experiments/data/` and rerun steps 1 and 2. Fingerprints match the recorded ones |
| SC-002 | The length profile table in `report2.md` |
| SC-003 | `families.json` fingerprints, and the disjointness test |
| SC-004 | `audit_result.json`: at least 50 items audited, 0% answer changes |
| SC-005, SC-006 | `summary.json`: no missing (condition, length) cell, and measured rows complete |
| SC-007 | The verdict table in `report2.md` |
| SC-008 | `final_scored_items: 0` in `evaluation_plan.json` and the report |
| SC-009 | The Phase 3 evidence table, each row linked to a result |

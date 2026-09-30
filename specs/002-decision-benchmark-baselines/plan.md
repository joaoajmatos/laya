# Implementation Plan: Decision Benchmark and Practical Baselines (Research Phase 2)

**Branch**: `002-decision-benchmark-baselines` (spec directory only; work stays on `main` unless the user branches) | **Date**: 2026-09-29 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-decision-benchmark-baselines/spec.md`

## Summary

Extend the Phase 1 `experiments/` package so it can (1) import upstream's `typed-decisions` data at a pinned revision, (2) build controlled long-context evaluation items from its test cases, (3) run native Laya and the simple baselines on those items, and (4) report quality, calibration, latency and memory on CPU, with GPU latency beside it.

`laya/` stays untouched, as in Phase 1. Every baseline is a *state transformation* in front of the unchanged `Agent`: truncation and the per-call `max_len` override, the existing `predict_long` windowed path, a small pure-Python BM25 retriever, and an oracle that keeps only the target record. Two optimized-system variants sit outside the library too: PyTorch's `mha` fast-path switch (already wired in Phase 1) and dynamic int8 quantization applied to the loaded model.

The approach separates two kinds of run, because quality over thousands of rows and clean latency need different things:

- **Quality runs** score every planned item once, in a resumable per-condition subprocess, and record predictions, probabilities and evidence visibility. No repeats.
- **Latency runs** reuse the Phase 1 clean-path timer on a small fixed sample of items per condition, with warmup and repeats, one document and one question per call.

Quality is scored on the GPU (from `.venv-gpu`), because the full grid takes days on CPU (change of 2026-09-30, spec FR-024, research.md R20). Every quality record carries its scoring device. CPU stays the device for latency, memory and a fixed parity subset (the 20-case variant sample at 512, 2,048 and 8,192 tokens) on which CPU and GPU predictions are compared; GPU-scored quality is called CPU-equivalent only if at least 98% of predictions agree and accuracy differs by at most 0.5 points. GPU latency is a separate, labeled table.

## Technical Context

**Language/Version**: Python 3.10+ (constitution); the `.venv` on the measuring machine runs 3.11.

**Primary Dependencies**: existing `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`. One addition, `pyarrow`, to read the upstream parquet files (optional `experiments` extra, next to `pytest`). No `pandas` or `datasets`. Retrieval, bootstrap and calibration are small pure-Python or NumPy code. Quantization uses `torch.ao.quantization.quantize_dynamic`, already in `torch`.

**Storage**: Git-ignored files under `experiments/results/<run_id>/` for run outputs, and `experiments/data/` for the imported dataset cache and derived items (also ignored). Raw upstream records are never committed (FR-003, FR-031); only code, specs and reviewed reports are. Manifests and fingerprints can be regenerated from the pinned source and seeds.

**Testing**: `pytest`. Unit tests run offline on the tiny fixture checkpoint from Phase 1 plus a small synthetic dataset written in the test's temp directory, in the upstream schema. Real-checkpoint and network tests are marked `slow`.

**Target Platform**: Windows PC, i7-8700, 16 GB RAM, RTX 4060 (Assumptions). CPU work in `.venv`; GPU latency in `.venv-gpu`. Both recorded in the manifest. Nothing is machine-specific in the code.

**Project Type**: Research tooling. The `python -m experiments` CLI gains Phase 2 commands.

**Performance Goals**: None for the tooling. The latency path must stay clean: no hooks or profilers (Phase 1 rule, FR-023).

**Constraints**: 16 GB RAM with sequences up to 8,192 tokens. One subprocess per condition, so peak memory and failures are per condition. Model weights, caches and generated items are ignored files. Quality runs must resume after interruption.

**Scale/Scope**: Upstream test split: 400 cases × 5 questions. Splits: 120 dev / 80 calibration / 200 final (final is built and fingerprinted, never scored). Five lengths, four context variants, eight baseline conditions. Full-grid quality at 512 to 2,048 tokens and a fixed half-sample at 4,096 and 8,192 tokens (research.md R11). The full grid is days of CPU wall time (`phase2-all --dry-run` estimates about 45 hours for native alone and 134 hours for all architectural baselines on dev, an estimate to be corrected from the first measured runs), so runs are staged and can be scaled with `--lengths`, `--conditions` and `--max-cases`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| I. Preserve the Inference Contract | No `laya/` edit. Baselines change the *input* (truncate, select, window) or the runtime setting, never the answer semantics. Quantization and the fast-path switch are applied to the loaded model outside the library and labeled as variants. | Pass |
| II. Measure Research Claims | Every report statement is labeled measured, estimated or hypothesized. Compute-time estimates in this plan are labeled estimated. | Pass |
| III. Valid Native Baseline | The reference is the pinned, unmodified checkpoint. The quality reference is chosen by a stated solvability rule (R14), and native is compared at matching length and against truncation. Architectural baselines and optimized-system variants are reported apart. | Pass |
| IV. Keep the Runtime Focused | All new code is in `experiments/`. `laya/` unchanged. | Pass |
| V. Reproducible Experiments | Manifest adds dataset revision, fingerprints, split seeds, family seeds, checkpoint revisions, device, threads, precision and variant flags. Results and derived items are ignored files. Every report has reproduce commands. | Pass |
| VI. Fresh-Document CPU Decisions | Primary latency is one fresh document, one question, tokenization through postprocessing, on CPU. Retrieval and windowing are inside the timed call. GPU is reported apart and never substitutes. | Pass |
| VII. Evaluation Independence | Splits are by case, all variants of a case stay together, and the final split is fingerprinted and not scored. Tuning uses dev only and temperatures use calibration only. Paired case-clustered uncertainty is used. Two limits are stated in the report: the fine-tuned checkpoint trained on upstream's training split (including the distractor pool), and upstream's test split is already public. Held-out template families are not available, so no template-generalization claim is made. | Pass, with two stated limits |

Technical Constraints check: total context counts task, option and special tokens; marker indices and positions are validated after every transformation (R3, R8). Calibration error uses the predicted answer's probability, not entropy (R10). Selection and compression code is in `experiments/`, explicit and reversible.

**Deviation to disclose**: constitution VII asks length studies to combine controlled examples with *realistic tasks*. Upstream data is synthetic and short, so this phase has controlled families only. The spec lists a realistic long-document set as a gap, and the report must repeat that.

**Result**: no violations. Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```text
specs/002-decision-benchmark-baselines/
├── plan.md              # This file
├── research.md          # Phase 0: decisions and their rationale
├── data-model.md        # Phase 1: entities and record shapes
├── quickstart.md        # Phase 1: end-to-end validation guide
├── contracts/
│   ├── cli.md           # Command-line contract (new commands)
│   └── results.md       # Result and data file contract
├── checklists/
│   └── requirements.md
└── tasks.md             # Produced later by /speckit-tasks
```

### Source Code (repository root)

```text
experiments/
├── (Phase 1 modules unchanged: results, manifest, runner, timing, tokens, inputs, audit, ...)
├── data.py              # Upstream import, data manifest, fingerprint, splits, low-confidence cutoff
├── lengths.py           # Case token accounting and length profile (per workflow distributions, fit shares)
├── families.py          # Evaluation item builder: neutral, near-match distractor, position; exact length; evidence span
├── audit_items.py       # Stratified audit sample, sheet generation, audit result ingestion
├── bm25.py              # Small pure-Python BM25, shared by distractor selection and retrieval
├── baselines.py         # Native, truncate, window, retrieve (BM25), oracle as state/setting transforms
├── variants.py          # Fast-path switch and dynamic int8 quantization, with output-equivalence check
├── evalrun.py           # Quality run: per-condition child process, resumable, writes predictions; tuning; solvability
├── summary.py           # Calibration on the calibration split; per-condition summary and paired comparisons
├── metrics.py           # Accuracy, ordinal MAE, ECE, temperature fit, paired case-clustered bootstrap
├── latency.py           # Latency sample per condition on top of timing.py; CPU and GPU
├── evalplan.py          # Pilot paired differences, power calculation, frozen evaluation plan
├── report2.py           # Phase 2 report: curves, verdicts, limits, reproduce commands
├── data/                # Git-ignored: dataset cache, derived items, split manifest
└── results/             # Git-ignored: run outputs

tests/
└── experiments/
    ├── conftest.py      # + tiny synthetic upstream-schema dataset fixture
    ├── test_data.py
    ├── test_lengths.py
    ├── test_bm25.py
    ├── test_audit_items.py
    ├── test_calibration_summary.py
    ├── test_families.py
    ├── test_baselines.py
    ├── test_variants.py
    ├── test_evalrun.py
    ├── test_metrics.py
    ├── test_latency.py
    ├── test_evalplan.py
    └── test_report2.py
```

**Structure Decision**: extend the flat `experiments/` package from Phase 1, as principle IV requires, and reuse its runner, timing, manifest and results modules rather than duplicating them. New modules are small and each maps to one part of the spec. `cli.py` gains the commands in [contracts/cli.md](contracts/cli.md). `pyproject.toml` gets `pyarrow` in the `experiments` extra, and `.gitignore` gets `experiments/data/`. Neither touches `laya/`.

## Complexity Tracking

No constitution violations. Nothing to justify.

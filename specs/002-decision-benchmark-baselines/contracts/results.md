# Contract: Result and Data Files (Phase 2 additions)

Phase 1's rules carry over ([results contract](../../001-cpu-path-audit/contracts/results.md)): `schema_version` and `run_id` at the
top level, unsupported and failed conditions present with a `reason` (never omitted), milliseconds and bytes, analytical and measured
numbers under different names, derived items carry `derived_from`, reports label statements, GPU files carry a `gpu_` prefix.

## Data directory: `experiments/data/` (git-ignored, regenerable)

| File | Producer | Content (see [data-model.md](../data-model.md)) |
|---|---|---|
| `manifest.json` | `data-import` | Data Manifest |
| `cache/` | `data-import` | Local copy of the pinned parquet files. Third-party raw records, never committed |
| `length_profile.json` | `length-profile` | Length Profile |
| `splits.json` | `splits` | Split Manifest |
| `items/<families_id>/<split>.jsonl` | `families` | Evaluation Items |
| `items/<families_id>/families.json` | `families` | Rule version, seeds, fingerprints of each split file |
| `audit_sheet.json`, `audit_sheet.md` | `audit-sample` | Hand-audit sheet |
| `audit_result.json` | `audit-record` | Hand-audit result, bound to a `families_id` |

## Run directory: `experiments/results/<run_id>/`

| File | Producer | Content |
|---|---|---|
| `manifest.json` | first command | Phase 1 Run Manifest plus `data` (dataset revision, fingerprints, split fingerprints, `families_id`), `checkpoints`, `variant` |
| `solvability.json` | `solvability` | Solvability Check |
| `conditions.json` | `eval --tune` | Tuned parameters per condition, with `tuned_on: dev` |
| `quality/<condition_id>/predictions.jsonl` | `eval` | Per-item Quality Result lines, appended as they finish |
| `quality/<condition_id>/summary.json` | `eval-summary` | Aggregates and paired differences |
| `calibration.json` | `calibrate` | Temperatures and the calibration-split counts used |
| `latency/<condition_id>.json` | `latency` | Cost Result, CPU |
| `gpu_latency/<condition_id>.json` | `latency --device gpu` | Cost Result, GPU only |
| `summary.json` | `eval-summary` | All conditions in one table, with statuses |
| `evaluation_plan.json`, `.md` | `freeze-plan` | Frozen Evaluation Plan |
| `report2.json`, `report2.md` | `phase2-report` | Phase 2 report |
| `commands.txt` | every command | Exact command lines |

`condition_id` is `<name>[.<params>].<variant>.<device>.L<length>` and is stable across reruns.

## Rules

1. **Predictions are append-only.** A line is written when an item finishes. A restart never rewrites finished lines. A rerun with changed
   conditions uses a new `condition_id`.
2. **No cross-substitution.** A failed, unsupported or partial item is never replaced by another length, device, checkpoint or truncated run
   (FR-025). Aggregates report the count of each status and compute accuracy over `measured` items only, with the denominator shown.
3. **Final split is unscored.** No file under `results/` may contain an item with `split: final`. `freeze-plan` and `phase2-report` verify this
   and report `final_scored_items: 0` (SC-008).
4. **Quality carries its scoring device.** Each `predictions.jsonl` record has `device` (`cpu` or `gpu`), and the device is part of the `condition_id`
   (`native.none.gpu.L2048` and `native.none.cpu.L2048` are different conditions, never merged). The headline quality is GPU-scored; the CPU
   twins exist for the fixed parity subset. `summary.json` carries a `parity` block (identical-prediction share, accuracy on each device, largest
   probability difference, and a pass flag for the 98% and 0.5-point criteria), and the report calls GPU-scored quality CPU-equivalent only when it passes.
   Latency and memory files (`latency/`, `gpu_latency/`) always come from the device they name.
5. **Windowed probabilities** carry `probability_scope: "deciding_window"` and are excluded from native-comparable calibration figures.
6. **Variants are labeled.** A result with a non-`none` variant carries `variant` in its condition and is listed in the optimized-system table,
   never in the architectural baseline ranking.
7. **Report labels.** `report2.md` labels each statement `[measured]`, `[estimated]` or `[hypothesized]`. Every curve point and every claim links to its
   result file. Estimated wall times in the plan are labeled estimated.
8. **Committed artifacts.** Only code, specs and the reviewed report may be committed. Data, items, predictions, model weights and caches stay ignored
   (FR-003, FR-031).

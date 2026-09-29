# Contract: Result Files

Directory: `experiments/results/<run_id>/` (git-ignored, constitution V).

| File | Producer | Content (see [data-model.md](../data-model.md)) |
|---|---|---|
| `manifest.json` | `manifest` | Run Manifest |
| `audit.json` | `audit` | Context Audit |
| `sweep.json` | `sweep` | array of Latency Measurement |
| `profile.json` | `profile` | array of Component Profile |
| `kernels.json` | `kernels` | array of Microbenchmark Result |
| `floor.json` | `kernels` | Cost Floor per length |
| `report.json` | `report` | Bottleneck Ranking |
| `report.md` | `report` | Human-readable report of the same content |
| `commands.txt` | every command | The exact command line, appended |
| `abtest.json` | `abtest` | Instrumentation A/B timings (research.md R16) |
| `gpu_reference.json` | `gpu-reference` | GPU-only timings of native Laya (research.md R15); never a CPU result |

## Rules

1. Every file has `schema_version` (integer, starts at 1) and `run_id` at the top level. A change to a field's meaning increments `schema_version` and is noted in the report (constitution III: methodology changes are versioned with results).
2. A condition that could not produce a measurement is present with `status` of `unsupported`, `failed` or `partial`, and a `reason`. Conditions are never omitted.
3. Times are milliseconds unless the field name says otherwise; memory is bytes.
4. Analytical numbers and measured numbers use different field names (for example `score_matrix_bytes_analytical` versus `peak_rss_bytes`).
5. Every result item that is derived carries `derived_from`, a list of `file#index` references.
6. `report.md` labels each statement `[measured]`, `[estimated]` or `[hypothesized]`.
7. GPU results are written to files with a `gpu_` prefix. `report` may cite `gpu_reference.json` only in the note on the hardware behind the historical ~33 ms figure, labeled as GPU, and never in rankings, curves, the cost floor or memory feasibility (FR-016).

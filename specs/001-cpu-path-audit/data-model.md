# Data Model: CPU Path Audit and Measurement

All records are JSON files written under `experiments/results/<run_id>/`. Field lists give
meaning and validation rules; the file-level contract is in [contracts/results.md](contracts/results.md).
`status` is always one of `measured`, `unsupported`, `failed`, `partial`.

## Run Manifest (`manifest.json`)

One per session. Attached (by `run_id`) to every other record.

| Field | Meaning / rule |
|---|---|
| `run_id` | Unique session id; also the results directory name |
| `created_at` | UTC timestamp |
| `code` | `git_sha`, `git_dirty` (bool), `laya_diff_empty` (bool: `laya/` unchanged from the recorded sha) |
| `model` | `id`, `revision` (resolved commit; required, never `null` for a Hub load), `config_sha256` |
| `software` | Python, PyTorch, Transformers, NumPy, safetensors versions |
| `hardware` | CPU model, architecture, physical/logical cores, RAM bytes, OS; on Windows also the active power scheme and whether on AC power |
| `runtime` | Torch intra-op and inter-op threads, thread-count source (`default` / `user`), `hybrid_cores` (`true` / `false` / `unknown`), BLAS/MKL/OpenMP info, `LAYA_CPU_AMP` value, precision (`dtype`, `amp_enabled`), `compile` flag, concurrent-process note |
| `device` | Requested device (must be `cpu`), effective `agent.device`, fallbacks observed |
| `cache_policy` | Question-token reuse on/off; document caches: none |
| `seeds` | Base seed and per-repeat seed rule |

Rule: `device.requested` other than `cpu` marks the whole run `non_comparable`.

## Context Audit (`audit.json`)

Read from the loaded model, not documentation.

| Field | Meaning / rule |
|---|---|
| `encoder` | class, hidden size, heads, head dim, layer count, attention implementation |
| `layers[]` | per encoder layer: index, `attention_type` (`local` / `global`), window (local only), rope theta |
| `positional` | `max_position_embeddings`, `configured_max_len` (agent config), `effective_supported_max` |
| `decision_head` | layer count, heads, hidden size; note that it runs over the full sequence |
| `executed_work_note` | Whether local layers were observed to skip out-of-window work (`verified`, `dense_masked`, `not_yet_determined`) |
| `fallbacks[]` | Any device/precision fallback events, with reason |

Rule: `effective_supported_max` equals `max_position_embeddings`; lengths above it are `unsupported`.

## Token Accounting Record (embedded in results)

| Field | Rule |
|---|---|
| `requested_total` | Length asked for |
| `state_tokens_original` | Full-state tokenization length |
| `task_option_tokens` | Question head + all option spans |
| `special_tokens` | CLS/SEP/MASK-marker tokens |
| `state_tokens_retained` | State tokens that reached the model |
| `padding_tokens` | Pad positions (0 for single requests and for batches built by the tooling, whose rows share one length) |
| `n_questions` | Question rows in the request |
| `options_per_question` | Options defined for each question (list, one entry per question) |
| `final_length` | Tensor length seen by the model |
| `position_limit` | From the audit |
| `truncated` | `state_tokens_retained < state_tokens_original` |

Validation: `special + task_option + state_retained + padding == final_length` (exact);
`final_length <= position_limit`; `requested_total == final_length` for every question row of every request, else the record is invalid. For requests with several rows the record holds one entry per row, and all rows must report the same `final_length`.

## Latency Measurement (`sweep.json` items)

One per condition = (`total_tokens`, `questions`, `options_per_question`, `batch_size`). `total_tokens` is the final length of each question row.

| Field | Rule |
|---|---|
| `condition` | The tuple above, including the option count |
| `kind` | `single` / `multi_question` / `batch` (never merged in summaries). `single` and `multi_question` time the public `predict`; `batch` times `predict_batch` |
| `beyond_configured_max_len` | True when `total_tokens` exceeds the agent's configured input cap (`max_len` in the checkpoint config) but not positional capacity. Cost-only measurement; carries no quality claim |
| `status` | See top of file; `failed` carries `cause` (`oom`, `exception`, `timeout`, `signal`, `crash`), `signal` (POSIX only, else null), `exit_code` |
| `load_seconds`, `compile_seconds` | Model load and one-time compilation, reported apart from the timings |
| `repeats` | Requested and completed counts |
| `timings_ms` | min, p50, p95, mean, std; `low_sample_p95` flag if repeats < 20 |
| `first_vs_median` | Warmup-leak check |
| `peak_rss_bytes` | From the condition's own process (peak working set on Windows) |
| `peak_commit_bytes` | Windows only (null elsewhere): peak private commit of the condition's process |
| `paging_suspected` | True when `peak_commit_bytes` exceeds physical RAM; timings are then cost-only |
| `token_accounting` | Record above |
| `fallbacks` | Events seen during the condition |

Rule: `multi_question` and `batch` entries never populate the primary single-request curve.

## Component Profile (`profile.json` items)

One per length, single request, from profile runs only.

| Field | Rule |
|---|---|
| `total_profiled_ms`, `total_clean_ms` | Profiled and matching clean p50; ratio recorded |
| `components` | ms and share for: `tokenization_collation`, `attention_projections`, `attention_score_value`, `mlp_norm`, `decision_head`, `option_scoring`, `decoding`, `other` |
| `explained_fraction` | Sum of named components (excluding `other`) over profiled total |
| `scaling` | Fitted time exponent versus length per component, across the sweep |

Rule: shares sum to 1 within rounding; `explained_fraction >= 0.90` or the shortfall is stated.

## Microbenchmark Result (`kernels.json` items)

One per (implementation, shape).

| Field | Rule |
|---|---|
| `impl` | `dense` / `local` / `gas` (and `reference` for correctness only) |
| `shape` | batch, heads, length, head dim, block/window size, selection size |
| `timings_ms` | As in latency measurement; includes mask, routing, padding, copies |
| `peak_rss_bytes`, `score_matrix_bytes_analytical` | Measured and analytical, kept as separate fields |
| `correctness[]` | Case name, max abs error, tolerance, pass/fail (`padding`, `boundary_chunk`, `non_divisible_length`, `fully_masked_row`, `mixed_batch`) |
| `executed_work` | Scaling exponent and verdict: `less_work`, `same_work`, `not_determined` |

## Cost Floor (`floor.json`)

Per length: `floor_ms` (total minus attention score/value), `floor_fraction`, and the analytical
`8 x (1 - f)` bound at the relevant lengths. Marked `estimate`.

## Bottleneck Ranking (`report.json` / `report.md`)

| Field | Rule |
|---|---|
| `audit_summary` | Short statement of the layer schedule, positional capacity, input cap, head depth and observed fallbacks, drawn from `audit.json` |
| `memory_feasibility[]` | Per length: analytical score-matrix bytes (`analytical`), measured peak RSS (`measured`) for the native run and each kernel, whether the full score matrix fits in available RAM, and which measured path avoids materializing it |
| `cost_floor[]` | Per length: floor ms and fraction from `floor.json`, tagged `estimate`, shown next to the ranking |
| `by_length[]` | Ordered components with share of clean end-to-end p50 and links to the supporting sweep/profile items |
| `assumptions[]` | For each plan assumption (attention dominates near 2K; 512-token latency near 33 ms; local layers skip work): `confirmed` / `contradicted` / `untested`, with evidence links |
| `statement_status` | Every claim tagged `measured`, `estimated`, or `hypothesized` |
| `limitations[]` | Stated measurement limits, always including the fixed-thread-count limit from research.md R12 with the recorded thread count and `hybrid_cores` value |
| `high_variance[]` | Conditions with `p95 / p50 > 1.5`, by reference |
| `not_run[]` | Unsupported lengths, failed runs, checks that could not run, with reasons |
| `reproduce[]` | The exact commands for each result |
| `phase3_implications` | Which prototype directions the data supports, weakens, or leaves open |

Relationships: every record carries `run_id`; sweep, profile and kernel items reference the audit
by `run_id`; the ranking cites items by file and index.

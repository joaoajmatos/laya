# Data Model: Decision Benchmark and Practical Baselines

Record shapes for the entities in the spec. All JSON files carry `schema_version` (integer, starts at 1) and `run_id` or
`data_id`, as in Phase 1. Times are milliseconds and memory is bytes. Field names for measured values and analytical
values differ (Phase 1 rule 4).

## Data Manifest (`experiments/data/manifest.json`)

| Field | Meaning |
|---|---|
| `source` | `LocalLLaMA/typed-decisions` |
| `revision` | Pinned dataset sha (`d51d99...`) |
| `license` | `apache-2.0` |
| `configs` | `all` and the four workflow configs cross-checked |
| `splits` | per upstream split: case count, question count, questions by type |
| `workflows` | cases per workflow per split |
| `fingerprint` | SHA-256 over canonical rows (research.md R1), per split and combined |
| `low_confidence` | `{cutoff, basis: "train", quantile: 0.25, n_questions}` (R13) |
| `checkpoints` | model id and pinned revision for `laya-typed-decisions` and base `laya`, with `max_len`, `head_max_len` |
| `imported_at`, `code` | time and the code revision |

## Upstream Case (in memory; cached under `experiments/data/`, never committed)

`id`, `workflow`, `split` (upstream), `state` (dict), `questions` (id → definition with `type`, `instructions`,
`criteria`), `gold` (id → `label`, `confidence`, `probabilities`), `label_agreement`, `factors` (diagnostic only).
Derived per question: `type` (`choice`, `noul`, `score`), `n_options`, `low_confidence` (bool).

## Length Profile (`length_profile.json`)

Per case and question: `row_tokens` (state, question head, options and special tokens under Laya's tokenizer), split
into `state_tokens`, `task_option_tokens`, `special_tokens`. Aggregated per workflow, per split and overall: `min`,
`median`, `p90`, `p99`, `max`, and `fit_share` for each of 512, 1,024, 2,048, 4,096, 8,192. Also `over_head_budget`
(questions whose options exceed `head_max_len`, reported separately).

## Solvability Check (`solvability.json`)

Per checkpoint (fine-tuned, base) on original-length dev rows: `accuracy` per question type and workflow,
`mean_abs_level_error`, `ece_raw`, `low_confidence` and `agreement` strata, `majority_baseline` (from train),
`paired_vs_majority` (case-clustered interval), and `solves` (boolean, by the rule of research.md R14). The quality
reference is recorded as `reference_checkpoint`, or `none` when no checkpoint solves (FR-006).

## Split Manifest (`experiments/data/splits.json`)

`seed`, `rule` (`stratified_by_workflow_30_20_50`), and for each of `dev`, `calibration`, `final`: `case_ids`,
`n_cases`, `n_per_workflow`, `fingerprint`. Also `half_sample` per split (case ids for the 4,096 and 8,192 tier,
research.md R11), `latency_sample` (12 dev items), and `variant_sample` (20 dev cases).

## Evaluation Item (`experiments/data/items/<families_id>/<split>.jsonl`, one row per item)

| Field | Meaning |
|---|---|
| `item_id` | Stable id: case, question, family variant, length, seed |
| `case_id`, `question_id`, `split`, `workflow`, `question_type` | Identity and split |
| `variant` | `neutral@mid`, `distractor@mid`, `distractor@begin`, `distractor@end`, `original` or `oracle` |
| `length` | Named total tokens, or `original` |
| `state_text` | The built state string (derived; regenerable) |
| `question` | The question definition exactly as the model sees it (`choice` options permuted per `option_order`) |
| `layout` | Ordered segments: `{kind: target|reference|neutral|adjust, ref_case_id?, tokens}` |
| `evidence_span` | `{state_start, state_end, row_start, row_end}` token positions of the target record |
| `gold` | Label (remapped if options were permuted), probabilities, `confidence`, `low_confidence` |
| `option_order` | Permutation applied to `choice` options, or null |
| `construction` | `{seed, rule_version, marker_constants, distractor_case_ids, neutral_vocab_version}` |
| `status` | `ok`, or `unsupported` with `reason` (for example `target_exceeds_length`) |
| `accounting` | Token Accounting Record from `tokens.account` (must reconcile) |

The spec's three families map to `variant` values: neutral padding = `neutral@mid`; near-matching distractors = `distractor@mid`; position = `distractor@begin`, `distractor@mid`, `distractor@end`. `original` and `oracle` are controls, not families.

`families.json` beside the items holds the `families_id` (fingerprint of the rule version and seeds) and the fingerprint of every split file.

## Baseline Condition (`conditions.json`, and the `condition` block of each result)

`name` (`native`, `trunc512`, `truncCap`, `window`, `retrieve512|1024|2048`, `oracle`), `params` (window size and stride,
budget, chunk size), `variant` (`none`, `fastpath_off`, `int8_encoder`, `int8_all_nofast`), `checkpoint` (a `ckpt` parameter in the condition id when it is not the fine-tuned one), `device`,
`length`, `max_len_used`, `beyond_configured_max_len`, `tuned_on: "dev"` when a parameter was tuned.

## Quality Result (`quality/<condition_id>/predictions.jsonl` and `summary.json`)

Per item (one line, appended as it finishes):

| Field | Meaning |
|---|---|
| `item_id`, `condition_id` | Keys |
| `predicted`, `probabilities`, `prob_predicted`, `correct` | Exact match with gold, and the predicted answer's probability |
| `level_error` | Absolute level error for ordinal questions |
| `tokens_seen` | Tokens that reached the model |
| `evidence_visible` | `full`, `partial` or `none`, with `evidence_fraction` |
| `window` | For `window`: deciding window index and range, best coverage of the evidence span |
| `status`, `reason` | `measured`, `unsupported` or `failed` |

`summary.json` per condition: accuracy per question type, workflow and stratum; `mean_abs_level_error`; `ece_raw`;
`temperature` (by question type, fitted on the calibration split) and `ece_scaled`; item counts by status and by
`evidence_visible`; `paired` (differences against named reference conditions with case-clustered 95% intervals and a
verdict of `better`, `equal`, `worse` or `inconclusive`).

## Cost Result (`latency/<condition_id>.json`)

Same shape as a Phase 1 latency item, with the sample kind and device added: `device` (`cpu` or `gpu`), `sample`
(the fixed 12 item ids), `repeats`, `warmup`, `p50_ms`, `p95_ms`, `samples_ms`, `peak_rss_bytes`, `peak_commit_bytes`
(CPU), `peak_gpu_bytes` (GPU), `status` (`measured`, `unsupported`, `failed`, `partial`), `reason`, `drift` (canary record),
`fallbacks`. GPU results are written to `gpu_`-prefixed files.

## Audit Sample and Result (`audit_sheet.json`, `audit_result.json`)

Sheet: `item_ids` (at least 60, stratified), `families_id`, `seed`, and per item the target record, gold, layout and
rubric. Result: per item `changes_answer` (bool), `ambiguous` (bool), `evidence_intact` (bool), `note`, plus
`share_answer_changed`, `share_ambiguous`, `families_id` audited, and `passed` (true only at 0% changed and 0% ambiguous on
the current `families_id`).

## Evaluation Plan (`evaluation_plan.json`, and `evaluation_plan.md`)

`frozen_at`, `fingerprint`, `metrics`, `margin_pp` (2), `comparisons` (pairs and the verdict rule), `pilot` (paired
differences and intra-case correlation), `required_cases` and `available_cases` (200), `power`, `resolvable` (bool), and a
plain statement of what the final split can and cannot resolve. `final_scored_items` must be 0 (SC-008).

## Relationships and lifecycle

Data Manifest → Upstream Cases → Length Profile and Solvability Check. Splits partition the test cases. Evaluation Items
derive from a case plus training-split reference records, and carry their split. A Baseline Condition applied to Evaluation
Items yields Quality Results, and a Cost Result comes from the fixed latency sample. States: an item is `ok` or
`unsupported`; a result is `measured`, `unsupported`, `failed` or `partial`. Families must be audited (`passed`) before
baseline comparison runs; the evaluation plan is frozen before Phase 4 scoring and never edited afterwards (a change creates a
new plan version).

# Research: CPU Path Audit and Measurement

Decisions for the open technical questions in the plan. No `NEEDS CLARIFICATION` markers remain.
Items marked *to verify* are hypotheses the tooling itself must confirm or contradict, and the
final report must label them as such.

## R1. Can `laya/` stay untouched?

**Decision**: Yes. Use zero edits to `laya/` in the initial implementation.

**Rationale**: Every needed measurement point exists outside the library.

- Length control: `predict_batch(..., max_len=N)` overrides the configured cap per call (`laya/agent.py:934`). `max_len` is a ceiling, not a target, so the input builder (R4) supplies text that fills it.
- Fallback detection: after loading and after each run, read `agent.device`, `agent.dtype`, `agent.amp_enabled`, `agent.cpu_fallback_count`, `agent.last_fallback_reason`, `agent._compiled`. The `CPU AMP` mode is chosen by the `LAYA_CPU_AMP` environment variable, which the manifest records.
- Stage timing: wrap `Agent._encode_state`, `Agent._forward`, `Agent._decode_answers` on the instance, and `collate_items` in the `laya.agent` namespace, with timers that call through to the originals. The real `predict_batch` still runs end to end, so there is no re-implemented copy of the logic to drift from it.
- Operator costs: `torch.profiler` and module forward hooks (R5).

**Alternatives considered**:
- Edit `laya/agent.py` to add timing points. Rejected as unnecessary. It adds diff surface and a burden to prove the native path is unchanged (FR-019).
- Re-implement the request path in the harness to time stages. Rejected: it could diverge from the real path and measure something else.

**Contingency**: if a later finding needs a probe that wrapping cannot reach, add it to `laya/` off by default, document its overhead, and test output equivalence (FR-017 to FR-019). Success criterion SC-005 is then checked by `git diff` on `laya/` being empty, or by the equivalence test.

## R2. How to run and isolate each measurement condition

**Decision**: Run each condition (length x question count x batch size) in a fresh subprocess started by `runner.py`. The parent collects a JSON result from the child's stdout or a temp file, and records exit status.

**Rationale**:
- Peak RSS (`resource.getrusage`) never decreases within a process, so per-condition peaks need per-condition processes.
- An out-of-memory condition on macOS or Linux usually ends the process with a signal instead of a catchable exception. A parent process is the only place this can be recorded (FR-015).
- It isolates allocator state, thread pools and cached kernels between conditions.

**Consequence**: model load happens once per subprocess, so model-load time is measured naturally and separately for each (FR-010). Steady-state timings start after warmup inside the child.

**Alternatives considered**: one long-lived process with a memory tracker. Rejected because it cannot give per-condition peaks or survive a kill.

## R3. What "native attention" actually executes

**Decision**: The audit reads the loaded encoder's configuration and modules instead of assuming. It reports layer types, the local window, global attention cadence, RoPE settings per layer type, positional capacity, attention implementation, decision-head layer count, and hidden and head sizes.

**Rationale**: the plan says the English encoder uses local attention of 128 with global layers every third layer, `max_position_embeddings=8192`, and `head_layers=2`. Those figures are background and must be confirmed from the loaded model (constitution III). `laya/common.py` builds the encoder with `attn_implementation="sdpa"` and `_apply_rope_config` normalizes RoPE across transformers versions, so the audit also records the transformers version that produced the settings.

**To verify**: in the Hugging Face ModernBERT SDPA path, the local layers may pass a full-sequence sliding-window mask to `scaled_dot_product_attention` instead of skipping out-of-window work. If so, native "local" layers still do dense-shaped computation at run time. The profiler's scaling behavior across lengths (R5) decides this, not the source code alone. It matters because it means the native model may not be cheaper than an all-global one by as much as its configuration suggests, and it changes what Phase 3 must beat.

**Decision-head observation** (from `laya/common.py:313`): the head's transformer layers run over the full sequence before marker positions are gathered, so head cost scales with sequence length. The component profile therefore reports head layers separately (FR-012), and the cost-floor estimate keeps them (FR-025).

## R4. Synthetic inputs with an exact total token count

**Decision**: `inputs.py` builds a request (one state string, one question, options) whose final model input has exactly the requested total length. Method: fix the question and options, compute the fixed token overhead with `build_sequence`, then grow a deterministic filler state until the tokenized state fills the remaining budget. Size it by tokenizing (binary search on filler units), then set `max_len` to the target. The token accounting record (R7) verifies the result, and the run fails if it misses the target.

**Rationale**: the total includes question, options, and special tokens (FR-005), and `max_len` alone does not force length. Filler comes from a seeded generator over ordinary English words, so it tokenizes like real text and not like a degenerate repeated token. A different seed gives a different document per repeat, so no state is reused across timed requests (the fresh-document condition).

**Several questions or documents**: `build_request(agent, total_tokens, seed, n_questions, options_per_question, n_states)` gives every question the same head length (question text and options are built from templates of equal token length) and gives all states the same filler budget. Every question row therefore reaches exactly `total_tokens`, batches have no padding by design, and a condition names one length. The accounting record holds one entry per row, and the build fails if any row misses the target.

**Alternatives considered**: random token IDs. Rejected because they bypass tokenization, which the timed path includes. Repeating one sentence. Rejected because tokenization cost and tokenizer merge behavior differ from natural text.

**Disclosure**: the cache policy for question tokens is recorded in the manifest. Laya's `_reuse_question_tokens` decorator caches fixed-schema question tokens during a call. The plan allows this if disclosed consistently; every run uses the same policy.

## R5. Cost attribution: components and executed work

**Decision**: two run types, kept apart (FR-013).

1. **Clean latency runs**: no hooks, no profiler. Only the outer timer around the public `predict` call.
2. **Profile runs**: `torch.profiler` (CPU activities, operator shapes) plus stage wrappers (R1). Module forward pre-hooks push `record_function` labels for encoder layers, attention modules, MLPs, norms, and head layers, so each profiled operator is attributed to a component. Operator names classify the rest: linear and matmul under an attention module are projections; `scaled_dot_product_attention` is score/value work; other modules map to MLP and normalization; wrapper timers give tokenization/collation, option scoring and decoding.

**Rationale**: forward hooks on a module give only that module's total, so projections and the SDPA op inside one attention module cannot be separated by hooks alone. The profiler sees operators. The report states what fraction of total profiled time the components explain and lists the unexplained remainder (SC-004).

**Executed-work check** (FR-023): for each implementation (native layers and the kernels in R8), run at several lengths and fit the scaling exponent of time and of profiler-reported FLOPs or bytes against length. A claimed sparse saving must show a lower exponent or lower measured cost at long lengths. A dense mask or zeroed head does not qualify. This is the test for the *to verify* item in R3.

**Overhead check**: profile runs are slower. The report states the ratio of profiled to clean total time per length, and uses the clean number for latency.

## R6. Repeats, percentiles, and warmup

**Decision** (defaults, all CLI parameters):

- Warmup: 3 requests, discarded. Detect and record whether the first timed request still differs from the median.
- Timed repeats: 30 per condition at 512 tokens and below, 20 at 1,024 to 2,048, 10 at 4,096 and 8,192.
- Every repeat uses a new seed, so a new fresh document.
- Report min, p50, p95, mean, standard deviation and the number of repeats. With 10 repeats a p95 is close to the maximum, so results with fewer than 20 repeats carry a `low_sample_p95` label.
- Long-length time limit: a per-condition wall-clock cap. A condition that cannot complete the minimum repeats within the cap is recorded as `partial` with the count it reached.

**Rationale**: p95 needs enough samples to mean something (spec: "repeated-run uncertainty"). At 8K on a small CPU a single request may take seconds, so fewer repeats are the only feasible choice, and the label keeps that visible. Thread count is held fixed within a comparison (manifest).

## R7. Token accounting

**Decision**: `tokens.py` returns, per request, `state_tokens_original`, `task_option_tokens`, `special_tokens`, `state_tokens_retained`, `padding_tokens`, `final_length`, plus the position limit and `truncated` flag. The fields are computed from the same functions Laya uses (`build_sequence` with `return_stats=True`, and `collate_items`) plus a separate tokenization of the full state, since `build_sequence` only returns the truncated ids. The record is rejected unless `special + task_option + state_retained + padding` equals the final tensor length.

**Rationale**: `build_sequence` budgets question and options before the state and truncates the state to `max_len - len(head) - 1` (`laya/common.py:187`). Only a comparison with an independent full tokenization shows how much state was dropped. "Padding" is reported apart from evidence: padding only appears when a batch holds sequences of different lengths, so single requests have none.

**Unsupported length rule**: a requested total above `max_position_embeddings` (from the audit) is refused before any model call, with the reason recorded. A total between the configured cap and positional capacity is allowed only because the harness passes an explicit `max_len`, and is labeled as extrapolating past the checkpoint's configured input cap. Whether long positions are usable is a quality question for later phases, and this phase measures cost only. Such results carry `beyond_configured_max_len: true` so the report cannot present them as validated context.

## R8. Kernel reference implementations

**Decision**: pure PyTorch implementations in `experiments/kernels/`, all with the same signature (`q, k, v`, mask spec, returns output), and one dense reference that defines the semantics:

- **reference**: fp32, explicit score matrix, explicit boolean mask, softmax, value product. A fully masked row returns zeros (a defined convention, since raw softmax over all `-inf` gives NaN). Every other kernel is compared to this.
- **dense**: `scaled_dot_product_attention` with the same mask, as in the native path.
- **local**: block-local attention. Queries are grouped into blocks and attend to their own and neighboring key blocks, shaped as a batched matmul over blocks so out-of-window work is not executed.
- **gas** (gather-attend-scatter): for each query block, gather a chosen set of key/value indices (selected by a fixed pattern in Phase 1, not learned), run dense attention on the gathered set, and scatter the results back.

Sequence shapes come from the audit (heads, head dimension, hidden size), across the swept lengths. Timings include mask construction, index building, padding, copies and allocations (FR-021).

**Correctness cases** (FR-022): padding at the end of a row, a boundary chunk shorter than the block size, chunk boundaries that do not divide the length, a query with no allowed keys, and a batch mixing all of these. Tolerance: absolute 1e-5 in fp32 against the reference, stated in the result.

**Rationale**: the goal is to learn whether a sparse implementation *can* be faster on this CPU (plan item 3), not to ship one. The selection rule in `gas` is deliberately fixed so that Phase 1 measures data movement cost, not selection quality.

**Alternatives considered**: `torch.nn.attention.flex_attention` (CPU support and stability vary by torch version and would add a version dependency); custom C++ or Triton kernels (out of scope, and would not be portable).

## R9. Memory feasibility and the cost floor

**Decision**:
- **Memory feasibility**: two things. (a) Analytical: score-matrix bytes as `batch x heads x L x L x dtype-size`, computed and printed for each swept length, labeled analytical. (b) Measured: peak RSS from the per-condition subprocess for the real native run and for each kernel. The report states which measured path avoids materializing the full score matrix and how it knows (peak RSS versus the analytical size).
- **Cost floor**: from the component profile at each length, `floor = total - attention_score_value_time`. This is the time that remains if attention score/value work were free. Projections, MLPs, norms, head layers and data preparation stay. It is labeled an estimate. The plan's operation-count bound (`8 x (1 - f)` at 4K versus 512) is reported next to it as an analytical figure, and the two are not merged.

**Rationale**: constitution II requires keeping analytical counts, kernel timings and end-to-end numbers distinct.

## R10. Testing offline

**Decision**: `tests/experiments/conftest.py` writes a tiny checkpoint to a temp directory: a small randomly initialized ModernBERT config (few layers, small hidden size, a local/global schedule), a matching `DecisionModel`, a small trained-from-nothing tokenizer, and `rl_agent_config.json`, so `laya.Agent(path)` loads it without network access. Unit tests cover manifest fields, audit values against the tiny config, token accounting reconciliation, exact-length input building, percentile summaries, profiler attribution sums, kernel-vs-reference agreement, and report ranking. Integration tests on the pinned checkpoint are marked `slow`.

**Rationale**: the real checkpoint needs a download, and the tooling logic does not depend on trained weights. Timings on a tiny model are not measurements and are never reported.

**Risk**: the installed `transformers` version changes ModernBERT internals (`laya/common.py` already contains version shims). The audit records the version, and tests pin behavior of the tiny model on the installed version.

## R11. Open items carried to the report

These are questions the run must answer, listed so they are not forgotten.

- Which figure is the "target CPU"? The development machine is an M1. If deployment targets another CPU class, the same commands rerun there and produce a separate result set.
- Does the loaded local attention skip out-of-window work, or run dense masked SDPA? (R3, R5)
- At what length, if any, does attention score/value time overtake the rest of the forward pass? The plan assumes near 2K and this is unconfirmed.
- Is 8,192 feasible on this machine's RAM for the native model?
- What is the measured 512-token latency, compared with the ~33 ms historical reference?
- Does the decision head's full-sequence pass contribute enough cost to change which prototype is worth building first?

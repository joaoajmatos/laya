# Laya Context Extension Research Plan

Evaluate attention sparsity, chunk selection, and early compression for 4K–8K decisions on CPU.
This plan follows the [project constitution](../.specify/memory/constitution.md).

## Objective and workload

The primary workload is a **fresh document per decision on CPU**, with model weights resident.
The primary latency case is one document and one question. Measure additional question counts
(2, 5, 10) and batches separately; they must not substitute for single-decision latency.
Each timed request includes tokenization, retrieval/compression, encoding or decoder prefill,
scoring, and postprocessing. Document representations and document KV caches start empty for
each request. Model loading and one-time compilation are measured separately from steady-state
request latency. Fixed-schema question-token caching is allowed if disclosed consistently.

The original ~33 ms at 512 tokens is a historical reference to reproduce on the target CPU,
not an established result for this fork. Attention dominance near 2K and a decoder crossover
near 4K are questions to measure. The original target of at most 2× the measured 512-token
latency at 4K is a **stretch target**, not a hard pass/fail requirement.

## Architectural starting point

Audit the pinned checkpoint and loaded implementation before using these observations:

- The published English encoder configuration reviewed on 2026-09-28 has 8,192 positions,
  local attention of 128, and global attention every third layer. Its Laya configuration sets
  `max_len=512` and `head_layers=2`. The input cap and positional capacity are distinct;
  positional capacity alone does not establish long-context decision quality.
- The unchanged reference is **native Laya**, including its mixed local/global backbone and
  decision head. An all-global backbone is an optional mask ablation, not the native reference.
  A legacy variant identifier such as `full` must be accompanied by the actual configuration.
- `laya/common.py:DecisionModel.forward` runs decision-head transformer layers over the full
  sequence before gathering option markers. Profile and modify this path explicitly where needed;
  unrestricted head layers would bypass a claimed CLS-only information bottleneck.
- `laya/agent.py:_encode_state` creates one state-plus-question sequence per question. State
  tokenization is shared, but encoder work is repeated across question rows.
- `build_sequence` budgets question/options before the state. Ordinary strings retain the start
  when truncated; conversation lists retain the end. Added summary tokens also consume context.
- `predict_long` already provides overlapping-window scoring and aggregation. Its selected-window
  probabilities are not calibrated document probabilities.
- `laya/layers/attention.py` is currently empty. The README selector example describes intended
  behavior; attention integration, training support, and correctness checks remain implementation work.

Sources: [published encoder configuration](https://huggingface.co/convaiinnovations/laya/raw/main/encoder/config.json),
[published Laya configuration](https://huggingface.co/convaiinnovations/laya/raw/main/rl_agent_config.json),
[ModernBERT](https://arxiv.org/abs/2412.13663). Record immutable revisions in experiment manifests;
these moving source URLs are background references.

## Hypotheses and falsification

**H1 — Quality:** question-conditioned chunk selection or compression can preserve decision
quality at 4K–8K relative to native Laya seeing the same evidence. Test coarse semantic decisions
separately from exact-value, exception, counting, temporal, and multi-chunk decisions.

**H2 — CPU efficiency:** reducing attention edges and, where necessary, the number of tokens
processed by later projections, MLPs, and decision-head layers improves the end-to-end CPU
latency–quality trade-off at matching context lengths.

**H3 — Compression capacity:** a small number of summary vectors may suffice for some tasks;
retaining selected token-level representations may be necessary for detailed evidence. Compare
summary-only, selection-only, and hybrid designs under explicit retained-token budgets.

The central hierarchical sparse attention reference uses CLS landmarks to select chunks and
then attends to their token representations. It does not establish that CLS-only scoring is
sufficient. Its trained architecture motivates an adaptation experiment, not a training-free
replacement of Laya. [Hierarchical sparse attention paper](https://arxiv.org/html/2510.17196v2)

Attention sparsity alone does not eliminate token-linear work. If a fraction `f` of baseline
512-token operations can be eliminated while remaining per-token work is unchanged, processing
4K tokens still needs at least approximately `8 * (1 - f)` times those baseline operations.
This is a simplified operation-count bound, not a latency prediction. Profile CPU utilization,
weight traffic, allocations, and non-attention work before deciding whether the stretch target
requires early token reduction, fewer layers, or cheaper state encoding.

## Phase 1 — Audit and measure the complete CPU path

_GPU: unnecessary for primary measurements. GPU measurements are optional, separately labeled._

1. **Pin the execution environment and validate context accounting**
   - Start with the English checkpoint. Treat mmBERT as a separate replication with its own
     tokenizer, checkpoint, native quality, and performance baseline.
   - Record model revision, code revision, Python/PyTorch/Transformers versions, CPU model,
     available RAM, thread counts, BLAS backend, precision, compiler settings, and concurrency.
   - Audit actual attention modules, layer types, RoPE settings, positional limits, head depth,
     and any implicit device/precision fallback. Use an explicit CPU device.
   - Log original state tokens, task/option tokens, special/summary tokens, retained state tokens,
     selected tokens, padded length, evidence retention, and final model input length.
   - Fail or label unsupported lengths explicitly. Setting `max_len` does not force input length;
     padding is not additional evidence. Count all tokens against position limits, including at 8K.

2. **Measure native Laya at actual input lengths**
   - Sweep 128, 256, 512, 1024, 2048, 4096, and 8192 total tokens where supported.
   - Measure single-request latency first; measure multi-question and batch throughput separately.
   - Break down tokenization/collation, backbone QKV/output projections, attention score/value
     operations, MLPs/norms, decision-head transformer layers, option/action scoring, and decoding.
   - Warm model/runtime kernels, then use representative requests with document caches cleared.
     Record repeated timings, p50/p95, peak process RSS, and failures. Reuse pretokenized inputs
     only for separately labeled component measurements.
   - Use profiling to explain costs; collect final latency without intrusive hooks. If GPU results
     are added, use synchronization or device events rather than unsynchronized Python hooks.

3. **Check achievable savings before substantial training**
   - Estimate the cost floor from retained projections/MLPs and all decision-head layers.
   - Microbenchmark dense, local/block, and gather-attend-scatter CPU implementations using realistic
     shapes. Include masks, routing, padding, copies, and allocations in their timings.
   - A dense attention mask or zeroed head does not prove less executed work. Check scaling and
     profiler evidence before claiming a sparse implementation is faster.
   - Compare each implementation to a dense reference with identical mask semantics, including
     padding, boundary chunks, and empty/fully masked-row handling. Check gradients where trained.
   - Quantify memory feasibility: a materialized FP32 tensor shaped `[16, 8192, 8192]` alone is
     4 GiB per sequence. Fused implementations may avoid it; measure the chosen path.

**Deliverable:** pinned run manifest, context audit, component cost curves, memory limits, and a
ranked list of bottlenecks. Use the measured profile to select prototypes rather than assuming
quadratic attention dominates. Dataset construction in Phase 2 can proceed alongside profiling.

## Phase 2 — Build independent benchmarks and practical baselines

_GPU: unnecessary for data work and CPU baseline inference. Budget wall time and RAM explicitly._

4. **Create realistic tasks and controlled length families**
   - Pilot datasets: at least 200 short, 100 medium, and 50 long documents, covering roughly
     <512, 1K–3K, and 4K–8K total-token inputs. These are screening sizes, not a power guarantee.
   - Include email/support routing, procurement or policy decisions, and reports/transcripts.
     Cover `noul`, `choice`, and ordinal `score`, with balanced diagnostic cases and a separately
     reported sample reflecting expected deployment class frequencies.
   - Evaluate the same controlled task families at 512, 1024, 2048, 4096, and 8192 total tokens.
     Maintain short evidence-only counterparts and annotate necessary evidence spans.
   - Verify that the native model can solve the short counterpart. Report baseline-unsolved cases
     separately; sparse failure on them does not isolate a context-processing failure.

   | Controlled family | Required variation |
   |---|---|
   | Distractor growth | Fixed evidence, increasing irrelevant text and near-matching distractors |
   | Position | Evidence at beginning, middle, and end |
   | Chunk boundaries | Evidence within a chunk, straddling a boundary, and with shifted boundaries |
   | Distributed reasoning | Two or more necessary facts in distant chunks |
   | Exceptions and updates | Negation, contradictory records, and later corrections |
   | Aggregation | Any/all conditions, counting, thresholds, and exact-value comparisons |
   | Missing evidence | Unanswerable or unsupported claims with explicit scoring rules |

   - Keep document identifiers, source groups, task family, labels, evidence spans, question/option
     counts, and length metadata in the dataset schema. Freeze labeling rules and adjudicate
     ambiguous gold labels; synthetic labels must come from the generator's known state.
   - Counterbalance option order and use distractor templates independent of the answer label.
     Do not create label leakage through padding text, IDs, or document construction artifacts.

5. **Separate data roles before experimentation**
   - Create disjoint train, development, calibration, and final test splits by document/source group.
     All length, position, and question variants of a document stay together. Hold out synthetic
     template families when evaluating template generalization.
   - Train only on training data, select architecture/hyperparameters on development data, and
     fit output temperatures/decision thresholds on calibration data.
   - Estimate final sample sizes from pilot paired differences and document clustering. Multiple
     questions on one document are not independent samples. Increase data before freezing the final
     protocol if its intended quality margin cannot be resolved.
   - Select finalists before final test evaluation. Do not tune on test outcomes or relabel a
     previously inspected test set as an untouched confirmatory set.

6. **Run the baselines early on development data**

   | Baseline | Purpose and accounting |
   |---|---|
   | Native Laya, matching lengths | Primary architectural reference; same checkpoint and visible evidence |
   | Native Laya, 512-token truncation | Historical cost anchor and truncation control, not the long-context quality reference |
   | Existing `predict_long` | Practical windowing alternative; include all windows and aggregation |
   | Cheap lexical retrieval + native Laya | Include per-document retrieval and scoring; tune retained-token budget on development data |
   | Oracle evidence selection + native Laya | Diagnostic evidence-sufficiency control; not a deployable latency competitor |
   | CPU quantization/runtime optimization | Test a simpler speed alternative and measure its quality/calibration changes |
   | Qwen3-0.6B via `score_shared` | Include fresh-document prefill, all question/option scoring, and cache allocation |

   - Use the same raw documents, questions, labels, and available evidence across model families.
     Record tokenizer-specific lengths and any different truncation; equal token counts alone
     do not imply equal evidence. Validate decoder prompt and option-probability semantics.
   - Keep state prefill inside decoder request latency. Sharing it across questions within one
     request is a separate multi-question measurement, not free processing of a new document.
   - Separate controlled algorithm comparisons from optimized system comparisons with different
     precision/backends. Laya versus a differently trained decoder is a system comparison, not
     proof about encoders versus decoders in general.
   - Record native failures/OOMs at long lengths; do not replace them silently with truncated or
     CPU/GPU-fallback results. Compare at feasible lengths and report the limitation.

**Deliverable:** versioned split manifests, evidence-aware pilot suite, baseline quality/calibration
and CPU curves, and a frozen plan for final sample sizes. This phase can falsify the need for a
complex attention prototype if a simple alternative already satisfies the research targets.

## Phase 3 — Prototype in order of diagnostic value

All variants must specify backbone and decision-head information flow, executed operations,
initialization, trainable modules, and CPU inference support. The adaptation and evaluation rules
in Phase 4 apply during this phase; training is not postponed until final evaluation.

7. **Local attention with existing task tokens global**
   - Start with windows 128, 256, and 512. Designate existing question, option-marker, and CLS
     positions as global; ablate the chosen set and count rather than adding new embeddings first.
   - Distinguish global tokens from global heads. A fixed number of global tokens adds `O(n*g)`
     edges; keeping fully global heads retains an `O(n²)` term.
   - Compare against the actual native local/global layer schedule and use a CPU implementation
     that skips work. Label mask-only prototypes as quality diagnostics until speed is demonstrated.
   - Preserve task-token paths through the decision head. Include a no-adaptation ablation and
     budget matched adaptation if quality changes. Any new learned global embeddings require an
     explicit initialization/training experiment.

8. **Primary experiment: chunk summaries, selection, and early reduction**
   - Start at chunk size 256; test 128 and 512 only after the initial result motivates the sweep.
     Compare fixed boundaries, shifted/overlapping chunks, and document-aware boundaries where useful.
   - Compare summary-only (1, 4, 8 vectors per chunk), landmarks selecting token-level chunks,
     and summaries plus selected tokens. Include an oracle-selection diagnostic and a simple
     pooling control. Learned CLS landmarks are distinct from a claim of lossless compression.
   - Compare question-conditioned and question-independent representations. Fresh-document requests
     already provide the question, so conditioning does not sacrifice a required document cache.
   - Vary the reduction depth: local encoding followed by retaining only task tokens, summaries,
     and/or selected evidence for later layers. Measure the savings in MLPs and projections as well
     as attention. Full-depth chunk encoding still processes every original token.
   - Define which representations the option markers and action-head CLS may access at each stage.
     Prevent raw-state access through a dense downstream head in summary-only runs. Remap markers,
     preserve selected-token positions where appropriate, and document the positional scheme.
   - Compare retained-token budgets, selection recall (including all required chunks), summary
     capacity, cross-chunk accuracy, and short-context regression. Do not expand a full Cartesian
     sweep; use development evidence to select the next ablation.

9. **Secondary: structured head pruning / SQA exploration**
   - Run inexpensive head-importance ablations first, then test modest structural reductions
     before aggressive 4/2/1-head variants. Include recovery training as a separate result.
   - Masking computed heads is a quality ablation. Speed requires reduced executed tensors and
     compatible Q/K/V and output projections. Reducing head count while enlarging head dimension
     does not establish reduced attention arithmetic.
   - SQA's trained small-model results do not establish lossless inference-time pruning of Laya.
     Proceed only if attention remains a material CPU cost after earlier experiments.

10. **Optional: MoSA-style learned token participation**
    - Use learned per-head routing scores and expert-choice top-k selection, not k-means centroids.
      In the cited method, selection controls query, key, and value participation.
    - Define how task/option tokens receive updates, preserve original positions, and measure token
      coverage, missing evidence, routing cost, and gather/scatter overhead on CPU.
    - Train the router and any necessary surrounding projections. Compare with simpler selection
      under matched data, adaptation, and retained-work budgets before increasing complexity.

11. **Optional: ASEntmax**
    - Treat this first as a quality/length-generalization experiment. The cited method includes
      learned context-dependent scaling; a fixed entmax activation swap is a separate ablation.
    - Exact zeros do not by themselves skip score computation. Include normalization overhead and
      any loss of optimized attention paths when measuring CPU latency and memory.
    - Evaluate calibration rather than assuming sparser attention improves output probabilities.
      Defer deployment work unless an efficient CPU path and a useful quality trade-off are shown.

**Deliverable:** correctness-checked prototypes, explicit information-flow descriptions, CPU
implementation measurements, and a small development-selected shortlist. Stop or redirect variants
whose failures are explained by CPU overhead or evidence loss; record those results.

## Phase 4 — Adapt, calibrate, and evaluate under a fixed protocol

12. **Budget adaptation from the start**
    - Separate no-adaptation runs, new-module/head training, and adapter/LoRA recovery. Train on
      disjoint training data with a mixture of short and long examples; never train on final test
      rows or calibration labels. Synthetic evidence tasks can supplement owned training data.
    - Compare each adapted candidate with native Laya adapted using the same data and a declared
      comparable budget. Record both training tokens/steps and wall time; exact equality of every
      resource is not assumed. Distillation from native predictions is an optional training ablation.
    - Determine feasible batch size, precision, activation checkpointing, trainable modules, and
      optimizer memory in a pilot. LoRA freezes base weights but still incurs activation/backward
      costs. Do not assume 200 examples or two GPU hours suffice to learn a new routing architecture.
    - Use at least three training seeds for finalist robustness when resources permit. If only one
      is feasible, state that training variance is unmeasured and restrict conclusions accordingly.

13. **Measure quality, calibration, evidence retention, and cost**

    | Dimension | Required reporting |
    |---|---|
    | Decisions | Accuracy and class-specific errors; macro-F1 for imbalanced classification; MAE for ordinal expected scores |
    | Calibration | NLL, Brier score, reliability plots, and ECE with fixed documented bins; inherited and held-out recalibrated temperatures |
    | Confidence | Classification ECE from unrounded max answer probability; risk/coverage when escalation is evaluated |
    | Evidence | Truncation, selected evidence recall, all-required-chunks recall, and errors by position/boundary/task family |
    | CPU performance | End-to-end p50/p95, repeated-run uncertainty, component times, and separate batch throughput |
    | Memory/resources | Peak process RSS, memory failures, training time/resources, and any fallback events |
    | Statistical uncertainty | Paired 95% document-cluster bootstrap intervals for quality deltas; paired request/repeated-run latency comparisons |

    - `confidence` is entropy-based for `choice`/`score`; use raw probabilities or the semantics of
      `answer_confidence` for classification calibration. Do not confuse answer probability with
      `action.act_probability`. Top-label ECE for a score distribution describes its modal class,
      not the correctness of the returned fractional expected score; report ordinal MAE separately.
    - Recalibrate window aggregation at document level on the calibration split. Do not interpret
      the winning window's probability as already calibrated for a document with many windows.
    - Report task types and critical failure families separately. Predeclare aggregation weights;
      a strong coarse-classification result must not hide failures on exceptions or distributed facts.
    - Keep caches, timing boundaries, threads, precision, and evidence equivalent within each
      comparison. Mark unsupported cells and missing measurements rather than dropping them.

14. **Freeze finalists and run the final test once per declared protocol**
    - Evaluate shortlisted variants and controls at supported 512/1K/2K/4K/8K lengths, separating
      short-context preservation from performance on genuinely long inputs.
    - Lock model selection, margins, aggregation weights, calibration fits, timing protocol, and
      critical subgroups before final test access. Publish pilot/exploratory results separately.
    - If confidence intervals cannot resolve a declared margin, report the finding as inconclusive.
      Follow-up confirmatory claims require a new independent test or a predeclared sequential design.

**Deliverable:** reproducible final report with uncertainty, error slices, calibration plots,
latency–quality trade-offs, and explicit data/training/resource limits.

## Phase 5 — Decide what the evidence supports

15. **Apply the research targets and publish scoped conclusions**

    Use these **research defaults**, not unconfirmed product acceptance requirements. Any changes must
    be documented from pilot/development evidence before the final test protocol is frozen.

    - **Quality target:** at 4K and, separately, 8K, the lower paired 95% confidence bound on candidate
      accuracy minus matching-length native accuracy is at least -2 percentage points. Trained variants
      must also meet this margin against their matched-adaptation native control. Apply the same
      default to short-context regression and predeclared critical classification families. Ordinal-only
      tasks use a predeclared domain-appropriate MAE margin instead of an invented accuracy threshold.
    - **Latency target:** at least 20% lower end-to-end p50 CPU latency than native Laya at the same
      length, with a speed improvement supported by timing uncertainty and no measured p95 regression.
      Report memory within a declared target-machine budget and all calibration changes alongside it.
    - **Practical comparison:** compare qualifying candidates with windowing, retrieval, optimized native
      inference, and decoder scoring. Prefer non-dominated systems on quality, CPU latency, calibration,
      and memory; a native-relative win alone does not establish the best deployment choice.
    - **Stretch target:** 4K end-to-end CPU latency no greater than 2× the newly measured native
      512-token latency. Also report the 8K ratio, without treating a stretch miss as research failure.

    A result may support a deployable improvement, a useful architectural result beaten by a simpler
    system, a quality gain with excess latency, or a bounded negative/inconclusive finding. If native
    quality is itself inadequate or a same-length reference cannot run within memory limits, report
    that explicitly; non-inferiority to a weak baseline is not sufficient evidence of useful decisions.
    Publish reproducible positive and negative findings. Failure of the tested sparse variants does
    not establish that decoder-with-KV-cache is universally the correct architecture.

## Resource allocation and experiments without GPU

| Work | CPU inference / analysis | Training accelerator |
|---|---|---|
| Audit, profiling, masks, cost/reachability checks | Required on target CPU | Unnecessary |
| Data generation, labeling, splits, power/calibration analysis | CPU/offline | Unnecessary |
| Native, windowing, lexical/oracle retrieval, decoder baselines | CPU; check RAM and wall time | Unnecessary for inference |
| Fixed local/global-token ablation | CPU | Optional for adaptation |
| Learned summaries or new global tokens | CPU deployment must be verified | Recommended for substantive training |
| Structured pruning / SQA | CPU ablation and deployment checks | Recommended for recovery training |
| MoSA | CPU routing/selection feasibility first | Recommended for learned routing/adaptation |
| ASEntmax | CPU functional and efficiency checks | Recommended for learned scaling/adaptation |
| Quantization/runtime baseline | CPU; measure quality as well as speed | Depends on chosen method |

Do not conflate GPU training with GPU deployment. CPU-only toy training and frozen-feature probes
are useful diagnostics, but they do not establish that full-scale adaptation is affordable.
Estimate engineering and training time after Phase 1 and a training pilot, with kernel integration,
labeling, repeated seeds, and final evaluation included. Allocate substantial training only after
both the information-preservation tests and target-CPU implementation checks justify it.

## References and how they inform the experiments

- [ModernBERT](https://arxiv.org/abs/2412.13663): native long-position capacity and mixed local/global
  encoder design; inspect the actual fine-tuned checkpoint rather than assuming its configuration.
- [Hierarchical sparse attention and length generalization](https://arxiv.org/html/2510.17196v2):
  CLS landmarks for chunk retrieval with retained token-level memory. Our summary-only scoring
  ablation is a stronger bottleneck and is explicitly a different hypothesis.
- [Longformer](https://arxiv.org/abs/2004.05150): local attention plus selected global token positions;
  distinguish this pattern from keeping fully global heads.
- [MoSA](https://arxiv.org/html/2505.00315v1): learned expert-choice token participation per head;
  online k-means is a different routing baseline.
- [SQA](https://arxiv.org/html/2510.01817v1): structurally reduced query-head computation and trained
  small-model evidence, not a guarantee for inference-time pruning of a Laya checkpoint.
- [ASEntmax](https://arxiv.org/html/2506.16640v3): sparse normalization with learned context-dependent
  scaling; output calibration and CPU speed remain separate empirical questions.
- Additional pattern references: [BigBird](https://arxiv.org/abs/2007.14062),
  [NSA](https://arxiv.org/abs/2502.11089),
  [Sparse Frontier](https://arxiv.org/abs/2504.17768), and
  [interleaved token selection](https://arxiv.org/abs/2602.03216).
- [Upstream Laya](https://github.com/NandhaKishorM/laya): checkpoint/runtime provenance. This fork's
  source and pinned artifacts are authoritative for the path actually benchmarked.

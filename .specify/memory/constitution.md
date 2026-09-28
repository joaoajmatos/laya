<!--
Sync Impact Report (temporary amendment review material; remove before committing)
Version: 1.0.0 -> 1.1.0
Bump rationale: new workload and data-integrity principles plus expanded research controls.
The native reference and public inference contract are preserved.
Modified principles:
- I. Preserve the Inference Contract: explicitly protect typed outputs and information paths.
- II. Measure Research Claims: distinguish hypotheses, measured savings, and negative results.
- III. Establish a Valid Baseline -> Establish a Valid Native Baseline: correct the description
  of the unchanged reference and require comparisons at matching context lengths.
- IV. Keep the Runtime Focused: include compression and selection within research scope.
- V. Make Experiments Reproducible: expand run metadata and resource reporting.
Added principles:
- VI. Optimize Fresh-Document CPU Decisions
- VII. Protect Evaluation Independence
Expanded sections: Technical Constraints, Workflow, Governance.
Removed sections: None.
Related document updated by explicit user request: docs/research-plan.md.
Template changes: None; templates consume the constitution at runtime.
Follow-up TODOs: None.
-->

# Laya-Sparse Constitution

## Core Principles

### I. Preserve the Inference Contract

Attention, selection, and compression variants MUST preserve Laya's prediction interface and
typed answer semantics unless an explicit feature specification requires a change. Implementations
MUST preserve option-marker attribution, padding behavior, probability meanings, and action-head
inputs. Variant selection MUST isolate experimental behavior from the unchanged native path.
Architecture changes MUST specify information flow through both the backbone and decision head;
an unrestricted downstream layer MUST NOT silently bypass a claimed compression restriction.

### II. Measure Research Claims

Performance, context-length, and quality claims MUST identify reproducible runs, models, datasets,
hardware, and configurations. Reports MUST distinguish hypotheses, analytical operation counts,
kernel timings, and end-to-end measurements. Mathematical sparsity or fewer active weights MUST
NOT be presented as measured runtime savings. Reports MUST include quality, calibration, latency,
memory, and actual processed context where applicable. Planned APIs and unrun experiments MUST
be labeled as planned. Negative and inconclusive results are valid research outcomes; conclusions
MUST be limited to the evaluated workloads, implementations, and resource budgets.

### III. Establish a Valid Native Baseline

The reference MUST be the unmodified, pinned Laya checkpoint and its actual attention configuration.
"Native" does not imply that every layer uses full attention: the loaded configuration MUST be
audited, including the backbone, decision-head layers, positions, and input cap. Quality comparisons
MUST use equivalent evidence at matching context lengths; a truncated 512-token run alone cannot
establish long-context quality preservation. Studies MUST include relevant simple alternatives and
separate architectural comparisons with controlled settings from comparisons of optimized systems.
Adapted variants MUST have a native control with a matched adaptation budget. Methodology changes
MUST be versioned with the results they affect.

### IV. Keep the Runtime Focused

Production library changes MUST remain focused on single-model inference and the attention,
selection, compression, or scoring changes needed by a documented experiment. Benchmark
orchestration, training experiments, datasets, and generated results MUST live in `experiments/`.
Integrations, services, exporters, and packaging surfaces removed from upstream MUST NOT be
restored unless justified by a feature specification. Complexity MUST be motivated by a measured
bottleneck or a stated information-preservation hypothesis.

### V. Make Experiments Reproducible

Runs MUST record model revisions, data versions and splits, seeds, attention configuration,
trainable modules, training budget, software versions, and relevant hardware/runtime settings.
Performance metadata MUST include actual sequence lengths, question and option counts, precision,
batching, thread settings, cache policy, and fallback events. Resource reports MUST separate
training requirements from deployment requirements. Commands needed to reproduce results MUST
accompany reports. Model weights, caches, credentials, and third-party raw records MUST NOT be
committed. Generated results belong in ignored locations unless reviewed artifacts are explicitly
required by a specification.

### VI. Optimize Fresh-Document CPU Decisions

The primary deployment workload MUST be CPU inference with a fresh document for each decision,
resident model weights, and no reusable document representations or document KV cache. Primary
latency MUST include tokenization, selection/compression, document encoding or decoder prefill,
scoring, and postprocessing. Single-document, single-question latency MUST be reported separately
from throughput and multi-question workloads. GPU results or cached-document results MUST NOT
substitute for measurements on the target CPU. These constraints ensure that reported savings
apply to the intended decision path rather than an amortization opportunity it does not have.

### VII. Protect Evaluation Independence

Training, development, calibration, and final test data MUST be disjoint at the document/source
group level. Related synthetic variants MUST remain in the same split; held-out template families
MUST be included when claiming template generalization. Model, threshold, and temperature selection
MUST use their designated non-test splits. Final metrics, quality margins, latency targets, and
comparison procedures MUST be fixed before final test evaluation. Quality reports MUST include
paired uncertainty estimates respecting shared documents and identify underpowered or inconclusive
comparisons. Length studies MUST include controlled evidence-preserving examples alongside realistic
tasks, with truncation and evidence retention recorded.

## Technical Constraints

- Support Python 3.10 and newer; retain the dependency and packaging approach in `pyproject.toml`
  unless a feature specification documents a reason to change it.
- Deployment candidates MUST run on the target CPU. Accelerator-only research prototypes MUST be
  labeled as such and MUST NOT qualify as successful CPU deployment results.
- New training or kernel dependencies MUST state their device requirements and fail clearly when
  unavailable. GPU training is permitted; it does not imply GPU inference is required.
- Experiments MUST account for the full forward pass, including MLPs, projections, decision-head
  attention, and data preparation. Dense masks and head masking require executed-work verification
  before a speed claim.
- Correctness checks for sparse kernels MUST use a reference with identical mask semantics;
  unrestricted-attention equality is required only when the patterns are equivalent.
- Total context MUST include task tokens, options, special tokens, and summaries. Position limits,
  marker indices, truncation, and selected-token positions MUST be validated after transformations.
- Calibration MUST use documented probability semantics. Entropy-based confidence MUST NOT be
  substituted for predicted-answer probability in classification ECE.
- Public behavior and defaults MUST remain backward compatible unless explicitly changed by a
  feature specification. Prototype selection MUST be explicit and reversible.

## Workflow

- Start feature implementation with a user-focused specification, technical plan, and actionable
  tasks. Research-plan and governance amendments can directly document agreed changes in direction.
- Inspect the code and `docs/research-plan.md` before proposing architecture changes. Verify cited
  mechanisms against their primary sources and document deviations from those mechanisms.
- Establish native baselines and a cost profile before committing substantial training effort.
  Use controlled evidence tests and target-CPU implementation checks to select promising variants.
- Define correctness, quality, calibration, latency, and memory checks appropriate to each change.
  Run the required checks and report checks that could not be run.
- Freeze the evaluation protocol before opening the final test set. Exploratory findings MUST be
  reported separately from confirmatory results; failed resource limits MUST remain visible.
- Keep generated outputs and machine-specific files out of commits. Review the final diff for
  unintended changes to results, dependencies, public interfaces, and research methodology.

## Governance

This constitution guides feature specifications, research plans, implementation, and review.
Reviews MUST assess compliance with the applicable principles and document deviations and their
effect on interpretation. If a feature request changes a principle, the conflict MUST be resolved
explicitly in the specification or a constitution amendment before implementation.

Amendments MUST state their rationale and affected principles, update the research plan when
needed, preserve the original ratification date, and update the version and amendment date.
Versioning uses MAJOR for incompatible governance changes, MINOR for new principles or materially
expanded guidance, and PATCH for non-semantic corrections. Experimental thresholds and schedules
belong in the research plan; they MUST NOT be represented as user-approved deployment requirements
without supporting instructions.

**Version**: 1.1.0 | **Ratified**: 2026-09-28 | **Last Amended**: 2026-09-28

# Laya-Sparse Constitution

## Core Principles

### I. Preserve the Inference Contract
Changes to attention variants must preserve the existing Laya prediction interface unless an explicitly specified feature requires a change. Keep attention selection pluggable and isolate variant-specific behavior from the full-attention baseline.

### II. Measure Research Claims
Treat this repository as an experimental research fork. Performance, context-length, and quality claims must be tied to reproducible runs on identified models, datasets, hardware, and configurations. Report accuracy or calibration alongside latency and context size where those measures apply. Do not present planned variants or unrun experiments as shipped results.

### III. Establish a Valid Baseline
Use the existing full bidirectional attention path as the reference when evaluating sparse variants. Compare variants on equivalent inputs and settings, and make limitations or tradeoffs visible. Changes to evaluation methodology must be documented with the results they affect.

### IV. Keep the Runtime Focused
Keep production library changes limited to single-model inference and attention experimentation. Put benchmark orchestration, profiling, datasets, and generated outputs in the `experiments/` tree. Avoid restoring integrations, services, exporters, or packaging surfaces removed from upstream unless a feature specification justifies them.

### V. Make Experiments Reproducible
Record the model and data identifiers, attention configuration, relevant software or hardware details, and commands needed to reproduce results. Do not commit model weights, caches, credentials, or third-party raw records. Generated benchmark output belongs in ignored output locations unless a specification explicitly calls for a reviewed artifact.

## Technical Constraints

- Support Python 3.10 and newer and retain the package metadata and dependency approach in `pyproject.toml` unless a specification documents a reason to change them.
- Preserve CPU usability for the baseline and variants that claim CPU support. GPU-only variants must fail clearly or document their requirements when GPU support is unavailable.
- Keep public behavior and defaults backward compatible unless the feature specification explicitly approves a change.
- Do not make speed or memory claims without measurements from the repository's experiment harnesses or an equivalent documented procedure.

## Workflow

- Start feature work with a user-focused specification, then produce a technical plan and actionable tasks before implementation.
- Inspect existing code and `docs/research-plan.md` before proposing architectural changes; avoid duplicating planned work or silently changing research priorities.
- Identify the relevant correctness, quality, latency, and memory measurements in the plan. Run checks appropriate to the change and report any that could not be run.
- Keep generated experiment output and machine-specific files out of commits. Review the final diff for unintended changes to results, dependencies, or public interfaces.

## Governance

This constitution guides specifications and implementation plans for Laya-Sparse. If it conflicts with a feature request, surface the conflict in the specification and resolve it explicitly before implementation. Amend this document when project priorities or constraints materially change, and update its version and amendment date in the same change.

**Version**: 1.0.0 | **Ratified**: 2026-09-28 | **Last Amended**: 2026-09-28

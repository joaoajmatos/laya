# Laya-Sparse

Extending Laya's encoder-only decision models to larger context windows via sparse attention.

Laya is reported at ~33 ms / decision at 512 tokens (hardware not stated; Laya's own runtime message gives ~35 ms on GPU and ~200-500 ms on CPU, so it is most likely a GPU figure). It is not a CPU baseline for this fork. The English checkpoint's encoder mixes global attention (10 of 28 layers) with local attention over a 128-token window (±64) in the other 18, and its 2-layer decision head attends over the full sequence. Whether attention dominates runtime beyond ~2K tokens is an open question that Phase 1 measures ([`specs/001-cpu-path-audit`](specs/001-cpu-path-audit/spec.md)). This fork experiments with chunk-based sparse attention, global CLS tokens, and content-based sparsity to push to 4K–8K context at comparable speed.

This is a research fork. It strips Laya down to the core inference path and adds pluggable attention variants. The full research plan is in [`docs/research-plan.md`](docs/research-plan.md).

## Quickstart

```bash
pip install -e .
python -c "
from laya import load
agent = load('convaiinnovations/laya')
result = agent.predict('short state', {'q1': {'type': 'noul', 'instructions': 'test?'}})
print(result['answers'])
"
```

## Running the Phase 1 measurements

Phase 1 measures the unmodified model on the target CPU: a pinned manifest, an audit of what the loaded model is, clean latency, a component profile, attention kernel microbenchmarks and a bottleneck report. Setup, commands and expected results are in [`specs/001-cpu-path-audit/quickstart.md`](specs/001-cpu-path-audit/quickstart.md); on Windows, `experiments/setup_windows.ps1` builds the CPU environment. The short version:

```powershell
python -m experiments all --run-id full --revision reviewed
```

Results go to `experiments/results/<run-id>/` (git-ignored), ending in `report.md`. Timings on the fixture model used by the tests are not measurements. GPU numbers come only from the separate `gpu-reference` check and never stand in for CPU results.

## Attention variants

| Variant | Description | GPU needed | Status |
|---|---|---|---|
| `native` | Unmodified Laya: 10 global layers and 18 local layers (128-token window), plus a 2-layer head over the full sequence. The Phase 1 audit found the local window applied as a mask over full-length attention | No | Shipped |
| `sliding` | Fixed-size local window + global heads | No | Planned |
| `chunk_cls` | State split into chunks, each encoded by a CLS token | Yes | Primary experiment |
| `mosa` | MoSA-style dynamic token selection per head | Yes | Secondary |
| `sqa` | Reduced query heads | No | Quick test |

Only `native` exists today. `laya/layers/attention.py` is still empty, so the selection API below is **planned** and does not run yet:

```python
from laya import load
from laya.layers.attention import set_attention_pattern

agent = load('convaiinnovations/laya')
set_attention_pattern('sliding', window_size=256, num_global=2)
result = agent.predict(long_state, questions)
```

## Repository structure

```
laya-sparse/
├── laya/                  # Core library (trimmed from upstream)
│   ├── __init__.py
│   ├── agent.py           # Agent, load, predict / predict_batch
│   ├── common.py          # Model, sequence building, shared utilities
│   ├── layers/
│   │   └── attention.py   # Planned home of attention variants (empty today)
│   └── ...                # Other internal modules from upstream
├── experiments/           # Phase 1 measurement tooling (python -m experiments)
│   ├── cli.py             # Commands: manifest, audit, sweep, profile, kernels, report, all, gpu-reference
│   ├── kernels/           # Dense, masked, block-local and gather-attend-scatter attention kernels
│   ├── ...                # manifest, audit, tokens, inputs, runner, timing, profile, floor, report, gpu
│   └── results/           # Output artifacts (gitignored)
├── tests/experiments/     # Offline tests on a tiny fixture model (pytest -m "not slow")
├── specs/                 # Spec Kit features (001-cpu-path-audit)
├── docs/
│   ├── research-plan.md   # The experiment roadmap
│   └── sparse-attention-report.md   # Draft report (claims marked as hypotheses until measured)
├── pyproject.toml
├── README.md
└── LICENSE
```

## Spec-driven development

This repository uses [GitHub Spec Kit](https://github.com/github/spec-kit) with Codex and Claude Code. The same workflow skills are installed in each agent's recognized folder: `.agents/skills/` for Codex and `.claude/skills/` for Claude Code. Skill files use a shared format, but agents do not all discover skills from the same folder. Shared templates and helper scripts are in `.specify/`, and project-specific principles are recorded in `.specify/memory/constitution.md`.

Use the skills in order for a feature. In Codex, invoke them as `$speckit-specify`, `$speckit-clarify` (when requirements are ambiguous), `$speckit-plan`, `$speckit-tasks`, and `$speckit-implement`. In Claude Code, invoke those same skills as `/speckit-specify`, `/speckit-clarify`, `/speckit-plan`, `/speckit-tasks`, and `/speckit-implement`. `$speckit-analyze`/`/speckit-analyze` and `$speckit-checklist`/`/speckit-checklist` are optional review steps. See the constitution for this project's research and reproducibility constraints.

## Comparison against decoder models

Results from this fork can be compared against decoder-based scorers (e.g. Qwen3-0.6B via the JEV-CPU repo) by running both against the same JSONL benchmark files.

```
Same JSONL → Laya fork (encoder, sparse) vs JEV-CPU (decoder, KV cache)
           → accuracy | calibration | latency | context size
```

## What's removed from upstream Laya

- Router, MCP, CLI, HTTP server
- LangChain / LlamaIndex / CrewAI integrations
- ONNX export / TileLang GPU kernels
- Docker / Nix flake
- TypeScript package
- Community benchmarks (feishu, zh_short_commands)
- Research eval harnesses and evals
- Notebooks, docs, assets, CI workflows
- Everything not needed for single-model inference + experimentation

## License

Apache 2.0 (inherited from upstream Laya)

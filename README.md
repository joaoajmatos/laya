# Laya-Sparse

Extending Laya's encoder-only decision models to larger context windows via sparse attention.

Laya does ~33 ms / decision at 512 tokens with full bidirectional attention. Beyond ~2K tokens, encoder O(n²) attention dominates runtime. This fork experiments with chunk-based sparse attention, global CLS tokens, and content-based sparsity to push to 4K–8K context at comparable speed.

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

## Attention variants

| Variant | Description | GPU needed | Status |
|---|---|---|---|
| `full` | Original full bidirectional attention (Laya baseline) | No | Shipped |
| `sliding` | Fixed-size local window + global heads | No | Planned |
| `chunk_cls` | State split into chunks, each encoded by a CLS token | Yes | Primary experiment |
| `mosa` | MoSA-style dynamic token selection per head | Yes | Secondary |
| `sqa` | Reduced query heads | No | Quick test |

Select at inference time (no training required for sliding/sqa):

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
│   ├── agent.py           # predict / predict_batch
│   ├── load.py            # (merged into agent.py upstream)
│   ├── common.py          # Shared utilities
│   ├── layers/            # Pluggable components
│   │   ├── attention.py   # Attention variants: full, sliding, chunk_cls, mosa, sqa
│   │   └── __init__.py
│   └── ...                # Other internal modules from upstream
├── experiments/           # All experiment code
│   ├── suite/             # Per-task harnesses
│   ├── datasets/          # Benchmarks (JSONL)
│   ├── scripts/           # Profiling and orchestration
│   └── results/           # Output artifacts (gitignored)
├── docs/
│   └── research-plan.md   # The experiment roadmap
├── pyproject.toml
├── README.md
└── LICENSE
```

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

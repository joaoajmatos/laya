# Quickstart (laya:004)

```powershell
.venv\Scripts\python -m experiments golden weights-inventory --out artifacts/raya-golden
.venv\Scripts\python -m experiments golden export            --out artifacts/raya-golden
.venv\Scripts\python -m experiments golden kernels           --out artifacts/raya-golden   # large; --lengths 256 for one size
.venv\Scripts\python -m pytest tests/experiments/test_golden.py
```

The generators force one thread regardless of `--threads`.
Outputs: `inventory.json`, `weights-h64.safetensors`, `weights-h128.safetensors`, `<case>/{tensors.safetensors,meta.json}`,
`kernels/<impl>_L<n>/{tensors.safetensors,meta.json}`.

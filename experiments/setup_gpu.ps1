# Separate CUDA environment for the GPU reference check (research.md R15).
# The CPU environment (.venv) stays CPU-only on purpose. Run after the CPU sweep has finished:
#   powershell -ExecutionPolicy Bypass -File experiments\setup_gpu.ps1
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) { $root = Split-Path -Parent $root }
if (-not $root) { Write-Host "pyproject.toml not found" -ForegroundColor Red; exit 1 }
Set-Location $root

Write-Host "=== GPU and driver ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ($LASTEXITCODE -ne 0) { Write-Host "nvidia-smi failed: is the NVIDIA driver installed?" -ForegroundColor Red; exit 1 }

$base = (py -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')" 2>$null)
if (-not $base) { Write-Host "No Python >= 3.10 found" -ForegroundColor Red; exit 1 }
if (-not (Test-Path ".venv-gpu\Scripts\python.exe")) { & $base -m venv .venv-gpu }
$py = Join-Path $root ".venv-gpu\Scripts\python.exe"
& $py -m pip install --upgrade pip | Out-Host

# Newest CUDA build first; fall back when the driver is too old for it.
$ok = $false
foreach ($cu in @("cu130", "cu128", "cu126")) {
    Write-Host "=== Trying PyTorch $cu ===" -ForegroundColor Cyan
    & $py -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/$cu" | Out-Host
    if ($LASTEXITCODE -ne 0) { continue }
    & $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Write-Host "$cu installed but CUDA is not usable with this driver; trying an older build" -ForegroundColor Yellow
}
if (-not $ok) { Write-Host "No CUDA build of PyTorch works with this driver. Update the NVIDIA driver and retry." -ForegroundColor Red; exit 1 }

& $py -m pip install --upgrade -e . | Out-Host
& $py -c "import torch; p = torch.cuda.get_device_properties(0); print('torch', torch.__version__, 'cuda', torch.version.cuda, '|', p.name, '%d.%d' % (p.major, p.minor), '|', round(p.total_memory / 2**30, 1), 'GiB')"
Write-Host "Done. Run: .venv-gpu\Scripts\python -m experiments gpu-reference --run-id full" -ForegroundColor Green

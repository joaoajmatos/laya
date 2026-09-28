# Phase 1 environment setup for the measuring machine (native Windows).
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File experiments\setup_windows.ps1
# Writes a full log to experiments\results\env-check.txt (git-ignored).

$ErrorActionPreference = "Continue"
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit $LASTEXITCODE)" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 } }
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "experiments\results" | Out-Null
$log = Join-Path $root "experiments\results\env-check.txt"
Start-Transcript -Path $log -Force | Out-Null

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. PC info
Section "Hardware"
$cpu = Get-CimInstance Win32_Processor
$cs  = Get-CimInstance Win32_ComputerSystem
$os  = Get-CimInstance Win32_OperatingSystem
[pscustomobject]@{
    CPU            = $cpu.Name.Trim()
    PhysicalCores  = ($cpu | Measure-Object NumberOfCores -Sum).Sum
    LogicalCores   = ($cpu | Measure-Object NumberOfLogicalProcessors -Sum).Sum
    MaxClockMHz    = $cpu.MaxClockSpeed
    L2KB           = $cpu.L2CacheSize
    L3KB           = $cpu.L3CacheSize
    RAM_GB         = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    FreeRAM_GB     = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 1)
    OS             = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
    Arch           = $env:PROCESSOR_ARCHITECTURE
} | Format-List

Section "Memory modules"
Get-CimInstance Win32_PhysicalMemory |
    Select-Object BankLabel, @{n="GB";e={$_.Capacity/1GB}}, Speed, ConfiguredClockSpeed |
    Format-Table -AutoSize

Section "Page file"
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage | Format-Table -AutoSize
Write-Host "Automatic page file management: $($cs.AutomaticManagedPagefile)"

Section "Power"
powercfg /getactivescheme
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat) { Write-Host "Battery present, status: $($bat.BatteryStatus) (2 = on AC)" } else { Write-Host "No battery (desktop, always on AC)" }

Section "Disk (repo drive)"
$drive = (Get-Item $root).PSDrive.Name
Get-PSDrive $drive | Select-Object Name, @{n="FreeGB";e={[math]::Round($_.Free/1GB,1)}}, @{n="UsedGB";e={[math]::Round($_.Used/1GB,1)}} | Format-Table -AutoSize

Section "Heaviest running processes (by memory)"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 12 Name, @{n="MB";e={[math]::Round($_.WorkingSet64/1MB)}} | Format-Table -AutoSize

# ---------------------------------------------------------------- 2. Python
Section "Python"
$want = "3.12"
$hasLauncher = [bool](Get-Command py -ErrorAction SilentlyContinue)
if ($hasLauncher) { py -0p }
$have = $false
if ($hasLauncher) { py -$want -c "import sys" *> $null; $have = ($LASTEXITCODE -eq 0) }
if (-not $have) {
    Write-Host "Python $want not found; installing with winget (python.org build)."
    winget install -e --id Python.Python.$want --scope user --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path","User") + ";" + [Environment]::GetEnvironmentVariable("Path","Machine")
}
py -$want --version; Check "Python $want"

# ---------------------------------------------------------------- 3. venv + packages
Section "Virtual environment (.venv, git-ignored)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -$want -m venv .venv; Check "venv" }
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel; Check "pip upgrade"

# CPU-only PyTorch wheel: this is a CPU study, and a CPU wheel makes a silent GPU path impossible.
& $py -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu; Check "torch"
# Project (editable) plus the experiments extra (pytest). transformers brings tokenizers.
& $py -m pip install --upgrade -e ".[experiments]"; Check "project install"

# ---------------------------------------------------------------- 4. Verify
Section "Verification"
$verify = Join-Path $env:TEMP "laya_verify.py"
@"
import platform, sys, torch, transformers, numpy, safetensors, tokenizers
print('python        ', sys.version.split()[0], platform.machine())
print('torch         ', torch.__version__, '| cuda built:', torch.version.cuda)
print('transformers  ', transformers.__version__)
print('tokenizers    ', tokenizers.__version__)
print('numpy         ', numpy.__version__, '| safetensors', safetensors.__version__)
print('cpu capability', torch.backends.cpu.get_cpu_capability())
print('mkldnn        ', torch.backends.mkldnn.is_available(), '| mkl', torch.backends.mkl.is_available())
print('threads       ', 'intra', torch.get_num_threads(), '| inter', torch.get_num_interop_threads())
x = torch.randn(512, 512); print('matmul ok     ', float((x @ x).sum()) == float((x @ x).sum()))
import laya; print('laya import    ok')
"@ | Out-File -Encoding ascii $verify
& $py $verify; Check "verification"
& $py -m pip freeze | Out-File -Encoding utf8 "experiments\results\pip-freeze.txt"
Write-Host "`nPackage list saved to experiments\results\pip-freeze.txt"

Stop-Transcript | Out-Null
Write-Host "`nDone. Log: $log" -ForegroundColor Green

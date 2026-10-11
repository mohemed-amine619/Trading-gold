<#
.SYNOPSIS
    Create the project virtual environment on Windows.

.DESCRIPTION
    Creates .\.venv with Python 3.12. Does NOT install packages —
    that is install-dependencies.ps1's job, so each step can be run
    and diagnosed independently.

.EXAMPLE
    .\scripts\windows\setup-venv.ps1
#>

[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Virtual environment setup"

# ── Locate a suitable Python ──────────────────────────────────────────────────

$launcher = $null
foreach ($candidate in @("py", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) { $launcher = $candidate; break }
}
if (-not $launcher) {
    Write-Fail "Python not found. Install Python 3.12 from https://www.python.org/downloads/windows/"
    Write-Info  "During install, tick 'Add python.exe to PATH'."
    exit 1
}

# Resolve the actual version we would use.
$versionText = if ($launcher -eq "py") { & py "-$PythonVersion" --version 2>&1 } else { & python --version 2>&1 }
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Python $PythonVersion not found (tried '$launcher')."
    Write-Info "Install Python $PythonVersion from https://www.python.org/downloads/windows/"
    exit 1
}
Write-Ok "Found $versionText"

# ── MT5 compatibility check ───────────────────────────────────────────────────
# MetaTrader5 publishes wheels for specific Python versions. 3.12 is current;
# if the user only has 3.10/3.11 they may still find a wheel, so this is a
# warning rather than a hard stop — the install step is the real test.
if ($versionText -notmatch "3\.12") {
    Write-Warn "Python $versionText — MetaTrader5 wheels are published most reliably for 3.12."
    Write-Info  "If 'pip install MetaTrader5' fails, install Python 3.12 and re-run."
}

# ── Create the venv ───────────────────────────────────────────────────────────

$paths = Get-Paths
if (Test-Path $paths.Python) {
    Write-Ok "Virtual environment already exists at $($paths.Venv)"
    exit 0
}

Write-Step "Creating virtual environment..."
& $launcher $(if ($launcher -eq "py") { "-$PythonVersion" } else { "" }) -m venv $paths.Venv
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to create the virtual environment."
    exit 1
}

if (-not (Test-Path $paths.Python)) {
    Write-Fail "venv created but no interpreter found at $($paths.Python)"
    exit 1
}

& $paths.Python --version | Out-Null
Write-Ok "Virtual environment ready: $($paths.Python)"

Write-Info "Next: .\scripts\windows\install-dependencies.ps1"
exit 0
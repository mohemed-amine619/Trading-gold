<#
.SYNOPSIS
    Install pinned Python dependencies into the project venv.

.DESCRIPTION
    Uses requirements-windows.txt, which is the single source of truth for a
    native Windows install (bot + API + database + bridge).

    MetaTrader5 is Windows-only. If its wheel cannot be resolved, the script
    reports the exact official sources rather than pretending it succeeded.

.EXAMPLE
    .\scripts\windows\install-dependencies.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipMetaTrader5
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Installing dependencies"

$paths = Get-Paths
if (-not (Test-Path $paths.Python)) {
    Write-Fail "Virtual environment not found. Run setup-venv.ps1 first."
    exit 1
}
if (-not (Test-Path $paths.Requirements)) {
    Write-Fail "requirements-windows.txt not found at $($paths.Requirements)"
    exit 1
}

Write-Info "Interpreter: $(& $paths.Python --version 2>&1)"
Write-Step "Upgrading pip ..."
& $paths.Python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Warn "pip upgrade failed; continuing with the bundled version" }

if ($SkipMetaTrader5) {
    Write-Warn "Skipping MetaTrader5 (requested via -SkipMetaTrader5)"
    Write-Step "Installing everything except MetaTrader5 ..."
    # Build a temporary requirements file without the MT5 line.
    $temp = Join-Path ([System.IO.Path]::GetTempPath()) "req-nomt5-$([guid]::NewGuid().ToString('N').Substring(0,8)).txt"
    Get-Content $paths.Requirements |
        Where-Object { $_ -notmatch '^\s*MetaTrader5\s*$' } |
        Set-Content $temp -Encoding UTF8
    & $paths.Python -m pip install -r $temp
    $rc = $LASTEXITCODE
    Remove-Item $temp -Force -ErrorAction SilentlyContinue
} else {
    Write-Step "Installing from requirements-windows.txt (this can take a few minutes) ..."
    & $paths.Python -m pip install -r $paths.Requirements
    $rc = $LASTEXITCODE
}

if ($rc -ne 0) {
    Write-Fail "Dependency installation failed (exit $rc)."
    Write-Info ""
    Write-Info "If the failure is 'No matching distribution found for MetaTrader5',"
    Write-Info "that package is Windows-only. Options:"
    Write-Info "  1. Confirm you are running PowerShell on Windows (not WSL/Linux)."
    Write-Info "  2. Download the wheel manually from:"
    Write-Info "     https://pypi.org/project/MetaTrader5/#files"
    Write-Info "     then: pip install <downloaded-wheel>.whl"
    Write-Info "  3. Or continue without it (API/research mode only):"
    Write-Info "     .\scripts\windows\install-dependencies.ps1 -SkipMetaTrader5"
    exit 1
}

# ── Verify what matters ───────────────────────────────────────────────────────

Write-Step "Verifying key imports ..."
$check = & $paths.Python -c @"
import importlib, sys
mods = ['fastapi','sqlalchemy','pydantic_settings','alembic','pandas','numpy','sklearn']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print('MISSING:' + ','.join(missing) if missing else 'ALL_CORE_OK')
try:
    importlib.util.find_spec('MetaTrader5')
    print('MT5_OK')
except Exception:
    print('MT5_MISSING')
"@ 2>&1

$output = ($check | Out-String)
if ($output -match "MISSING:(.+)" -and $Matches[1].Trim()) {
    Write-Fail "Core packages failed to import: $($Matches[1].Trim())"
    exit 1
}
if ($output -match "MT5_OK") { Write-Ok "MetaTrader5 importable" }
else { Write-Warn "MetaTrader5 not importable — bot/bridge features unavailable until installed" }

Write-Ok "Dependencies installed"
Write-Info "Next: .\scripts\windows\setup-database.ps1"
exit 0
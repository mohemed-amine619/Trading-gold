<#
.SYNOPSIS
    One-shot installer for the native Windows platform.

.DESCRIPTION
    Runs the full sequence and stops at the first hard failure:

      0. doctor      - establish the starting state
      1. venv        - create .\.venv with Python 3.12
      2. deps        - install pinned dependencies
      3. .env        - create from the template if absent (never overwrites)
      4. database    - verify/create schema
      5. diagnostics - re-check everything

    It does NOT install PostgreSQL or MetaTrader 5; those need manual,
    official downloads. Where one is missing the script says so and tells you
    the exact source.

.EXAMPLE
    .\scripts\windows\install.ps1
.EXAMPLE
    .\scripts\windows\install.ps1 -SkipDependencies
#>

[CmdletBinding()]
param(
    [switch]$SkipDoctor,
    [switch]$SkipDependencies,
    [switch]$SkipMetaTrader5
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

$paths = Get-Paths
$started = Get-Date

function Invoke-Step {
    param([string]$Name, [string]$Script, [string[]]$Arguments = @())
    Write-Host ""
    Write-Host "-> $Name" -ForegroundColor Cyan
    & "$PSScriptRoot\$Script" @Arguments
    return $LASTEXITCODE
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Trading Platform - Windows Installer" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

if (-not $SkipDoctor) {
    Write-Host ""
    Write-Info "Pre-flight diagnostics (failures here are informational at this stage)"
    & "$PSScriptRoot\doctor.ps1"
}

# ── 1. Virtual environment ────────────────────────────────────────────────────
if ((Invoke-Step "Creating virtual environment" "setup-venv.ps1") -ne 0) {
    Write-Fail "Virtual environment setup failed. Aborting."
    exit 1
}

# ── 2. Dependencies ───────────────────────────────────────────────────────────
if (-not $SkipDependencies) {
    $depArgs = @()
    if ($SkipMetaTrader5) { $depArgs = @("-SkipMetaTrader5") }
    if ((Invoke-Step "Installing dependencies" "install-dependencies.ps1" $depArgs) -ne 0) {
        Write-Fail "Dependency installation failed. Aborting."
        exit 1
    }
}

# ── 3. Configuration ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "-> Configuration" -ForegroundColor Cyan
if (-not (Test-Path $paths.EnvFile)) {
    Copy-Item $paths.EnvExample $paths.EnvFile
    Write-Ok "Created .env from .env.example"
    Write-Warn "Edit .env and set your values before trading."
    Write-Host ""
    Write-Host "  Minimum: TRADING_MODE (keep 'paper'), DATABASE_URL" -ForegroundColor White
    Write-Host "  For live: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, DASHBOARD_SECRET_KEY" -ForegroundColor White
} else {
    Write-Ok ".env already exists (not overwritten)"
}

Initialize-RuntimeDirectories

# ── 4. Database ───────────────────────────────────────────────────────────────
if ((Invoke-Step "Database setup" "setup-database.ps1") -ne 0) {
    Write-Warn "Database setup reported an issue."
    Write-Host ""
    Write-Info "PostgreSQL: https://www.postgresql.org/download/windows/"
    Write-Info "MetaTrader 5: https://www.metatrader5.com/en/download"
    Write-Info "Or switch to SQLite in .env (DATABASE_URL=sqlite:///./trading.db)"
}

$stampCode = Invoke-Step "Applying migrations" "migrate-database.ps1"
if ($stampCode -eq 2) {
    Write-Warn "Existing schema detected but not stamped."
    Write-Info "Run: .\scripts\windows\migrate-database.ps1 -Stamp"
} elseif ($stampCode -ne 0) {
    Write-Warn "Migrations did not complete - see docs/DATABASE.md"
}

# ── 5. Summary ────────────────────────────────────────────────────────────────
$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Installation finished in ${elapsed}s" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Review .env" -ForegroundColor Gray
Write-Host "  2. Run diagnostics: .\scripts\windows\doctor.ps1" -ForegroundColor Gray
Write-Host "  3. Start the platform: .\scripts\windows\start.ps1" -ForegroundColor Gray
Write-Host ""
Write-Info "Read docs/WINDOWS_INSTALLATION.md for the full walkthrough."
exit 0
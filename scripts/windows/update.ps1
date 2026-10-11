<#
.SYNOPSIS
    Update the platform after a code or dependency change.

.DESCRIPTION
    Safe, repeatable update:
      1. back up the database
      2. pull / apply code changes (only if -Pull is passed)
      3. reinstall pinned dependencies
      4. apply migrations
      5. restart services

    Never resets the database.

.EXAMPLE
    .\scripts\windows\update.ps1
.EXAMPLE
    .\scripts\windows\update.ps1 -Pull
#>

[CmdletBinding()]
param(
    [switch]$Pull,
    [switch]$SkipBackup
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

Write-Section "Update"

$paths = Get-Paths
if (-not (Test-Path $paths.Python)) { Write-Fail "Not installed. Run install.ps1 first."; exit 1 }

# ── 1. Backup first ───────────────────────────────────────────────────────────
if (-not $SkipBackup) {
    & "$PSScriptRoot\backup.ps1"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Backup failed - aborting update to protect data."; exit 1 }
} else {
    Write-Warn "Skipping backup (-SkipBackup). Data risk if migrations change."
}

# ── 2. Code ───────────────────────────────────────────────────────────────────
if ($Pull) {
    if (-not (Get-Command git -EA SilentlyContinue)) {
        Write-Fail "git not found; cannot -Pull."
        exit 1
    }
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    Write-Step "Pulling $branch ..."
    & git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "git pull failed. Resolve locally, then re-run."
        Write-Info "Your working tree was NOT modified by this script."
        exit 1
    }
    Write-Ok "Code updated"
} else {
    Write-Info "Skipping git pull (pass -Pull to fetch changes)"
}

# ── 3. Dependencies ───────────────────────────────────────────────────────────
Write-Step "Reinstalling pinned dependencies ..."
& $paths.Python -m pip install -r $paths.Requirements --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Dependency update failed."; exit 1 }
Write-Ok "Dependencies updated"

# ── 4. Migrations ─────────────────────────────────────────────────────────────
$code = & "$PSScriptRoot\migrate-database.ps1"
$exit = $LASTEXITCODE
if ($exit -eq 2) {
    Write-Warn "Schema exists but is unstamped. Run: .\scripts\windows\migrate-database.ps1 -Stamp"
} elseif ($exit -ne 0) {
    Write-Fail "Migration failed. The database was NOT modified further."
    Write-Info "Restore with: .\scripts\windows\restore.ps1 -File <backup>"
    exit 1
}

# ── 5. Restart ────────────────────────────────────────────────────────────────
if (Get-RunningProcess -Name "api") {
    Write-Step "Restarting the API ..."
    & "$PSScriptRoot\stop.ps1" -Target api | Out-Null
    Start-Sleep -Seconds 2
    & "$PSScriptRoot\start.ps1" | Out-Null
    Write-Ok "API restarted"
}

Write-Ok "Update complete"
exit 0
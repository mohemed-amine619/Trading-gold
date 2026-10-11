<#
.SYNOPSIS
    Remove the platform's local installation.

.DESCRIPTION
    Removes the virtual environment, runtime directories and generated state.
    NEVER removes: your source code, .env, or any backup files. Removing the
    virtual environment and logs is safe and reproducible; deleting your
    configuration or trading history is not, so this script refuses.

    Use -RemoveDatabase only if you explicitly want the local SQLite file
    gone, and it will ask first.

.EXAMPLE
    .\scripts\windows\uninstall.ps1
#>

[CmdletBinding()]
param(
    [switch]$RemoveDatabase,
    [switch]$KeepVirtualEnv
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

Write-Section "Uninstall"

$paths = Get-Paths

Write-Info "This removes local runtime files only."
Write-Host "  Preserved: source code, .env, backups\" -ForegroundColor Gray
Write-Host ""

# Stop anything running first.
if (Get-RunningProcess -Name "api" -or Get-RunningProcess -Name "bot") {
    Write-Step "Stopping running processes ..."
    & "$PSScriptRoot\stop.ps1" | Out-Null
}

# ── Virtual environment ───────────────────────────────────────────────────────
if (-not $KeepVirtualEnv -and (Test-Path $paths.Venv)) {
    Write-Host "  Remove virtual environment? [y/N]: " -NoNewline -ForegroundColor Yellow
    if ((Read-Host) -eq "y") {
        Remove-Item $paths.Venv -Recurse -Force -EA SilentlyContinue
        Write-Ok "Removed .venv"
    } else { Write-Info "Kept .venv" }
}

# ── Runtime directories ───────────────────────────────────────────────────────
foreach ($dir in @($paths.Logs, $paths.Run)) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force -EA SilentlyContinue
        Write-Ok "Removed $dir"
    }
}

# ── Backups: offered, never automatic ─────────────────────────────────────────
if (Test-Path $paths.Backups) {
    $count = (Get-ChildItem $paths.Backups -File -EA SilentlyContinue).Count
    if ($count -gt 0) {
        Write-Warn "backups\ contains $count file(s). These were NOT removed."
        Write-Info "Delete manually if you no longer need them."
    }
}

# ── SQLite database ───────────────────────────────────────────────────────────
if ($RemoveDatabase) {
    if (Test-Path $paths.Database) {
        Write-Host "  Delete trading.db and ALL its data? [y/N]: " -NoNewline -ForegroundColor Red
        if ((Read-Host) -eq "y") {
            Remove-Item $paths.Database -Force -EA SilentlyContinue
            foreach ($s in @("-wal", "-shm")) { Remove-Item "$($paths.Database)$s" -Force -EA SilentlyContinue }
            Write-Ok "Removed local SQLite database"
        } else { Write-Info "Kept trading.db" }
    }
} else {
    Write-Info "trading.db preserved (use -RemoveDatabase to delete)"
}

Write-Host ""
Write-Ok "Uninstall complete"
Write-Info "PostgreSQL is a Windows service and was NOT touched."
exit 0
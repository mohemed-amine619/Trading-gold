<#
.SYNOPSIS
    Restore the database from a backup.

.DESCRIPTION
    DESTRUCTIVE. Overwrites the current database. This script therefore
    requires an interactive confirmation and always takes a safety backup of
    the current state first.

.EXAMPLE
    .\scripts\windows\restore.ps1 -File .\backups\trading-20260101-120000.sql
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$File,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Database restore"

$paths = Get-Paths
if (-not (Test-Path $File)) { Write-Fail "Backup file not found: $File"; exit 1 }

$dbUrl = Get-Setting "DATABASE_URL" ""
$pg    = Get-PostgresConnection -Url $dbUrl

Write-Host ""
Write-Host "  WARNING: this OVERWRITES the current database." -ForegroundColor Red
Write-Host "  Target : $($paths.Database)" -ForegroundColor White
Write-Host "  Backup : $File" -ForegroundColor White
Write-Host ""

if ($File -notmatch "\.db$") {
    Write-Fail "SQLite restore expects a .db file (got: $File)."
    exit 1
}

# Verify source backup file before doing anything
$verifyScript = @"
import sqlite3, sys
try:
    c = sqlite3.connect(r'''$File''')
    cur = c.cursor()
    cur.execute('PRAGMA integrity_check')
    res = cur.fetchone()
    c.close()
    if not res or res[0] != 'ok':
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
"@
$checkBackup = & $paths.Python -c $verifyScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Source backup file failed integrity check. Aborting restore."
    exit 1
}

if (-not $Force) {
    Write-Host "  A safety backup of the CURRENT database will be taken first." -ForegroundColor Yellow
    Write-Host "  Type 'RESTORE' to proceed, anything else to abort: " -NoNewline -ForegroundColor Yellow
    $answer = Read-Host
    if ($answer -cne "RESTORE") { Write-Info "Aborted. Nothing changed."; exit 1 }
}

# ── Safety backup of current state if it exists ──────────────────────────────
if (Test-Path $paths.Database) {
    Write-Step "Taking a safety backup of the current database ..."
    $safety = & "$PSScriptRoot\backup.ps1" -KeepDays 30
    if ($LASTEXITCODE -ne 0) { Write-Fail "Safety backup failed - refusing to restore."; exit 1 }
    Write-Ok "Safety backup complete"
}

# ── Stop the app so nothing writes during the restore ─────────────────────────
$apiProc = Get-RunningProcess -Name "api"
if ($apiProc) {
    Write-Step "Stopping the API to prevent concurrent writes ..."
    Stop-ProcessByName -ProcessId $apiProc.Id -GraceSeconds 10 | Out-Null
    Clear-StoredPid -Name "api"
}

Write-Step "Restoring SQLite database ..."
if (-not (Test-Path $paths.Data)) {
    New-Item -ItemType Directory -Path $paths.Data -Force | Out-Null
}
foreach ($suffix in @("-wal", "-shm")) {
    if (Test-Path "$($paths.Database)$suffix") { Remove-Item "$($paths.Database)$suffix" -Force -EA SilentlyContinue }
}
Copy-Item $File $paths.Database -Force

if (Test-DatabaseReachable) {
    Write-Ok "Database restored and integrity verified from $File"
} else {
    Write-Fail "Restored database failed integrity check!"
    exit 1
}

Write-Info "Verify with: .\scripts\windows\migrate-database.ps1 -ShowCurrent"
exit 0
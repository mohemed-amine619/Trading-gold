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
Write-Host "  Target : $(if ($pg) { "$($pg.Host):$($pg.Port)/$($pg.Database)" } else { $dbUrl })" -ForegroundColor White
Write-Host "  Backup : $File" -ForegroundColor White
Write-Host ""

if (-not $Force) {
    Write-Host "  A safety backup of the CURRENT database will be taken first." -ForegroundColor Yellow
    Write-Host "  Type 'RESTORE' to proceed, anything else to abort: " -NoNewline -ForegroundColor Yellow
    $answer = Read-Host
    if ($answer -cne "RESTORE") { Write-Info "Aborted. Nothing changed."; exit 1 }
}

# ── Safety backup of current state ────────────────────────────────────────────
Write-Step "Taking a safety backup of the current database ..."
$safety = & "$PSScriptRoot\backup.ps1" -KeepDays 30
if ($LASTEXITCODE -ne 0) { Write-Fail "Safety backup failed - refusing to restore."; exit 1 }
Write-Ok "Safety backup complete"

# ── Stop the app so nothing writes during the restore ─────────────────────────
$apiProc = Get-RunningProcess -Name "api"
if ($apiProc) {
    Write-Step "Stopping the API to prevent concurrent writes ..."
    Stop-ProcessByName -ProcessId $apiProc.Id -GraceSeconds 10 | Out-Null
    Clear-StoredPid -Name "api"
}

if ($pg) {
    $psql = Get-PsqlCommand
    $psqlRestore = $null
    if (Get-Command "psql" -EA SilentlyContinue) { $psqlRestore = "psql" }
    else {
        foreach ($root in @("C:\Program Files\PostgreSQL", "C:\Program Files (x86)\PostgreSQL")) {
            if (Test-Path $root) {
                $c = Get-ChildItem $root -Filter "psql.exe" -Recurse -EA SilentlyContinue | Select-Object -First 1
                if ($c) { $psqlRestore = $c.FullName; break }
            }
        }
    }
    if (-not $psqlRestore) { Write-Fail "psql not found."; exit 1 }

    Write-Step "Restoring SQL dump ..."
    $env:PGPASSWORD = $pg.Password
    try {
        # --clean --if-exists removes existing objects so the dump is authoritative.
        & $psqlRestore -h $pg.Host -p $pg.Port -U $pg.User -d $pg.Database `
                       --clean --if-exists -v ON_ERROR_STOP=1 -f $File
        if ($LASTEXITCODE -ne 0) { Write-Fail "Restore failed (exit $LASTEXITCODE)."; exit 1 }
    } finally {
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
    Write-Ok "Database restored from $File"
}
else {
    if ($File -notmatch "\.db$") { Write-Fail "SQLite restore expects a .db file."; exit 1 }
    Write-Step "Restoring SQLite database ..."
    foreach ($suffix in @("-wal", "-shm")) {
        if (Test-Path "$($paths.Database)$suffix") { Remove-Item "$($paths.Database)$suffix" -Force -EA SilentlyContinue }
    }
    Copy-Item $File $paths.Database -Force
    Write-Ok "Database restored from $File"
}

Write-Info "Verify with: .\scripts\windows\migrate-database.ps1 -ShowCurrent"
exit 0
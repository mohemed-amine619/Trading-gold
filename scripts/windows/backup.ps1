<#
.SYNOPSIS
    Back up the database.

.DESCRIPTION
    PostgreSQL: uses pg_dump for a consistent, restorable dump.
    SQLite:     copies the .db/.db-wal/.db-shm files together (checkpointing first).

    Backups land in .\backups with a timestamp. Retention is applied only to
    files this script created (trading_*), never to unrelated files.

.EXAMPLE
    .\scripts\windows\backup.ps1
.EXAMPLE
    .\scripts\windows\backup.ps1 -KeepDays 30
#>

[CmdletBinding()]
param(
    [int]$KeepDays = 14,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Database backup"

$paths = Get-Paths
if (-not $OutputDir) { $OutputDir = $paths.Backups }
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$dbUrl   = Get-Setting "DATABASE_URL" ""
$pg      = Get-PostgresConnection -Url $dbUrl

if ($pg) {
    $psql = Get-PsqlCommand
    $pgDump = $null
    if (Get-Command "pg_dump" -ErrorAction SilentlyContinue) { $pgDump = "pg_dump" }
    else {
        foreach ($root in @("C:\Program Files\PostgreSQL", "C:\Program Files (x86)\PostgreSQL")) {
            if (Test-Path $root) {
                $c = Get-ChildItem $root -Filter "pg_dump.exe" -Recurse -EA SilentlyContinue | Select-Object -First 1
                if ($c) { $pgDump = $c.FullName; break }
            }
        }
    }
    if (-not $pgDump) { Write-Fail "pg_dump not found. Add PostgreSQL\bin to PATH."; exit 1 }

    $target = Join-Path $OutputDir "trading-$stamp.sql"
    Write-Step "Dumping PostgreSQL to $target ..."

    $env:PGPASSWORD = $pg.Password
    try {
        & $pgDump -h $pg.Host -p $pg.Port -U $pg.User -d $pg.Database -F p -f $target
        if ($LASTEXITCODE -ne 0) { Write-Fail "pg_dump failed (exit $LASTEXITCODE)"; exit 1 }
    } finally {
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
    $size = [math]::Round((Get-Item $target).Length / 1KB, 1)
    Write-Ok "Backup written: $target ($size KB)"
}
elseif ($dbUrl -match "sqlite") {
    if (-not (Test-Path $paths.Database)) { Write-Fail "SQLite database not found at $($paths.Database)"; exit 1 }
    $target = Join-Path $OutputDir "trading-$stamp.db"
    Write-Step "Copying SQLite database ..."

    # Checkpoint so the .db file is self-contained before copying.
    $ck = & $paths.Python -c "import sqlite3; c=sqlite3.connect(r'$($paths.Database)'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()" 2>$null
    Copy-Item $paths.Database $target -Force
    foreach ($suffix in @("-wal", "-shm")) {
        $side = "$($paths.Database)$suffix"
        if (Test-Path $side) { Copy-Item $side "$target$suffix" -Force }
    }
    $size = [math]::Round((Get-Item $target).Length / 1KB, 1)
    Write-Ok "Backup written: $target ($size KB)"
}
else {
    Write-Fail "DATABASE_URL is not recognised."
    exit 1
}

# ── Retention (only our own files) ────────────────────────────────────────────
$cutoff = (Get-Date).AddDays(-$KeepDays)
$removed = 0
foreach ($file in Get-ChildItem -Path $OutputDir -Filter "trading-*" -File -EA SilentlyContinue) {
    if ($file.LastWriteTime -lt $cutoff) {
        Remove-Item $file.FullName -Force -EA SilentlyContinue
        $removed++
    }
}
if ($removed -gt 0) { Write-Info "Removed $removed backup(s) older than $KeepDays days" }

Write-Info "Restore with: .\scripts\windows\restore.ps1 -File `"$target`""
exit 0
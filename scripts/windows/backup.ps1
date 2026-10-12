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
if (-not ($dbUrl -match "^sqlite")) {
    Write-Fail "DATABASE_URL must be SQLite. Found: $dbUrl"
    exit 1
}

if (-not (Test-Path $paths.Database)) {
    Write-Fail "SQLite database not found at $($paths.Database)"
    exit 1
}

$target = Join-Path $OutputDir "trading-$stamp.db"
Write-Step "Creating online atomic SQLite backup to $target ..."

$pyScript = @"
import sqlite3
import sys

src_path = r'''$($paths.Database)'''
dst_path = r'''$target'''

try:
    src = sqlite3.connect(src_path, timeout=30.0)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst, pages=100)
    dst.close()
    src.close()
    
    # Verify backup integrity
    verify = sqlite3.connect(dst_path)
    cur = verify.cursor()
    cur.execute('PRAGMA integrity_check')
    res = cur.fetchone()
    verify.close()
    if not res or res[0] != 'ok':
        print(f"INTEGRITY_CHECK_FAILED: {res}", file=sys.stderr)
        sys.exit(1)
    print("BACKUP_SUCCESS")
except Exception as ex:
    print(f"BACKUP_ERROR: {ex}", file=sys.stderr)
    sys.exit(1)
"@

$backupResult = & $paths.Python -c $pyScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "SQLite online backup failed: $backupResult"
    exit 1
}

$size = [math]::Round((Get-Item $target).Length / 1KB, 1)
Write-Ok "Backup verified and written: $target ($size KB)"

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
<#
.SYNOPSIS
    Apply database migrations (Alembic).

.DESCRIPTION
    Runs `alembic upgrade head`. Safe to re-run: applying migrations that are
    already applied is a no-op. This script never downgrades or resets data
    unless you explicitly pass -DowngradeTo.

.EXAMPLE
    .\scripts\windows\migrate-database.ps1
.EXAMPLE
    .\scripts\windows\migrate-database.ps1 -ShowCurrent
#>

[CmdletBinding()]
param(
    [switch]$ShowCurrent,
    [string]$Revision,
    [switch]$DowngradeTo,
    [switch]$Autogenerate,
    [switch]$Stamp,
    [string]$Message = "auto migration"
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Database migrations"

$paths = Get-Paths
if (-not (Test-Path $paths.Python)) {
    Write-Fail "Virtual environment missing. Run install.ps1 first."
    exit 1
}

if (-not (Test-Path (Join-Path $paths.Root "alembic.ini"))) {
    Write-Fail "alembic.ini not found in $($paths.Root)"
    exit 1
}

Set-Location $paths.Root

# ── Informational ─────────────────────────────────────────────────────────────
if ($ShowCurrent) {
    Write-Step "Current revision:"
    & $paths.Python -m alembic current
    Write-Step "Migration history:"
    & $paths.Python -m alembic history
    exit $LASTEXITCODE
}

# ── Autogenerate ──────────────────────────────────────────────────────────────
if ($Autogenerate) {
    Write-Step "Generating migration: $Message ..."
    & $paths.Python -m alembic revision --autogenerate -m $Message
    if ($LASTEXITCODE -ne 0) { Write-Fail "Autogenerate failed."; exit 1 }
    Write-Ok "Migration file created under alembic\versions"
    Write-Warn "Review the generated file before applying it."
    exit 0
}

# ── Downgrade (explicit and guarded) ──────────────────────────────────────────
if ($DowngradeTo) {
    Write-Warn "Downgrading to '$Revision' — this removes schema, not data."
    Write-Host "Continue? Type 'yes' to proceed: " -NoNewline -ForegroundColor Yellow
    $answer = Read-Host
    if ($answer -ne "yes") { Write-Info "Aborted."; exit 1 }
    & $paths.Python -m alembic downgrade $Revision
    exit $LASTEXITCODE
}

# ── Detect the "tables exist but never stamped" case ──────────────────────────
# Databases created by the legacy Base.metadata.create_all() path contain the
# schema but an EMPTY alembic_version table. Running upgrade head against one
# fails with "table X already exists". Detect it and offer a stamp, which is
# safe: it records the revision without altering data.
if (-not $DowngradeTo -and -not $Autogenerate -and -not $Stamp) {
    # The child process needs DATABASE_URL in its own environment: it lives in
    # .env, which pydantic-settings reads but os.environ does not.
    $env:DATABASE_URL = (Get-Setting "DATABASE_URL" "")
    $stampCheck = @'
import os, sys
from sqlalchemy import create_engine, text, inspect
url = os.environ.get("DATABASE_URL", "")
if not url:
    sys.exit(0)
try:
    insp = inspect(create_engine(url))
    tables = set(insp.get_table_names())
except Exception:
    sys.exit(0)
if "alembic_version" not in tables:
    sys.exit(0)          # not migrated yet -> normal first-run path
with create_engine(url).connect() as conn:
    rows = [r[0] for r in conn.execute(text("SELECT version_num FROM alembic_version"))]
if not rows and len([t for t in tables if t != "alembic_version"]) > 0:
    sys.exit(3)          # schema present but unstamped
sys.exit(0)
'@
    $stampCheck = $stampCheck -replace "`r`n", "`n"
    & $paths.Python -c $stampCheck
    if ($LASTEXITCODE -eq 3) {
        Write-Warn "Schema exists but was never stamped by Alembic."
        Write-Info "This database was created by the legacy create_all() path."
        Write-Host ""
        Write-Info "Recommended: record the current schema as applied (no data change):" -ForegroundColor White
        Write-Host "    .\scripts\windows\migrate-database.ps1 -Stamp" -ForegroundColor Cyan
        Write-Host ""
        Write-Info "For a FRESH database, delete the file and re-run instead." -ForegroundColor White
        exit 2
    }
}

# ── Stamp (opt-in, non-destructive) ───────────────────────────────────────────
if ($Stamp) {
    Write-Step "Stamping database at head (schema is recorded, not modified) ..."
    $env:DATABASE_URL = (Get-Setting "DATABASE_URL" "")
    & $paths.Python -m alembic stamp head
    if ($LASTEXITCODE -ne 0) { Write-Fail "Stamp failed."; exit 1 }
    Write-Ok "Database stamped at head"
    exit 0
}

# ── Upgrade ───────────────────────────────────────────────────────────────────
$before = (& $paths.Python -m alembic current 2>&1 | Out-String).Trim()
Write-Info "Current: $before"

Write-Step "Applying migrations ..."
& $paths.Python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Migration failed."
    Write-Info "Check the database is reachable and the schema is not locked."
    Write-Info "Backup first if the failure occurred mid-migration: .\scripts\windows\backup.ps1"
    exit 1
}

$after = (& $paths.Python -m alembic current 2>&1 | Out-String).Trim()
if ($after -ne $before) {
    Write-Ok "Schema updated: $after"
} else {
    Write-Ok "Schema already up to date"
}
exit 0
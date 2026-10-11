<#
.SYNOPSIS
    Create and verify the database.

.DESCRIPTION
    Supports two backends:
      * SQLite  - nothing to do; the file is created on first use.
      * PostgreSQL - verifies the server is up and the database exists,
                     and prints the exact SQL to run if it does not.

    This script NEVER drops, recreates or resets an existing database.

.EXAMPLE
    .\scripts\windows\setup-database.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipConnectionCheck
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Database setup"

$dbUrl = Get-Setting "DATABASE_URL" ""
if (-not $dbUrl) {
    Write-Fail "DATABASE_URL is not set. Add it to .env"
    exit 1
}

$pg = Get-PostgresConnection -Url $dbUrl

if (-not $pg) {
    if ($dbUrl -match "sqlite") {
        Write-Ok "SQLite configured: $dbUrl"
        Write-Info "The database file is created automatically on first use."
        Write-Info "Apply the schema with: .\scripts\windows\migrate-database.ps1"
        exit 0
    }
    Write-Fail "DATABASE_URL is not a recognised PostgreSQL or SQLite URL."
    exit 1
}

# ── PostgreSQL path ───────────────────────────────────────────────────────────
Write-Info "Target: $($pg.Host):$($pg.Port)/$($pg.Database) as user '$($pg.User)'"

$svc = Get-PostgresService
if ($svc) {
    Write-Ok "PostgreSQL service running: $($svc.Name)"
} else {
    Write-Fail "No running PostgreSQL service."
    Write-Info "Start it:  Start-Service postgresql*"
    Write-Info "Or install from https://www.postgresql.org/download/windows/"
    exit 1
}

$psql = Get-PsqlCommand
if (-not $psql) {
    Write-Warn "psql not found. Add PostgreSQL\bin to PATH, or run:"
    Write-Info "  C:\Program Files\PostgreSQL\<version>\bin\psql.exe"
    exit 1
}
Write-Ok "psql found"

if ($SkipConnectionCheck) {
    Write-Info "Skipping connection check (-SkipConnectionCheck)"
    exit 0
}

$env:PGPASSWORD = $pg.Password
try {
    # Does the database exist? Never create or drop it here.
    & $psql -h $pg.Host -p $pg.Port -U $pg.User -d "postgres" -tAc `
        "SELECT 1 FROM pg_database WHERE datname='$($pg.Database)'" 2>$null | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Cannot authenticate to PostgreSQL as user '$($pg.User)'."
        Write-Info "Check POSTGRES credentials in .env"
        exit 1
    }

    $exists = & $psql -h $pg.Host -p $pg.Port -U $pg.User -d "postgres" -tAc `
        "SELECT 1 FROM pg_database WHERE datname='$($pg.Database)'" 2>$null

    if ($exists.Trim() -eq "1") {
        Write-Ok "Database '$($pg.Database)' exists"
    } else {
        Write-Warn "Database '$($pg.Database)' does not exist."
        Write-Host ""
        Write-Info "Create it manually (this script will not do it for you):"
        Write-Host "    psql -h $($pg.Host) -p $($pg.Port) -U $($pg.User) -d postgres" -ForegroundColor White
        Write-Host '    CREATE DATABASE "' + $pg.Database + '";' -ForegroundColor White
        Write-Host ""
    }

    # Does the app user exist?
    $roleCount = & $psql -h $pg.Host -p $pg.Port -U $pg.User -d "postgres" -tAc `
        "SELECT count(*) FROM pg_roles WHERE rolname='$($pg.User)'" 2>$null
    if ($roleCount.Trim() -eq "0") {
        Write-Warn "Role '$($pg.User)' does not exist."
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Info "Next: .\scripts\windows\migrate-database.ps1"
exit 0
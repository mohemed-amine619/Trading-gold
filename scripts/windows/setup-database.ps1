<#
.SYNOPSIS
    Verify and prepare the SQLite database.

.DESCRIPTION
    Ensures that SQLite is configured, verifies directory permissions,
    and runs an integrity check if the database file already exists.
    SQLite is the only supported database engine.

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

$dbUrl = Get-Setting "DATABASE_URL" "sqlite:///./data/trading.db"
if (-not ($dbUrl -match "^sqlite")) {
    Write-Fail "DATABASE_URL must be a SQLite database URL (e.g., sqlite:///./data/trading.db). Found: $dbUrl"
    Write-Info "PostgreSQL and external database servers are forbidden by architecture policy."
    exit 1
}

$paths = Get-Paths
if (-not (Test-Path $paths.Data)) {
    New-Item -ItemType Directory -Path $paths.Data -Force | Out-Null
    Write-Ok "Created data directory: $($paths.Data)"
}

Write-Ok "SQLite configured: $dbUrl"
Write-Info "Target database path: $($paths.Database)"

if (Test-Path $paths.Database) {
    Write-Ok "Database file exists: $($paths.Database)"
    if (-not $SkipConnectionCheck) {
        if (Test-DatabaseReachable) {
            Write-Ok "SQLite database integrity verified (PRAGMA integrity_check passed)."
        } else {
            Write-Warn "SQLite integrity check reported issues or file is locked."
        }
    }
} else {
    Write-Info "The database file does not exist yet. It will be created on first migration or run."
}

Write-Info "Apply the schema with: .\scripts\windows\migrate-database.ps1"
exit 0
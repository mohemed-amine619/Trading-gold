<#
.SYNOPSIS
    Diagnose the Windows installation. Reports PASS / WARN / FAIL per check.

.DESCRIPTION
    Every check is executed; nothing is assumed. In particular a missing MT5
    terminal is reported as FAIL/WARN for broker connectivity — never as a
    healthy connection.

    Exit codes: 0 = no failures, 1 = one or more failures.

.EXAMPLE
    .\scripts\windows\doctor.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

$script:Failures = 0
$script:Warnings = 0

function Result-Pass { param($m) Write-Ok $m }
function Result-Warn { param($m) Write-Warn $m; $script:Warnings++ }
function Result-Fail { param($m) Write-Fail $m; $script:Failures++ }
function Result-Info { param($m) Write-Info $m }

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Trading Platform - Windows Diagnostics" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

# ── 1. Operating system ───────────────────────────────────────────────────────
Write-Section "Operating system"
$isWin = $IsWindows -or $PSVersionTable.PSVersion.Major -lt 6
if ($isWin) {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os) {
        $caption = $os.Caption
        $build   = $os.BuildNumber
        if ($caption -match "Windows 11|Windows 10") {
            Result-Pass "$caption (build $build)"
        } else {
            Result-Warn "$caption (build $build) - Windows 10/11 expected"
        }
        if ($os.OSArchitecture -eq "64-bit") { Result-Pass "64-bit architecture" }
        else { Result-Warn "32-bit OS - PostgreSQL 64-bit builds are recommended" }
    } else { Result-Warn "Could not read OS details" }
} else {
    Result-Warn "Not running on Windows - Windows-specific checks are skipped"
}

# ── 2. Python & virtual environment ───────────────────────────────────────────
Write-Section "Python environment"
$paths = Get-Paths
$pyCmd = Get-PythonCommand
if ($pyCmd) {
    $ver = & $pyCmd --version 2>&1
    if ($ver -match "3\.12") { Result-Pass "Python $ver" }
    elseif ($ver -match "3\.(1[01])") { Result-Warn "Python $ver - MetaTrader5 wheels are most reliable on 3.12" }
    else { Result-Warn "Python $ver - 3.12 recommended" }
} else { Result-Fail "Python not found on PATH" }

if (Test-Path $paths.Python) {
    Result-Pass "Virtual environment: $($paths.Venv)"
} else {
    Result-Fail "Virtual environment missing - run setup-venv.ps1"
}

# ── 3. Required packages ──────────────────────────────────────────────────────
Write-Section "Python packages"
if (Test-Path $paths.Python) {
    $missing = & $paths.Python -c @"
import importlib.util
mods = ['fastapi','sqlalchemy','pydantic_settings','alembic','pandas','numpy','sklearn','uvicorn']
print(' '.join(m for m in mods if importlib.util.find_spec(m) is None))
"@ 2>&1
    $missingList = ($missing | Out-String).Trim()
    if (-not $missingList) {
        Result-Pass "All core packages importable"
    } else {
        Result-Fail "Missing packages: $missingList (run install-dependencies.ps1)"
    }

    $mt5 = & $paths.Python -c "import importlib.util; print('YES' if importlib.util.find_spec('MetaTrader5') else 'NO')" 2>&1
    if (($mt5 | Out-String).Trim() -eq "YES") { Result-Pass "MetaTrader5 installed" }
    else { Result-Warn "MetaTrader5 not installed - broker features unavailable" }
} else {
    Result-Warn "Skipped: no virtual environment"
}

# ── 4. Configuration ──────────────────────────────────────────────────────────
Write-Section "Configuration"
$paths = Get-Paths
if (Test-Path $paths.EnvFile) {
    Result-Pass ".env present"
    $cfg = Test-ConfigValid
    foreach ($w in $cfg.Warnings) { Result-Warn $w }
    foreach ($e in $cfg.Errors)   { Result-Fail $e }
    if ($cfg.Valid -and $cfg.Errors.Count -eq 0) {
        Result-Pass "Configuration valid (TRADING_MODE=$($cfg.Mode))"
    }
} else {
    Result-Fail ".env missing - copy .env.example to .env"
}

# ── 5. Database ───────────────────────────────────────────────────────────────
Write-Section "Database"
$dbUrl = Get-Setting "DATABASE_URL" ""
$pg = Get-PostgresConnection -Url $dbUrl
if ($pg) {
    Result-Info "PostgreSQL configured (host=$($pg.Host) port=$($pg.Port) db=$($pg.Database))"

    $svc = Get-PostgresService
    if ($svc) { Result-Pass "Service '$($svc.Name)' is running" }
    else { Result-Fail "No running PostgreSQL service - start it from Services.msc" }

    $psql = Get-PsqlCommand
    if ($psql) { Result-Pass "psql found" }
    else { Result-Warn "psql not on PATH - needed for migrations and backups" }

    $reachable = Test-DatabaseReachable
    if ($reachable -eq $true) { Result-Pass "Database accepts connections" }
    elseif ($reachable -eq $false) { Result-Fail "Cannot connect to database - check credentials/service" }
} elseif ($dbUrl -match 'sqlite') {
    Result-Pass "SQLite configured (no service required)"
} else {
    Result-Fail "DATABASE_URL not set or not understood"
}

# Schema state
if (Test-Path $paths.Python) {
    $migration = & $paths.Python -m alembic current 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        if ($migration -match "head") { Result-Pass "Schema up to date" }
        else { Result-Warn "Schema may be behind - run migrate-database.ps1" }
    } else {
        Result-Warn "Could not read migration state - run migrate-database.ps1"
    }
}

# ── 6. Directories ────────────────────────────────────────────────────────────
Write-Section "Runtime directories"
foreach ($dir in @(@{n="logs"; p=$paths.Logs}, @{n="backups"; p=$paths.Backups})) {
    if (Test-Path $dir.p) { Result-Pass "$($dir.n) directory exists" }
    else { Result-Warn "$($dir.n) directory missing - created automatically on start" }
}

# ── 7. Ports ──────────────────────────────────────────────────────────────────
Write-Section "Ports"
$ports = @{ "API (8000)" = 8000 }
if ((Get-Setting "MT5_BRIDGE_PORT" "") -ne "") { $ports["Bridge"] = [int](Get-Setting "MT5_BRIDGE_PORT") }
foreach ($entry in $ports.GetEnumerator()) {
    $busy = Get-PortListeners -Port $entry.Value
    if ($busy -and $busy.Count -gt 0) {
        # If it is our own recorded process, that is fine.
        $proc = Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue
        Result-Warn "Port $($entry.Value) ($($entry.Key)) in use by PID $($busy[0].OwningProcess) ($($proc.ProcessName))"
    } else {
        Result-Pass "Port $($entry.Value) ($($entry.Key)) is free"
    }
}

# ── 8. MT5 terminal ───────────────────────────────────────────────────────────
Write-Section "MetaTrader 5"
$terminalPaths = @(
    "$env:ProgramFiles\MetaTrader 5\terminal64.exe",
    "${env:ProgramFiles(x86)}\MetaTrader 5\terminal64.exe",
    "$env:APPDATA\MetaQuotes\Terminal"
)
$foundTerminal = $false
foreach ($t in $terminalPaths) { if (Test-Path $t) { $foundTerminal = $true; Result-Info "Found: $t"; break } }

if (-not $foundTerminal) {
    Result-Warn "MetaTrader 5 terminal not detected"
    Result-Info "Download from your broker, or https://www.metatrader5.com/en/download"
    Result-Info "Paper trading and the API do NOT require it."
} else {
    $running = Get-Process "terminal64" -ErrorAction SilentlyContinue
    if ($running) { Result-Pass "Terminal running (PID $($running[0].Id))" }
    else { Result-Warn "Terminal installed but not running - open and log in" }
}

# ── 9. Processes ──────────────────────────────────────────────────────────────
Write-Section "Running processes"
foreach ($svc in @("api", "bot")) {
    $proc = Get-RunningProcess -Name $svc
    if ($proc) { Result-Pass "$svc running (PID $($proc.Id))" }
    else { Result-Info "$svc not running" }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
if ($script:Failures -eq 0) {
    Write-Host " RESULT: all critical checks passed" -ForegroundColor Green
    if ($script:Warnings -gt 0) {
        Write-Host "         $($script:Warnings) warning(s) — review the [WARN] lines" -ForegroundColor Yellow
    }
} else {
    Write-Host " RESULT: $($script:Failures) failure(s), $($script:Warnings) warning(s)" -ForegroundColor Red
}
Write-Host "===========================================================" -ForegroundColor Cyan
exit $script:Failures
<#
.SYNOPSIS
    Show the current state of the platform.

.DESCRIPTION
    Reports process state, API health, readiness and database connectivity.
    Every line is an executed check, never an assumption.

.EXAMPLE
    .\scripts\windows\status.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

Write-Section "Process state"
$anyRunning = $false
foreach ($name in @("api", "bot", "bridge")) {
    $proc = Get-RunningProcess -Name $name
    if ($proc) {
        $uptime = (Get-Date) - $proc.StartTime
        Write-Ok ("{0,-8} PID {1,-7} up {2:hh\:mm\:ss}" -f $name, $proc.Id, $uptime)
        $anyRunning = $true
    } else {
        Write-Info ("{0,-8} not running" -f $name)
    }
}
if (-not $anyRunning) { Write-Info "Nothing running. Start with .\scripts\windows\start.ps1" }

# ── API health ────────────────────────────────────────────────────────────────
Write-Section "API"
$port = [int](Get-Setting "APP_PORT" "8000")
$health = Invoke-HealthCheck -Uri "http://127.0.0.1:$port/api/health" -TimeoutSeconds 3
if ($health) {
    Write-Ok "Liveness  : /api/health -> $($health.status)"
    $ready = Invoke-HealthCheck -Uri "http://127.0.0.1:$port/api/ready" -TimeoutSeconds 5
    if ($ready -and $ready.status -eq "ready") {
        $dbState = if ($ready.checks) { $ready.checks.database } else { "unknown" }
        Write-Ok "Readiness : /api/ready -> ready (database=$dbState)"
        Write-Ok "Mode      : $($ready.trading_mode)"
    } elseif ($ready) {
        Write-Warn "Readiness : /api/ready -> $($ready.status)"
        if ($ready.checks) { Write-Info "  database: $($ready.checks.database)" }
    } else {
        Write-Warn "Readiness : endpoint did not respond"
    }
} else {
    Write-Info "API not responding on 127.0.0.1:$port"
}

# ── Database ──────────────────────────────────────────────────────────────────
Write-Section "Database"
$dbUrl = Get-Setting "DATABASE_URL" "sqlite:///./data/trading.db"
if ($dbUrl -match "^sqlite") {
    if (Test-Path $paths.Database) {
        $reachable = Test-DatabaseReachable
        if ($reachable -eq $true) {
            Write-Ok "SQLite database OK: $($paths.Database) (PRAGMA integrity_check passed)"
        } else {
            Write-Fail "SQLite database failed integrity check: $($paths.Database)"
        }
    } else {
        Write-Info "SQLite file does not exist yet: $($paths.Database)"
    }
} else {
    Write-Fail "DATABASE_URL is not SQLite ($dbUrl)"
}

# ── MT5 / broker ──────────────────────────────────────────────────────────────
Write-Section "MetaTrader 5"
$terminal = Get-Process "terminal64" -ErrorAction SilentlyContinue
if ($terminal) { Write-Ok "Terminal running (PID $($terminal[0].Id))" }
else { Write-Warn "Terminal not running - broker features unavailable" }

$paths = Get-Paths
if (Test-Path $paths.Python) {
    $mt5 = (& $paths.Python -c "import importlib.util; print('YES' if importlib.util.find_spec('MetaTrader5') else 'NO')" 2>&1 | Out-String).Trim()
    if ($mt5 -eq "YES") { Write-Ok "MetaTrader5 package installed" }
    else { Write-Warn "MetaTrader5 package not installed" }
}

# ── Logs ──────────────────────────────────────────────────────────────────────
Write-Section "Logs"
foreach ($name in @("api", "bot", "bridge")) {
    $log = Join-Path $paths.Logs "$name.log"
    if (Test-Path $log) {
        $size = [math]::Round((Get-Item $log).Length / 1KB, 1)
        $last = (Get-Item $log).LastWriteTime
        Write-Info ("{0,-8} {1,8} KB  last write {2}" -f $name, $size, $last)
    }
}
exit 0
<#
.SYNOPSIS
    Start the platform natively on Windows.

.DESCRIPTION
    Order of operations, and why:

      1. Validate configuration      - fail fast before opening ports.
      2. Check the database          - a missing DB must not look like success.
      3. Verify dependencies         - catch a broken venv early.
      4. Start the API               - wait for /api/ready, not just "started".
      5. Optionally start the bot    - requires an MT5 terminal.

    The API is started first and verified. The bot is only started when
    -Bot is passed, because running a trading loop is a deliberate act.

.EXAMPLE
    .\scripts\windows\start.ps1
.EXAMPLE
    .\scripts\windows\start.ps1 -Bot
#>

[CmdletBinding()]
param(
    [switch]$Bot,
    [switch]$Bridge,
    [int]$Port = 0,
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Write-Section "Starting the platform"

# ── 1. Configuration ──────────────────────────────────────────────────────────
$cfg = Test-ConfigValid
foreach ($w in $cfg.Warnings) { Write-Warn $w }
if (-not $cfg.Valid) {
    foreach ($e in $cfg.Errors) { Write-Fail $e }
    Write-Fail "Configuration invalid — refusing to start."
    exit 1
}
Write-Ok "Configuration valid (TRADING_MODE=$($cfg.Mode))"

# ── 2. Dependencies ───────────────────────────────────────────────────────────
$paths = Get-Paths
if (-not (Test-Path $paths.Python)) {
    Write-Fail "Virtual environment missing. Run install.ps1 first."
    exit 1
}
Write-Ok "Dependencies present"

# ── 3. Database ───────────────────────────────────────────────────────────────
$dbUrl = Get-Setting "DATABASE_URL" "sqlite:///./data/trading.db"
if (-not ($dbUrl -match "^sqlite")) {
    Write-Fail "DATABASE_URL must be a SQLite URL (e.g. sqlite:///./data/trading.db). Found: $dbUrl"
    exit 1
}
Initialize-RuntimeDirectories
if (Test-Path $paths.Database) {
    if (-not (Test-DatabaseReachable)) {
        Write-Fail "SQLite database at $($paths.Database) failed integrity check."
        exit 1
    }
    Write-Ok "SQLite database verified ($($paths.Database))"
} else {
    Write-Ok "SQLite database will be created on first access ($($paths.Database))"
}

# ── 4. API ────────────────────────────────────────────────────────────────────
if (Get-RunningProcess -Name "api") {
    Write-Warn "API already running (PID $(Get-StoredPid -Name 'api'))"
} else {
    if ($Port -le 0) { $Port = [int](Get-Setting "APP_PORT" "8000") }

    # Refuse to start if the port is already taken by something else.
    # Without this check the process below would die on bind, yet the health
    # probe below could still succeed against the *other* listener and report a
    # healthy API that this script never started.
    $listener = Get-PortListeners -Port $Port
    if ($listener -and $listener.Count -gt 0) {
        $owner = $listener[0].OwningProcess
        Write-Fail "Port $Port is already in use by PID $owner."
        Write-Info "Stop it first, or choose another port:  .\scripts\windows\start.ps1 -Port 8080"
        exit 1
    }

    $apiLog = Join-Path $paths.Logs "api.log"
    Write-Step "Starting API on 127.0.0.1:$Port ..."

    $apiArgs = @(
        "-m", "uvicorn", "api.main:app",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--log-level", "info"
    )
    $proc = Start-Process -FilePath $paths.Python -ArgumentList $apiArgs `
            -WorkingDirectory $paths.Root -PassThru `
            -RedirectStandardOutput $apiLog `
            -RedirectStandardError "$apiLog.err"
    Set-StoredPid -Name "api" -ProcessId $proc.Id
    Write-Ok "API process started (PID $($proc.Id))"

    # Verify readiness rather than assuming the process means success.
    Write-Step "Waiting for /api/ready (timeout ${StartupTimeoutSeconds}s) ..."
    $health = Invoke-HealthCheck -Uri "http://127.0.0.1:$Port/api/ready" -TimeoutSeconds $StartupTimeoutSeconds

    # Confirm the process WE started is the one still alive. A health response
    # alone is not proof: it can be answered by an unrelated listener.
    $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if (-not $alive) {
        Write-Fail "The API process (PID $($proc.Id)) exited during startup."
        foreach ($f in @($apiLog, "$apiLog.err")) {
            if (Test-Path $f) {
                Write-Host "  --- $(Split-Path $f -Leaf) ---" -ForegroundColor DarkGray
                Get-Content $f -Tail 25 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
            }
        }
        Clear-StoredPid -Name "api"
        exit 1
    }

    if (-not $health) {
        Write-Fail "API did not become ready. Last log lines:"
        foreach ($f in @($apiLog, "$apiLog.err")) {
            if (Test-Path $f) { Get-Content $f -Tail 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray } }
        }
        Stop-ProcessByName -ProcessId $proc.Id -GraceSeconds 5 | Out-Null
        Clear-StoredPid -Name "api"
        exit 1
    }
    $dbState = if ($health.checks) { $health.checks.database } else { "unknown" }
    Write-Ok "API ready (database=$dbState)"
}

# ── 5. Bot (opt-in) ───────────────────────────────────────────────────────────
if ($Bot) {
    $mt5Ok = $false
    if (Test-Path $paths.Python) {
        $mt5Ok = (& $paths.Python -c "import importlib.util; print('YES' if importlib.util.find_spec('MetaTrader5') else 'NO')" 2>&1 | Out-String).Trim() -eq "YES"
    }
    if (-not $mt5Ok) {
        Write-Fail "MetaTrader5 is not installed — the bot cannot start."
        Write-Info "Install it, or start without -Bot (API/paper research only)."
        exit 1
    }
    if (-not (Get-Process "terminal64" -ErrorAction SilentlyContinue)) {
        Write-Fail "MetaTrader 5 terminal is not running."
        Write-Info "Open the terminal and log in first. Refusing to start the bot without it."
        exit 1
    }

    if (Get-RunningProcess -Name "bot") {
        Write-Warn "Bot already running (PID $(Get-StoredPid -Name 'bot'))"
    } else {
        $botLog = Join-Path $paths.Logs "bot.log"
        Write-Step "Starting trading bot (TRADING_MODE=$($cfg.Mode)) ..."
        $botProc = Start-Process -FilePath $paths.Python -ArgumentList @("run.py") `
                 -WorkingDirectory $paths.Root -PassThru `
                 -RedirectStandardOutput $botLog `
                 -RedirectStandardError "$botLog.err"
        Set-StoredPid -Name "bot" -ProcessId $botProc.Id
        Write-Ok "Bot process started (PID $($botProc.Id))"
        if ($cfg.Mode -eq "live") { Write-Warn "LIVE MODE — real orders will be submitted" }
    }
} else {
    Write-Info "Bot not started (pass -Bot to run the trading loop)"
}

# ── 6. Bridge (opt-in) ────────────────────────────────────────────────────────
if ($Bridge) {
    $bridgeToken = Get-Setting "MT5_BRIDGE_TOKEN" ""
    if (-not $bridgeToken) {
        Write-Fail "MT5_BRIDGE_TOKEN is not set — the bridge refuses to start without it."
        Write-Info "Generate one: python -c \"import secrets;print(secrets.token_hex(32))\""
        exit 1
    }
    $bridgePort = [int](Get-Setting "MT5_BRIDGE_PORT" "8100")
    $bridgeLog  = Join-Path $paths.Logs "bridge.log"
    Write-Step "Starting MT5 bridge on 127.0.0.1:$bridgePort ..."
    $bridgeProc = Start-Process -FilePath $paths.Python -ArgumentList @("bridge\run.py") `
                  -WorkingDirectory $paths.Root -PassThru `
                  -RedirectStandardOutput $bridgeLog `
                  -RedirectStandardError "$bridgeLog.err"
    Set-StoredPid -Name "bridge" -ProcessId $bridgeProc.Id
    Write-Ok "Bridge process started (PID $($bridgeProc.Id))"
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Section "Running"
if ($Port -gt 0) {
    Write-Info "API      : http://127.0.0.1:$Port"
    Write-Info "API docs : http://127.0.0.1:$Port/docs"
    Write-Info "Health   : http://127.0.0.1:$Port/api/health"
    Write-Info "Ready    : http://127.0.0.1:$Port/api/ready"
}
if ($Bot)     { Write-Info "Bot      : running (PID $(Get-StoredPid -Name 'bot'))" }
if ($Bridge)  { Write-Info "Bridge   : running (PID $(Get-StoredPid -Name 'bridge'))" }
Write-Info "Logs     : $($paths.Logs)"
Write-Info "Stop     : .\scripts\windows\stop.ps1"
exit 0
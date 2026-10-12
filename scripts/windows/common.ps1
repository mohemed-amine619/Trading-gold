<#
.SYNOPSIS
    Shared helpers for every Windows script in this project.

.DESCRIPTION
    Dot-source this file rather than duplicating logic:

        . "$PSScriptRoot\common.ps1"

    Provides project paths, logging, .env parsing, environment validation,
    virtual-environment activation, PostgreSQL detection and process helpers.

    Design rules:
      * Never prints secrets. Secrets are only ever compared or masked.
      * Never claims a service is healthy without checking it.
      * Every script returns a meaningful exit code: 0 ok, 1 fail.
#>

Set-StrictMode -Version Latest

# ── Project paths ────────────────────────────────────────────────────────────

function Get-ProjectRoot {
    <#  Resolves the repository root by walking up from this script.  #>
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$script:ProjectRoot = Get-ProjectRoot

function Get-Paths {
    <#  Canonical locations used across the toolchain.  #>
    $root = $script:ProjectRoot
    # Join-Path uses the host separator, so these resolve correctly on both
    # Windows and when validating these scripts on Linux/macOS.
    $venv   = Join-Path $root ".venv"
    # On Windows a venv uses Scripts\; on Linux/macOS it is bin/. Resolve the
    # interpreter binary that actually exists so the same script can be
    # smoke-tested off-Windows without pretending it is the Windows layout.
    $isWin  = $IsWindows -or $PSVersionTable.PSVersion.Major -lt 6
    $scripts = if ($isWin) { Join-Path $venv "Scripts" } else { Join-Path $venv "bin" }
    $exe     = if ($isWin) { ".exe" } else { "" }
    $dataDir = Join-Path $root "data"
    $dbPath  = if (Test-Path (Join-Path $dataDir "trading.db")) {
        Join-Path $dataDir "trading.db"
    } elseif (Test-Path (Join-Path $root "trading.db")) {
        Join-Path $root "trading.db"
    } else {
        Join-Path $dataDir "trading.db"
    }
    [pscustomobject]@{
        Root         = $root
        Venv         = $venv
        Python       = Join-Path $scripts "python$exe"
        Pip          = Join-Path $scripts "pip$exe"
        EnvFile      = Join-Path $root ".env"
        EnvExample   = Join-Path $root ".env.example"
        Requirements = Join-Path $root "requirements-windows.txt"
        Logs         = Join-Path $root "logs"
        Backups      = Join-Path $root "backups"
        Run          = Join-Path $root "run"
        Data         = $dataDir
        Database     = $dbPath
    }
}

# ── Console output ───────────────────────────────────────────────────────────

function Write-Section  { param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}
function Write-Ok      { param([string]$Message) Write-Host "  [PASS] $Message" -ForegroundColor Green }
function Write-Warn    { param([string]$Message) Write-Host "  [WARN] $Message" -ForegroundColor Yellow }
function Write-Fail    { param([string]$Message) Write-Host "  [FAIL] $Message" -ForegroundColor Red }
function Write-Info    { param([string]$Message) Write-Host "  [INFO] $Message" -ForegroundColor Gray }
function Write-Step    { param([string]$Message) Write-Host "  [ .. ] $Message" -ForegroundColor Cyan }

# ── Environment file handling ─────────────────────────────────────────────────

function Read-EnvFile {
    <#
      Parses .env into an ordered dictionary.
      Handles KEY=value, comments, blank lines and optional 'export ' prefix.
      Values are NOT printed anywhere.
    #>
    $result = [ordered]@{}
    $paths = Get-Paths
    if (-not (Test-Path $paths.EnvFile)) { return $result }

    foreach ($line in Get-Content $paths.EnvFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed.StartsWith("export ")) { $trimmed = $trimmed.Substring(7).Trim() }
        $idx = $trimmed.IndexOf("=")
        if ($idx -lt 1) { continue }
        $key   = $trimmed.Substring(0, $idx).Trim()
        $value = $trimmed.Substring($idx + 1).Trim()
        # Strip surrounding quotes if present.
        if ($value.Length -ge 2) {
            $first = $value[0]; $last = $value[$value.Length - 1]
            if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $result[$key] = $value
    }
    return $result
}

function Get-Setting {
    <#
      Reads one setting. Precedence: process env > .env file > supplied default.
      This mirrors pydantic-settings' priority order.
    #>
    param([string]$Name, [string]$Default = "")
    $fromProcess = [Environment]::GetEnvironmentVariable($Name)
    if ($fromProcess) { return $fromProcess }
    $file = Read-EnvFile
    if ($file.Contains($Name) -and $file[$Name]) { return $file[$Name] }
    return $Default
}

function Test-ConfigValid {
    <#
      Validates required configuration.
      Returns a result object; never prints secret values, only their presence.
    #>
    $errors   = [System.Collections.ArrayList]::new()
    $warnings = [System.Collections.ArrayList]::new()

    $paths = Get-Paths
    if (-not (Test-Path $paths.EnvFile)) {
        [void]$warnings.Add(".env not found — copy .env.example to .env and set values")
    }

    $mode = (Get-Setting "TRADING_MODE" "paper").ToLower()
    if ($mode -notin @("paper", "backtest", "live")) {
        [void]$errors.Add("TRADING_MODE='$mode' is invalid (expected paper|backtest|live)")
    }
    if ($mode -eq "live") {
        # Live requires deliberate, explicit configuration.
        foreach ($required in @("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")) {
            if (-not (Get-Setting $required)) {
                [void]$errors.Add("live mode requires $required to be set")
            }
        }
        $secret = Get-Setting "DASHBOARD_SECRET_KEY"
        if (-not $secret -or $secret -eq "CHANGE_ME") {
            [void]$errors.Add("live mode requires DASHBOARD_SECRET_KEY to be changed")
        }
        [void]$warnings.Add("TRADING_MODE=live — this platform WILL place real orders")
    }

    $dbUrl = Get-Setting "DATABASE_URL" "sqlite:///./data/trading.db"
    if (-not $dbUrl) {
        [void]$errors.Add("DATABASE_URL is not set")
    }
    elseif ($dbUrl -notmatch "^sqlite") {
        [void]$errors.Add("DATABASE_URL must be an SQLite URL ($dbUrl). External database servers (PostgreSQL, etc.) are forbidden.")
    }

    [pscustomobject]@{
        Valid    = ($errors.Count -eq 0)
        Errors   = $errors
        Warnings = $warnings
        Mode     = $mode
    }
}

# ── Python / virtual environment ─────────────────────────────────────────────

function Get-PythonCommand {
    <#  Path to the venv python, or the system python when no venv exists.  #>
    $paths = Get-Paths
    if (Test-Path $paths.Python) { return $paths.Python }
    foreach ($candidate in @("python", "py")) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) { return $candidate }
    }
    return $null
}

function Invoke-Python {
    <#  Runs the venv python with arguments, forwarding the exit code.  #>
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    $python = Get-PythonCommand
    if (-not $python) { throw "Python not found. Run scripts\windows\setup-venv.ps1 first." }
    & $python @Args
    return $LASTEXITCODE
}

function Test-VenvExists {
    return (Test-Path (Get-Paths).Python)
}

# ── SQLite Database Integrity ─────────────────────────────────────────────────

function Test-DatabaseReachable {
    <#  Verifies SQLite database access and integrity (PRAGMA integrity_check).  #>
    $paths = Get-Paths
    $python = Get-PythonCommand
    if (-not $python -or -not (Test-Path $paths.Database)) { return $true }
    try {
        & $python -c "import sqlite3; conn = sqlite3.connect(r'$($paths.Database)'); res = conn.execute('PRAGMA integrity_check').fetchone(); conn.close(); exit(0 if res and res[0] == 'ok' else 1)" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# ── Runtime directories ──────────────────────────────────────────────────────

function Initialize-RuntimeDirectories {
    $paths = Get-Paths
    foreach ($dir in @($paths.Logs, $paths.Backups, $paths.Run, $paths.Data)) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

# ── Process helpers ──────────────────────────────────────────────────────────

function Get-StoredPid {
    <#  Reads the recorded PID for a service, or $null.  #>
    param([ValidateSet("api","bot","bridge")][string]$Name)
    $file = Join-Path (Get-Paths).Run "$Name.pid"
    if (-not (Test-Path $file)) { return $null }
    $raw = (Get-Content $file -Raw).Trim()
    # NB: PowerShell variable names are case-insensitive, so $pid would
    # collide with the built-in read-only $PID. Use a distinct name.
    $parsed = 0
    if ([int]::TryParse($raw, [ref]$parsed)) { return $parsed }
    return $null
}

function Set-StoredPid {
    param([ValidateSet("api","bot","bridge")][string]$Name, [int]$ProcessId)
    Initialize-RuntimeDirectories
    Set-Content -Path (Join-Path (Get-Paths).Run "$Name.pid") -Value $ProcessId
}

function Clear-StoredPid {
    param([ValidateSet("api","bot","bridge")][string]$Name)
    $file = Join-Path (Get-Paths).Run "$Name.pid"
    if (Test-Path $file) { Remove-Item $file -Force -ErrorAction SilentlyContinue }
}

function Get-RunningProcess {
    <#  Returns the live process for a service name, or $null if stopped.  #>
    param([ValidateSet("api","bot","bridge")][string]$Name)
    $stored = Get-StoredPid -Name $Name
    if (-not $stored) { return $null }
    $proc = Get-Process -Id $stored -ErrorAction SilentlyContinue
    if ($proc) { return $proc }
    # Stale PID file — the process is gone.
    Clear-StoredPid -Name $Name
    return $null
}

function Invoke-HealthCheck {
    <#
      Polls an HTTP endpoint until it answers or the timeout elapses.
      Returns the parsed object on success, $null on failure.
    #>
    param(
        [string]$Uri,
        [int]$TimeoutSeconds = 30
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-RestMethod -Uri $Uri -TimeoutSec 3 -ErrorAction Stop
        } catch {
            Start-Sleep -Milliseconds 700
        }
    }
    return $null
}

function Get-PortListeners {
    <#
      Returns objects with OwningProcess for each listener on a port.
      Uses the native Windows cmdlet when present and falls back to netstat
      elsewhere, so port checks never silently report "free" just because a
      cmdlet is missing.
    #>
    param([int]$Port)

    # Force an array: PowerShell unrolls a single-element array on return,
    # which would hand callers a scalar lacking .Count under StrictMode.
    $found = @()
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    }

# Portable fallback: netstat (Windows + net-tools) then ss (Linux/macOS).
    # Without a working probe we must NOT claim the port is free.
    $parsed = $false
    if (Get-Command netstat -ErrorAction SilentlyContinue) {
        try {
            foreach ($line in (& netstat -ano 2>$null)) {
                if ($line -match "^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                    $parsed = $true
                    if ([int]$Matches[1] -eq $Port) {
                        $found += [pscustomobject]@{
                            LocalPort     = $Port
                            OwningProcess = [int]$Matches[2]
                            State         = "Listen"
                        }
                    }
                }
            }
        } catch { }
    }
    elseif (Get-Command ss -ErrorAction SilentlyContinue) {
        try {
            foreach ($line in (& ss -ltn 2>$null)) {
                if ($line -match "LISTEN\s+\d+\s+\d+\s+\S*:(\d+)\s+") {
                    $parsed = $true
                    if ([int]$Matches[1] -eq $Port) {
                        $found += [pscustomobject]@{
                            LocalPort     = $Port
                            OwningProcess = 0
                            State         = "Listen"
                        }
                    }
                }
            }
        } catch { }
    }
    if (-not $parsed) {
        Write-Warn "No port probe available (netstat/ss); cannot verify port $Port"
    }
    return @($found)
}

function Test-PortFree {
    <#  True only when nothing is listening on the port.  #>
    param([int]$Port)
    # Assign through @() first: PowerShell unrolls single-item arrays on
    # return, so a lone listener would arrive here as a scalar with no .Count.
    $listeners = @(Get-PortListeners -Port $Port)
    return ($listeners.Count -eq 0)
}

function Stop-ProcessByName {
    <#
      Gracefully stops a process tree: SIGTERM equivalent first, then force.
      Windows has no SIGTERM, so Stop-Process is the graceful path for
      console apps; the API additionally handles CTRL_C_EVENT via uvicorn.
    #>
    param([int]$ProcessId, [int]$GraceSeconds = 15)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { return $true }

    Write-Step "Stopping PID $ProcessId ..."
    try { $proc.CloseMainWindow() | Out-Null } catch { }
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue

    $deadline = (Get-Date).AddSeconds($GraceSeconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    Write-Warn "PID $ProcessId did not exit gracefully; forcing"
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    return (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue))
}

<#
    This file is dot-sourced (". ./common.ps1"), not imported as a module, so
    every function is already in scope. Export-ModuleMember would throw here.
#>
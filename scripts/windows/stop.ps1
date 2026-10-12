<#
.SYNOPSIS
    Stop the platform's processes.

.DESCRIPTION
    Stops API, bot and bridge processes tracked in run\*.pid. Graceful first,
    forced only if a process refuses to exit.
    SQLite is embedded and stops when the processes exit.

.EXAMPLE
    .\scripts\windows\stop.ps1
.EXAMPLE
    .\scripts\windows\stop.ps1 -WhatIf    # preview only
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet("api","bot","bridge","all")]
    [string]$Target = "all",
    [int]$GraceSeconds = 15
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

Write-Section "Stopping the platform"

$targets = if ($Target -eq "all") { @("api", "bot", "bridge") } else { @($Target) }

foreach ($name in $targets) {
    $proc = Get-RunningProcess -Name $name
    if (-not $proc) {
        Write-Info "$name is not running"
        continue
    }

    if (-not $PSCmdlet.ShouldProcess("Process $($proc.Id)", "Stop $name")) {
        Write-Info "Would stop $name (Process $($proc.Id))"
        continue
    }

    $stopped = Stop-ProcessByName -ProcessId $proc.Id -GraceSeconds $GraceSeconds
    Clear-StoredPid -Name $name
    if ($stopped) { Write-Ok "$name stopped" }
    else { Write-Fail "$name could not be stopped (PID $($proc.Id))" }
}

exit 0
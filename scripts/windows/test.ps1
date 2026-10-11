<#
.SYNOPSIS
    Run the test suite.

.DESCRIPTION
    Always runs `pytest tests\` with an explicit path. Never bare `pytest`:
    root-level scripts matching test_*.py are collected and can have import
    side effects.

.EXAMPLE
    .\scripts\windows\test.ps1
.EXAMPLE
    .\scripts\windows\test.ps1 -Keyword "risk"
#>

[CmdletBinding()]
param(
    [string]$Keyword = "",
    [string]$Path = "tests",
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"

Write-Section "Tests"

$paths = Get-Paths
if (-not (Test-Path $paths.Python)) {
    Write-Fail "Virtual environment missing. Run install.ps1 first."
    exit 1
}

$args = @("-m", "pytest", $Path, "-q")
if ($Keyword)         { $args += @("-k", $Keyword) }
if ($VerboseOutput)   { $args += @("-v") } else { $args += @("--no-header") }

Write-Info "Command: python $($args -join ' ')"
Write-Step "Running ..."
& $paths.Python @args
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Ok "All tests passed"
} else {
    Write-Fail "Tests failed (exit $code)"
}
exit $code
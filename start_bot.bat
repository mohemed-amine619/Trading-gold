@echo off
title AI Gold Scalper - Auto Start
cd C:\Users\mohamed.bougrioua\Desktop\Trading-gold

:: Display startup message
echo.
echo ==========================================
echo AI GOLD SCALPER - BOT AUTO-START
echo ==========================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Check if bot is already running
tasklist /fi "imagename eq python.exe" 2>nul | find "python.exe" >nul
if %errorlevel% equ 0 (
    echo ⚠ AI Gold Scalper is ALREADY RUNNING
    echo Check bot_output.log for current status
    echo.
    timeout /t 5 >nul
    exit /b
)

:: Start the bot with logging
echo Starting AI Gold Scalper Bot...
echo Log file: bot_output.log
echo.

:: Start bot in background
start /min /wait .venv\Scripts\python.exe run.py > bot_output.log 2>&1

:: Get the PID of the started process
for /f "tokens=2 delims= " %%A in ('tasklist /fi "imagename eq python.exe" ^| find "python.exe"') do (
    set PID=%%A
)

echo.
echo AI Gold Scalper Bot started with PID: %PID%
echo To stop: Open new CMD and taskkill /pid %PID% /f
echo.
echo "To view logs: type type bot_output.log"
echo.

:: Monitoring loop - check every 60 seconds
:MONITOR_LOOP
timeout /t 60 >nul
tasklist /fi "imagename eq python.exe" 2>nul | find "python.exe" >nul
if %errorlevel% equ 0 (
    echo [%TIME%] Bot is RUNNING - %PID%
) else (
    echo [%TIME%] Bot has STOPPED - Auto-restarting...
    goto START
)

goto MONITOR_LOOP

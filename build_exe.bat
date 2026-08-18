@echo off
REM ==========================================================================
REM Build the Windows .exe for the trading bot GUI.
REM
REM REQUIREMENTS
REM   * Run this on WINDOWS (MT5 is Windows-only, and PyInstaller cannot
REM     cross-compile a Windows exe from Linux).
REM   * Use 64-bit Python matching your MT5 terminal build:
REM       - 64-bit terminal  ->  64-bit Python (default on most installs)
REM     The MetaTrader5.dll must match the terminal's bitness.
REM   * The MT5 terminal should be CLOSED during the build.
REM
REM OUTPUT
REM   dist\GoldTradingBot.exe  - single-file, windowed (no console).
REM   Copy this exe anywhere; config.py values are compiled in, so edit
REM   config.py before building if you want different defaults.
REM ==========================================================================

echo [1/4] Upgrading pip...
py -3 -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [2/4] Installing dependencies...
py -3 -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo [3/4] Building the exe (this can take several minutes)...
py -3 -m PyInstaller trading_bot.spec --noconfirm
if errorlevel 1 goto :error

echo [4/4] Done.
echo.
echo Your executable is here:  dist\GoldTradingBot.exe
echo.
echo TIP: launch it with DRY_RUN enabled (default) first, watch the
echo      Signals tab, then switch to live trading from the Dashboard.
pause
exit /b 0

:error
echo.
echo BUILD FAILED. Scroll up to see the error.
pause
exit /b 1

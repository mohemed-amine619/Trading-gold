@echo off
REM ==========================================================================
REM  build_exe.bat  -  Build GoldTradingBot.exe (AI edition)
REM
REM  REQUIREMENTS
REM  ────────────
REM  • Run this script on WINDOWS (MT5 is Windows-only; PyInstaller cannot
REM    cross-compile a Windows exe from Linux or macOS).
REM  • Use 64-bit Python 3.9 – 3.12 matching your MT5 terminal build.
REM    Check: py -3 -c "import struct; print(struct.calcsize('P')*8)"
REM    → should print 64.
REM  • The MT5 terminal should be CLOSED during the build.
REM  • Internet access is required for the first run (pip downloads packages).
REM
REM  OUTPUT
REM  ──────
REM  dist\GoldTradingBot.exe   – single-file, windowed (no console).
REM
REM  USAGE
REM  ─────
REM  1. Edit config.py to set your default SYMBOLS, DRY_RUN, etc.
REM  2. Double-click this file (or run it in cmd.exe).
REM  3. Copy dist\GoldTradingBot.exe to any Windows machine that has MT5.
REM  4. Launch with DRY_RUN = True (default) first to verify the GUI works.
REM  5. Flip DRY_RUN = False in config.py and rebuild for live trading.
REM ==========================================================================

setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   AI Trading Bot  –  EXE Build Script           ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM --------------------------------------------------------------------------
REM Step 1 – Verify Python is 64-bit
REM --------------------------------------------------------------------------
echo [1/5] Checking Python architecture...
py -3 -c "import struct; bits=struct.calcsize('P')*8; print(f'Python {bits}-bit'); exit(0 if bits==64 else 1)"
if errorlevel 1 (
    echo.
    echo  ERROR: 32-bit Python detected.  Install 64-bit Python 3.9+.
    goto :error
)
echo  OK.
echo.

REM --------------------------------------------------------------------------
REM Step 2 – Upgrade pip and install all dependencies
REM --------------------------------------------------------------------------
echo [2/5] Upgrading pip...
py -3.12 -m pip install --upgrade pip --quiet
if errorlevel 1 goto :error

echo [3/5] Installing dependencies (this may take a few minutes)...
py -3 -m pip install --upgrade ^
    MetaTrader5 ^
    pandas ^
    pandas_ta ^
    numpy ^
    PyQt5 ^
    matplotlib ^
    xgboost ^
    requests ^
    scikit-learn ^
    scipy ^
    joblib ^
    pyinstaller ^
    --quiet
if errorlevel 1 goto :error
echo  All packages installed.
echo.

REM --------------------------------------------------------------------------
REM Step 3 – Clean previous build artefacts
REM --------------------------------------------------------------------------
echo [4/5] Cleaning old build artefacts...
if exist "build"            rmdir /s /q "build"
if exist "dist\GoldTradingBot.exe" del /q "dist\GoldTradingBot.exe"
echo  Clean.
echo.

REM --------------------------------------------------------------------------
REM Step 4 – Run PyInstaller
REM --------------------------------------------------------------------------
echo [5/5] Building GoldTradingBot.exe  (this takes 2-5 minutes)...
echo.
py -3 -m PyInstaller trading_bot.spec --noconfirm
if errorlevel 1 goto :error

REM --------------------------------------------------------------------------
REM Done
REM --------------------------------------------------------------------------
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   BUILD SUCCESSFUL                               ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  Your executable:   dist\GoldTradingBot.exe
echo.
echo  Tips
echo  ────
echo  • First launch:    keep DRY_RUN = True (default) and watch the
echo                     Signals tab to verify signals are firing.
echo  • Live trading:    set DRY_RUN = False in config.py and rebuild.
echo  • The exe is self-contained – copy it to any Windows PC with MT5.
echo  • If MT5 is not in the default path, set MT5_PATH in config.py.
echo.
pause
exit /b 0

:error
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   BUILD FAILED  –  scroll up to see the error   ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
exit /b 1

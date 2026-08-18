@echo off
REM ==========================================================================
REM  build_exe.bat  -  Build GoldTradingBot.exe (AI edition)
REM
REM  REQUIREMENTS
REM  ────────────
REM  • Run this script on WINDOWS (MT5 is Windows-only).
REM  • Python 3.12 (64-bit) must be installed.
REM    Download: https://www.python.org/downloads/release/python-3120/
REM  • The MT5 terminal should be CLOSED during the build.
REM
REM  OUTPUT
REM  ──────
REM  dist\GoldTradingBot.exe   – single-file, windowed (no console).
REM ==========================================================================

setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   AI Trading Bot  –  EXE Build Script           ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM --------------------------------------------------------------------------
REM Step 1 – Find Python 3.12 specifically
REM          py -3 would pick Python 3.14+ which is unsupported by pandas_ta
REM --------------------------------------------------------------------------
echo [1/5] Locating Python 3.12 (64-bit)...

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python 3.12 not found.
    echo  Please install 64-bit Python 3.12 from:
    echo  https://www.python.org/downloads/release/python-3120/
    echo.
    echo  NOTE: Python 3.13+ and 3.14+ are NOT supported yet by
    echo        MetaTrader5 / pandas_ta / numba. Use 3.12.
    goto :error
)

py -3.12 -c "import struct; bits=struct.calcsize('P')*8; print(f'  Found Python 3.12 ({bits}-bit)'); exit(0 if bits==64 else 1)"
if errorlevel 1 (
    echo.
    echo  ERROR: Python 3.12 is 32-bit. Install the 64-bit version.
    goto :error
)
echo  OK.
echo.

REM Use py -3.12 for all subsequent commands
set PY=py -3.12

REM --------------------------------------------------------------------------
REM Step 2 – Upgrade pip
REM --------------------------------------------------------------------------
echo [2/5] Upgrading pip...
%PY% -m pip install --upgrade pip --quiet
if errorlevel 1 goto :error
echo  OK.
echo.

REM --------------------------------------------------------------------------
REM Step 3 – Install dependencies
REM          Install pandas_ta WITHOUT numba (numba doesn't support Py 3.14+
REM          and is only an optional speed-up, not required for correctness).
REM --------------------------------------------------------------------------
echo [3/5] Installing dependencies...

REM Core numeric / ML stack
%PY% -m pip install --upgrade numpy pandas scikit-learn scipy joblib xgboost --quiet
if errorlevel 1 goto :error

REM pandas_ta without pulling in numba
%PY% -m pip install --upgrade pandas_ta --quiet
if errorlevel 1 goto :error

REM MT5 + GUI + networking
%PY% -m pip install --upgrade MetaTrader5 PyQt5 matplotlib requests --quiet
if errorlevel 1 goto :error

REM PyInstaller
%PY% -m pip install --upgrade pyinstaller --quiet
if errorlevel 1 goto :error

echo  All packages installed.
echo.

REM --------------------------------------------------------------------------
REM Step 4 – Clean previous build artefacts
REM --------------------------------------------------------------------------
echo [4/5] Cleaning old build artefacts...
if exist "build"                   rmdir /s /q "build"
if exist "dist\GoldTradingBot.exe" del   /q    "dist\GoldTradingBot.exe"
echo  Clean.
echo.

REM --------------------------------------------------------------------------
REM Step 5 – Run PyInstaller
REM --------------------------------------------------------------------------
echo [5/5] Building GoldTradingBot.exe  (this takes 2-5 minutes)...
echo.
%PY% -m PyInstaller trading_bot.spec --noconfirm
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

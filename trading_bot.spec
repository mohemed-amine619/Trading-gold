# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the trading bot GUI.

Build on WINDOWS with a 64-bit Python (must match your MT5 terminal build):
    build_exe.bat
or manually:
    pip install pyinstaller
    pyinstaller trading_bot.spec --noconfirm

Output: dist/GoldTradingBot.exe (single-file, windowed - no console).
"""
from PyInstaller.utils.hooks import collect_all

# collect_all() bundles data files, DLLs and hidden submodules for packages
# that load resources at runtime. MetaTrader5 needs its .dll; pandas/pandas_ta
# load data files and lazily-imported modules.
datas, binaries, hiddenimports = [], [], []
for pkg in ("MetaTrader5", "pandas", "numpy", "pandas_ta"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["gui_main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5.QtQml", "PyQt5.QtQuick", "PyQt5.QtMultimedia",
              "PyQt5.QtWebEngineWidgets", "matplotlib", "sklearn",
              "scipy", "xgboost", "winsound"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GoldTradingBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed app - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

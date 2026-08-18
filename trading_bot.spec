# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the AI Trading Bot GUI.

Build on WINDOWS with 64-bit Python (must match your MT5 terminal build):
    Double-click build_exe.bat
or manually:
    pip install pyinstaller
    pyinstaller trading_bot.spec --noconfirm

Output: dist\\GoldTradingBot.exe
  - Single-file, windowed (no console window).
  - Bundles: MetaTrader5, pandas, pandas_ta, scikit-learn, xgboost,
             scipy, PyQt5 and all core/ modules.

NOTE: Build ONLY on a Windows machine where MT5 is installed.
      The MT5 terminal should be CLOSED during the build.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# --------------------------------------------------------------------------
# Runtime data + DLL collection
# --------------------------------------------------------------------------
# collect_all() gathers:
#   datas      – data files the package ships (e.g. MetaTrader5.dll)
#   binaries   – native shared libraries
#   hiddenimports – lazily-imported sub-modules PyInstaller can't see statically
# --------------------------------------------------------------------------
datas, binaries, hiddenimports = [], [], []

COLLECT_PKGS = [
    "MetaTrader5",
    "pandas",
    "pandas_ta",
    "numpy",
    "sklearn",       # scikit-learn  (AI ensemble: RF, GBT, ExtraTrees)
    "scipy",         # pulled in by scikit-learn internals
    "xgboost",       # available in requirements; kept for future strategies
    "joblib",        # scikit-learn model persistence
    "threadpoolctl", # sklearn internal thread management
]

for pkg in COLLECT_PKGS:
    try:
        d, b, h = collect_all(pkg)
        datas    += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass   # package absent in build env – skip gracefully

# sklearn uses importlib to load estimators dynamically; enumerate them all.
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("pandas_ta")

# --------------------------------------------------------------------------
# Extra hidden imports that PyInstaller misses via static analysis
# --------------------------------------------------------------------------
hiddenimports += [
    # scikit-learn estimators used by AIEnsemble
    "sklearn.ensemble._forest",
    "sklearn.ensemble._gb",
    "sklearn.ensemble._voting",
    "sklearn.preprocessing._data",
    "sklearn.pipeline",
    # pandas_ta indicator modules loaded at runtime
    "pandas_ta.core",
    "pandas_ta.overlap",
    "pandas_ta.momentum",
    "pandas_ta.trend",
    "pandas_ta.volatility",
    "pandas_ta.volume",
    # project modules
    "core.ai_model",
    "core.strategy",
    "core.risk_manager",
    "core.bot",
    "core.data_handler",
    "core.execution",
    "core.mt5_engine",
    "core.state_manager",
    "core.alerts",
    "core.credentials_store",
    # stdlib / misc
    "requests",
    "logging",
    "asyncio",
    "concurrent.futures",
]

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
a = Analysis(
    ["gui_main.py"],            # GUI entry point
    pathex=["."],               # project root on PATH so `import core.*` works
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Qt modules not used
        "PyQt5.QtQml", "PyQt5.QtQuick", "PyQt5.QtMultimedia",
        "PyQt5.QtWebEngineWidgets", "PyQt5.QtBluetooth",
        # matplotlib not needed at runtime (GUI uses PyQt5 widgets directly)
        "matplotlib",
        # winsound is a built-in Windows module – always present on Windows
        "winsound",
        # test / dev tools
        "pytest", "IPython", "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# --------------------------------------------------------------------------
# Executable (single-file, no console)
# --------------------------------------------------------------------------
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
    upx=True,                  # compress with UPX to reduce file size
    upx_exclude=[
        # UPX can corrupt some DLLs – exclude them
        "vcruntime140.dll",
        "msvcp140.dll",
        "python3*.dll",
    ],
    runtime_tmpdir=None,
    console=False,             # windowed app – no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                 # set to "icon.ico" if you have one
)

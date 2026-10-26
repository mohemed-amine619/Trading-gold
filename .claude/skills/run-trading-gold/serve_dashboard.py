"""Serve run.py's real dashboard on Linux, paper mode only.

``python run.py`` needs a live MetaTrader5 terminal (Windows only). This starts
the same FastAPI dashboard that run.py builds, with MetaTrader5 stubbed out and
the bot in paper mode. ``.env`` is NOT read, so no broker credential or
DASHBOARD_TOKEN from the real environment reaches this process.

Usage (from the repo root):
    DASHBOARD_TOKEN=driver-token .venv/bin/python .claude/skills/run-trading-gold/serve_dashboard.py
Env: DASHBOARD_PORT (default 8765).
"""
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ["TRADING_MODE"] = "paper"

mt5 = types.ModuleType("MetaTrader5")
mt5.initialize = lambda *a, **k: False
# Constants (TIMEFRAME_M15, ORDER_TYPE_BUY, ...) are ints; calls such as
# positions_get() return "nothing" so the bot sees an empty broker.
mt5.__getattr__ = lambda name: 0 if name.isupper() or name.startswith("TIMEFRAME") else (lambda *a, **k: None)
sys.modules["MetaTrader5"] = mt5

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *a, **k: False
sys.modules["dotenv"] = dotenv

import uvicorn  # noqa: E402
import run  # noqa: E402

cfg = dict(run.CONFIG)
cfg["dry_run"] = True
bot = run.StandaloneAIBot(cfg)
app = run.create_app([bot])

port = int(os.environ.get("DASHBOARD_PORT", "8765"))
print(f"dashboard on http://127.0.0.1:{port}/ (paper mode, MT5 stubbed)", flush=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

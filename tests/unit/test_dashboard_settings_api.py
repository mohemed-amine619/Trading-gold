"""
tests/unit/test_dashboard_settings_api.py — Verification for Dashboard Settings & Gold Protection.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import numpy as np
import pandas as pd
import pytest


class _Info:
    def __init__(self, point=0.01):
        self.digits = 2
        self.point = point
        self.trade_tick_size = point
        self.trade_tick_value = 1.0
        self.volume_min = 0.01
        self.volume_max = 10.0
        self.volume_step = 0.01
        self.contract_size = 100.0
        self.trade_stops_level = 10
        self.spread = 10
        self.filling_mode = 2


class _Tick:
    def __init__(self, bid, ask):
        self.bid, self.ask = bid, ask


class _Acc:
    def __init__(self):
        self.balance = self.equity = 10_000.0
        self.margin = 0.0
        self.margin_free = 10_000.0


class StubMT5:
    def __init__(self):
        self.price = 2650.0
        self.TRADE_ACTION_DEAL = 1
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_IOC = 1
        self.TRADE_RETCODE_DONE = 10009
        self.TIMEFRAME_M1 = 1
        self.TIMEFRAME_M5 = 5
        self.TIMEFRAME_M15 = 15
        self.TIMEFRAME_H1 = 16385
        self.TIMEFRAME_H4 = 16388

    def symbol_info(self, symbol): return _Info()
    def symbol_info_tick(self, symbol): return _Tick(self.price - 0.1, self.price + 0.1)
    def account_info(self): return _Acc()
    def order_calc_margin(self, *a, **k): return 100.0
    def positions_get(self, *a, **k): return []
    def last_error(self): return 0


@pytest.fixture(scope="module")
def run_and_stub():
    snapshot = dict(os.environ)

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv_stub

    stub = StubMT5()
    mt5_module = types.ModuleType("MetaTrader5")
    for name in dir(stub):
        if not name.startswith("_"):
            setattr(mt5_module, name, getattr(stub, name))
    mt5_module.__getattr__ = lambda n: 0
    sys.modules["MetaTrader5"] = mt5_module

    for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        os.environ[key] = ""
    os.environ["TRADING_MODE"] = "paper"

    sys.modules.pop("run", None)
    import run as run_module_
    run_module_.mt5 = mt5_module

    yield run_module_, stub

    sys.modules.pop("run", None)
    sys.modules.pop("MetaTrader5", None)
    sys.modules["dotenv"] = importlib.import_module("dotenv")
    os.environ.clear()
    os.environ.update(snapshot)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment File Updater Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_update_env_file_preserves_comments_and_updates_keys(run_and_stub):
    run_module, _ = run_and_stub
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tf:
        tf.write("# Test Header\nRISK_PER_TRADE=0.01\nMAX_OPEN_POS=3\n# Footer comment\n")
        temp_path = tf.name

    try:
        updates = {"RISK_PER_TRADE": "0.0075", "NEW_KEY": "CUSTOM_VAL"}
        success = run_module._update_env_file(updates, env_path=temp_path)
        assert success is True

        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Test Header\n" in content
        assert "RISK_PER_TRADE=0.0075\n" in content
        assert "MAX_OPEN_POS=3\n" in content
        assert "NEW_KEY=CUSTOM_VAL\n" in content
        assert "# Footer comment\n" in content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gold Macro Trend Veto & Score Floor Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_gold_macro_trend_veto(run_and_stub, monkeypatch):
    """Gold trades are vetoed when H4 opposes the entry with high trend score."""
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["min_signal_score"] = 55
    cfg["gold_min_signal_score"] = 58
    cfg["location_gate"] = False

    bot = run_module.StandaloneAIBot(cfg)

    # Synthetic M15 frame with ADX >= 20 and ATR
    idx = pd.date_range("2026-01-01", periods=60, freq="15min")
    m15 = pd.DataFrame({
        "open": np.linspace(2640, 2650, 60),
        "high": np.linspace(2641, 2651, 60),
        "low": np.linspace(2639, 2649, 60),
        "close": np.linspace(2640, 2650, 60),
        "tick_volume": [100.0] * 60,
    }, index=idx)
    m15p = run_module._prepare(m15, cfg)

    # Force BUY direction with high score (75)
    monkeypatch.setattr(run_module, "_session_ok", lambda c: True)
    monkeypatch.setattr(
        run_module, "_score_direction",
        lambda ai_dir, ai_conf, m, h1, h4, d, c: (75.0 if d == 1 else 10.0),
    )

    # Mock H4 dataframe with established BEARISH opposing trend
    h4p = pd.DataFrame({
        "ema50": [2660.0] * 30,
        "ema200": [2670.0] * 30,
        "ema9": [2650.0] * 30,
        "ema21": [2655.0] * 30,
        "rsi": [35.0] * 30,
    })

    # Test opposing score >= 0.65 -> Macro Veto triggers
    monkeypatch.setattr(run_module, "_trend_score", lambda df, direction: 0.85 if direction == -1 else 0.10)

    direction, score, conf, reason = bot._get_signal(m15p, h1p=None, h4p=h4p)
    assert direction == 0
    assert "gold macro veto" in reason


def test_gold_minimum_signal_score_floor(run_and_stub, monkeypatch):
    """Gold trades require minimum 58 confluence score."""
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["min_signal_score"] = 50
    cfg["gold_min_signal_score"] = 58
    cfg["location_gate"] = False

    bot = run_module.StandaloneAIBot(cfg)

    idx = pd.date_range("2026-01-01", periods=60, freq="15min")
    m15 = pd.DataFrame({
        "open": np.linspace(2640, 2650, 60),
        "high": np.linspace(2641, 2651, 60),
        "low": np.linspace(2639, 2649, 60),
        "close": np.linspace(2640, 2650, 60),
        "tick_volume": [100.0] * 60,
    }, index=idx)
    m15p = run_module._prepare(m15, cfg)

    monkeypatch.setattr(run_module, "_session_ok", lambda c: True)
    # Score 54 is above standard threshold (50) but below Gold floor (58)
    monkeypatch.setattr(
        run_module, "_score_direction",
        lambda ai_dir, ai_conf, m, h1, h4, d, c: (54.0 if d == 1 else 10.0),
    )

    direction, score, conf, reason = bot._get_signal(m15p, h1p=None, h4p=None)
    assert direction == 0
    assert "gold score" in reason and "threshold" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dashboard API Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dashboard_api_get_settings(run_and_stub):
    run_module, _ = run_and_stub
    from fastapi.testclient import TestClient

    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["risk_per_trade"] = 0.0075
    cfg["max_daily_dd"] = 0.05
    cfg["max_open_pos"] = 5
    cfg["timeframe"] = 5
    cfg["dry_run"] = False

    bot = run_module.StandaloneAIBot(cfg)
    app = run_module.create_app([bot])
    client = TestClient(app)

    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()

    assert data["trading_mode"] == "live"
    assert data["entry_timeframe"] == "M5"
    assert data["risk_per_trade"] == 0.75
    assert data["max_daily_drawdown_pct"] == 5.0
    assert data["max_open_positions"] == 5
    assert "XAUUSDm" in data["raw_symbols"]


def test_dashboard_api_post_settings(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    from fastapi.testclient import TestClient

    # Intercept _update_env_file so tests don't modify real .env
    saved_env = {}
    monkeypatch.setattr(run_module, "_update_env_file", lambda updates, env_path=".env": saved_env.update(updates) or True)

    cfg = dict(run_module.CONFIG)
    bot = run_module.StandaloneAIBot(cfg)
    app = run_module.create_app([bot])
    client = TestClient(app)

    payload = {
        "trading_mode": "live",
        "entry_timeframe": "M5",
        "risk_per_trade": 1.0,         # 1.0% -> 0.01
        "max_daily_drawdown_pct": 4.0, # 4.0% -> 0.04
        "max_open_positions": 8,
        "sl_atr_mult": 1.9,
        "enable_breakeven": True,
        "min_signal_score": 62,
    }

    res = client.post("/api/settings", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Verify live bot instance in memory was updated
    assert bot.cfg["dry_run"] is False
    assert bot.cfg["risk_per_trade"] == 0.01
    assert bot.cfg["max_daily_dd"] == 0.04
    assert bot.cfg["max_open_pos"] == 8
    assert bot.cfg["sl_atr_mult"] == 1.9
    assert bot.cfg["breakeven_trigger_atr"] == 0.50
    assert bot.cfg["min_signal_score"] == 62

    # Verify .env persistence dictionary was populated
    assert saved_env["TRADING_MODE"] == "live"
    assert saved_env["RISK_PER_TRADE"] == "0.01"
    assert saved_env["MAX_DAILY_DD"] == "0.04"
    assert saved_env["MAX_OPEN_POS"] == "8"
    assert saved_env["SL_ATR_MULT"] == "1.9"


def test_dashboard_api_config_sections(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    from fastapi.testclient import TestClient

    saved_env = {}
    monkeypatch.setattr(run_module, "_update_env_file", lambda updates, env_path=".env": saved_env.update(updates) or True)

    cfg = dict(run_module.CONFIG)
    bot = run_module.StandaloneAIBot(cfg)
    app = run_module.create_app([bot])
    client = TestClient(app)

    # Test GET /api/config/risk
    res_risk_get = client.get("/api/config/risk")
    assert res_risk_get.status_code == 200
    assert "risk_per_trade" in res_risk_get.json()

    # Test POST /api/config/risk
    res_risk_post = client.post("/api/config/risk", json={"risk_per_trade": 0.8, "max_pos": 4, "enable_be": True})
    assert res_risk_post.status_code == 200
    assert bot.cfg["risk_per_trade"] == 0.008
    assert bot.cfg["max_open_pos"] == 4

    # Test GET /api/config/strategy
    res_strat_get = client.get("/api/config/strategy")
    assert res_strat_get.status_code == 200
    assert "trend_min_score" in res_strat_get.json()

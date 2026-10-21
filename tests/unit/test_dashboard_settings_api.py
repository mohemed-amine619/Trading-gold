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
        self.TRADE_ACTION_SLTP = 6
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_IOC = 1
        self.TRADE_RETCODE_DONE = 10009
        self.TRADE_RETCODE_NO_CHANGES = 10025
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
    def order_send(self, req): return types.SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)
    def last_error(self): return (1, "Success")


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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modify SL/SLTP retcode 10025 & No-Change Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_modify_sl_skips_identical_sl(run_and_stub):
    run_module, stub = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    # Position already has SL at 2640.00
    pos = types.SimpleNamespace(
        symbol="XAUUSDm",
        ticket=12345,
        type=stub.ORDER_TYPE_BUY,
        sl=2640.00,
        tp=2680.00,
        magic=888992
    )

    calls = []
    stub.order_send = lambda req: calls.append(req) or types.SimpleNamespace(retcode=stub.TRADE_RETCODE_DONE)

    # Modifying to the exact same SL should return True immediately with NO order_send
    res = bot._modify_sl(pos, 2640.00)
    assert res is True
    assert len(calls) == 0, "Expected no order_send when SL is identical"


def test_modify_sl_handles_retcode_10025(run_and_stub):
    run_module, stub = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    pos = types.SimpleNamespace(
        symbol="XAUUSDm",
        ticket=12345,
        type=stub.ORDER_TYPE_BUY,
        sl=2635.00,
        tp=2680.00,
        magic=888992
    )

    # Broker responds with TRADE_RETCODE_NO_CHANGES (10025)
    stub.order_send = lambda req: types.SimpleNamespace(retcode=stub.TRADE_RETCODE_NO_CHANGES)

    res = bot._modify_sl(pos, 2640.00)
    assert res is True, "Expected True when broker returns 10025 NO_CHANGES"


def test_modify_sltp_skips_identical_sltp(run_and_stub):
    run_module, stub = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    pos = types.SimpleNamespace(
        symbol="XAUUSDm",
        ticket=12345,
        type=stub.ORDER_TYPE_BUY,
        sl=2640.00,
        tp=2680.00,
        magic=888992
    )

    calls = []
    stub.order_send = lambda req: calls.append(req) or types.SimpleNamespace(retcode=stub.TRADE_RETCODE_DONE)

    res = bot._modify_sltp(pos, 2640.00, 2680.00)
    assert res is True
    assert len(calls) == 0, "Expected no order_send when SL and TP are unchanged"


def test_close_position_sanitizes_comment_and_recovers_from_err_neg2(run_and_stub):
    """Ensure _close_position formats comments cleanly and handles MT5 -2 invalid comment error."""
    run_module, stub = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    pos = types.SimpleNamespace(
        symbol="XAUUSDm",
        ticket=2095561400,
        volume=0.01,
        type=stub.ORDER_TYPE_BUY,
        profit=5.0,
        swap=0.0,
        magic=888992
    )

    calls = []
    def mock_order_send(req):
        calls.append(dict(req))
        # First call with complex comment triggers -2 error
        if "(" in req.get("comment", "") or ":" in req.get("comment", "") or len(req.get("comment", "")) > 25:
            run_module.mt5.last_error = lambda: (-2, 'Invalid "comment" argument')
            return None
        run_module.mt5.last_error = lambda: (1, 'Success')
        return types.SimpleNamespace(retcode=run_module.mt5.TRADE_RETCODE_DONE)

    run_module.mt5.order_send = mock_order_send

    reason = "Scalp Rejection Exit (Opposite Pinbar)"
    res = bot._close_position(pos, reason)

    assert res is True
    assert len(calls) >= 1
    # Check that initial comment was sanitized (no colons, no parentheses, short)
    first_comment = calls[0]["comment"]
    assert ":" not in first_comment
    assert "(" not in first_comment
    assert ")" not in first_comment
    assert len(first_comment) <= 20



# ─────────────────────────────────────────────────────────────────────────────
# 5. Spatial Trap Veto Tests in _get_signal
# ─────────────────────────────────────────────────────────────────────────────

def test_spatial_trap_veto_breakdown_long(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["strict_location_veto"] = True
    cfg["min_signal_score"] = 40
    cfg["ai_confidence_threshold"] = 0.50
    bot = run_module.StandaloneAIBot(cfg)

    # Mock location where price is below range low (breakdown)
    mock_loc = types.SimpleNamespace(
        is_valid=True,
        price=2630.0,
        range_low=2635.0,
        range_high=2660.0,
        below_range=True,
        above_range=False,
        atr=2.0,
        distance_to_resistance=lambda: 5.0,
        distance_to_support=lambda: 10.0,
        nearest_resistance=lambda: None,
        nearest_support=lambda: None,
        structure=types.SimpleNamespace(last_choch=None),
        entry_gate=lambda d: types.SimpleNamespace(allowed=True, cautions=[], reason="ok")
    )
    monkeypatch.setattr(bot, "_location_for", lambda df: mock_loc)

    # DataFrame with indicators
    df = pd.DataFrame({
        "open": [2630.0]*60, "high": [2632.0]*60, "low": [2628.0]*60, "close": [2630.0]*60,
        "adx": [25.0]*60, "atr": [2.0]*60, "ema_fast": [2631.0]*60, "ema_slow": [2629.0]*60,
        "volume": [1000]*60, "rsi": [55.0]*60
    })

    # Mock AI ensemble predict
    bot.ai = types.SimpleNamespace(
        update=lambda df: None,
        predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.65)
    )

    monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
    monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 65.0)

    direction, score, conf, reason = bot._get_signal(df, df, df)
    assert direction == 0
    assert "breakdown" in reason


def test_spatial_trap_veto_brick_wall_overhead(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["strict_location_veto"] = True
    cfg["min_signal_score"] = 40
    cfg["ai_confidence_threshold"] = 0.50
    bot = run_module.StandaloneAIBot(cfg)

    # Mock location with strong resistance 0.20 ATR overhead
    mock_res = types.SimpleNamespace(price=2640.40, strength=0.8)
    mock_loc = types.SimpleNamespace(
        is_valid=True,
        price=2640.0,
        range_low=2620.0,
        range_high=2660.0,
        below_range=False,
        above_range=False,
        atr=2.0,
        distance_to_resistance=lambda: 0.40, # 0.20 ATR away!
        distance_to_support=lambda: 10.0,
        nearest_resistance=lambda: mock_res,
        nearest_support=lambda: None,
        structure=types.SimpleNamespace(last_choch=None),
        entry_gate=lambda d: types.SimpleNamespace(allowed=True, cautions=[], reason="ok")
    )
    monkeypatch.setattr(bot, "_location_for", lambda df: mock_loc)

    df = pd.DataFrame({
        "open": [2640.0]*60, "high": [2642.0]*60, "low": [2638.0]*60, "close": [2640.0]*60,
        "adx": [25.0]*60, "atr": [2.0]*60, "volume": [1000]*60, "rsi": [55.0]*60
    })

    bot.ai = types.SimpleNamespace(
        update=lambda df: None,
        predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.65)
    )

    monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
    monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 65.0)

    direction, score, conf, reason = bot._get_signal(df, df, df)
    assert direction == 0
    assert "strong resistance" in reason


def test_ai_confidence_gate_rejects_coin_flip(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "BTCUSDm" # Non-gold symbol
    cfg["min_signal_score"] = 45
    cfg["ai_confidence_threshold"] = 0.55
    bot = run_module.StandaloneAIBot(cfg)

    df = pd.DataFrame({
        "open": [82000.0]*60, "high": [82100.0]*60, "low": [81900.0]*60, "close": [82000.0]*60,
        "adx": [25.0]*60, "atr": [50.0]*60, "volume": [1000]*60, "rsi": [55.0]*60
    })

    bot.ai = types.SimpleNamespace(
        update=lambda df: None,
        predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.50) # 50% coin flip
    )

    monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
    monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 58.0)

    direction, score, conf, reason = bot._get_signal(df, df, df)
    assert direction == 0
    assert "AI conf" in reason


def test_api_settings_syncs_profit_lock_and_spatial_veto(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    from fastapi.testclient import TestClient

    saved_env = {}
    monkeypatch.setattr(run_module, "_update_env_file", lambda updates, env_path=".env": saved_env.update(updates) or True)

    cfg = dict(run_module.CONFIG)
    bot = run_module.StandaloneAIBot(cfg)
    app = run_module.create_app([bot])
    client = TestClient(app)

    # 1. Test GET contains new keys
    res_get = client.get("/api/settings")
    assert res_get.status_code == 200
    data = res_get.json()
    assert "breakeven_trigger_atr" in data
    assert "profit_lock_activate_atr" in data
    assert "profit_giveback_pct" in data
    assert "tp_rr_ratio" in data
    assert "strict_location_veto" in data

    # 2. Test POST updates both bot memory and .env updates
    payload = {
        "breakeven_trigger_atr": 0.40,
        "profit_lock_activate_atr": 0.65,
        "profit_giveback_pct": 25.0,
        "tp_rr_ratio": 2.2,
        "strict_location_veto": True
    }
    res_post = client.post("/api/settings", json=payload)
    assert res_post.status_code == 200

    assert bot.cfg["breakeven_trigger_atr"] == 0.40
    assert bot.cfg["profit_lock_activate_atr"] == 0.65
    assert bot.cfg["profit_giveback_pct"] == 0.25
    assert bot.cfg["tp_rr_ratio"] == 2.2
    assert bot.cfg["strict_location_veto"] is True

    assert saved_env["BREAKEVEN_TRIGGER_ATR"] == "0.4"
    assert saved_env["PROFIT_LOCK_ACTIVATE_ATR"] == "0.65"
    assert saved_env["PROFIT_GIVEBACK_PCT"] == "0.25"
    assert saved_env["TP_RR_RATIO"] == "2.2"
    assert saved_env["STRICT_LOCATION_VETO"] == "true"


def test_llm_critic_veto_in_get_signal(run_and_stub, monkeypatch):
    """When LLM critic vetoes trade candidate (trap/exhaustion), bot rejects signal."""
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["min_signal_score"] = 50
    cfg["gold_min_signal_score"] = 50
    cfg["llm_critic_enabled"] = True
    bot = run_module.StandaloneAIBot(cfg)
    bot.ai = types.SimpleNamespace(
        update=lambda df: None,
        predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.70)
    )

    df = pd.DataFrame({
        "open": [2640.0]*60, "high": [2645.0]*60, "low": [2638.0]*60, "close": [2642.0]*60,
        "adx": [25.0]*60, "atr": [4.0]*60, "volume": [1000]*60, "rsi": [60.0]*60
    })

    monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
    monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 65.0)

    # Mock reasoner with Veto
    class MockCritique:
        approved = False
        decision = "PASS"
        confidence = 35.0
        reasoning = "Overextended into key liquidity pool, high reversal probability"
        risk_assessment = "Unfavorable"

    class MockReasoner:
        def evaluate_trade_candidate(self, **kwargs):
            return MockCritique()

    bot._sltp_reasoner = MockReasoner()

    direction, score, conf, reason = bot._get_signal(df, df, df)
    assert direction == 0
    assert "LLM critic veto" in reason
    assert "reversal probability" in reason


def test_api_settings_llm_controls_roundtrip(run_and_stub, monkeypatch):
    """Dashboard API GET and POST synchronize LLM critic toggle, provider, URL, and model."""
    run_module, _ = run_and_stub
    from fastapi.testclient import TestClient

    saved_env = {}
    monkeypatch.setattr(run_module, "_update_env_file", lambda updates, env_path=".env": saved_env.update(updates) or True)

    cfg = dict(run_module.CONFIG)
    bot = run_module.StandaloneAIBot(cfg)
    app = run_module.create_app([bot])
    client = TestClient(app)

    # 1. Test GET returns LLM fields
    res_get = client.get("/api/settings")
    assert res_get.status_code == 200
    data = res_get.json()
    assert "llm_critic_enabled" in data
    assert "llm_provider" in data
    assert "local_llm_url" in data
    assert "local_llm_model" in data

    # 2. Test POST updates LLM config
    payload = {
        "llm_critic_enabled": True,
        "llm_provider": "openai_compatible",
        "local_llm_url": "https://api.groq.com/openai/v1",
        "local_llm_model": "llama-3.3-70b-versatile"
    }
    res_post = client.post("/api/settings", json=payload)
    assert res_post.status_code == 200

    assert bot.cfg["llm_critic_enabled"] is True
    assert bot.cfg["llm_provider"] == "openai_compatible"
    assert bot.cfg["local_llm_url"] == "https://api.groq.com/openai/v1"
    assert bot.cfg["local_llm_model"] == "llama-3.3-70b-versatile"

    assert saved_env["LLM_CRITIC_ENABLED"] == "true"
    assert saved_env["LLM_PROVIDER"] == "openai_compatible"
    assert saved_env["LOCAL_LLM_URL"] == "https://api.groq.com/openai/v1"
    assert saved_env["LOCAL_LLM_MODEL"] == "llama-3.3-70b-versatile"



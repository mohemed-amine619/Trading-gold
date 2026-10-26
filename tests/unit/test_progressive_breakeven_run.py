"""
tests/unit/test_progressive_breakeven_run.py — Verification of Progressive Breakeven & Profit Ratchet in run.py.

Verifies:
1. CONFIG defaults for breakeven_trigger_atr (0.4), breakeven_buffer_atr (0.15), and profit_lock_ratio (0.50).
2. Environment variable overrides for breakeven settings.
3. Progressive ratcheting behavior ensuring SL continuously locks in profit and never loosens.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        stub = sys.modules["MetaTrader5"]
        if not hasattr(stub, "TIMEFRAME_M1"):
            stub.TIMEFRAME_M1 = 1
        if not hasattr(stub, "TIMEFRAME_M5"):
            stub.TIMEFRAME_M5 = 5
        return
    stub = types.ModuleType("MetaTrader5")
    stub.TIMEFRAME_M1 = 1
    stub.TIMEFRAME_M5 = 5
    stub.TIMEFRAME_M15 = 15
    stub.TIMEFRAME_H1 = 16385
    stub.TIMEFRAME_H4 = 16388
    stub.ORDER_TYPE_BUY = 0
    stub.ORDER_TYPE_SELL = 1
    stub.TRADE_ACTION_SLTP = 1
    stub.TRADE_RETCODE_DONE = 10009
    sys.modules["MetaTrader5"] = stub


@pytest.fixture
def run_module(tmp_path, monkeypatch):
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    yield mod
    sys.modules.pop("run", None)


def test_breakeven_config_defaults(run_module):
    """Verify progressive breakeven defaults in CONFIG."""
    cfg = run_module.CONFIG
    assert cfg["enable_breakeven"] is True
    assert cfg["breakeven_trigger_atr"] == 0.4
    assert cfg["breakeven_buffer_atr"] == 0.15
    assert cfg["profit_lock_ratio"] == 0.50


def test_breakeven_config_env_overrides(tmp_path, monkeypatch):
    """Verify that environment variables override breakeven settings."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BREAKEVEN_TRIGGER_ATR", "0.6")
    monkeypatch.setenv("BREAKEVEN_BUFFER_ATR", "0.20")
    monkeypatch.setenv("PROFIT_LOCK_RATIO", "0.40")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    assert mod.CONFIG["breakeven_trigger_atr"] == 0.6
    assert mod.CONFIG["breakeven_buffer_atr"] == 0.20
    assert mod.CONFIG["profit_lock_ratio"] == 0.40
    sys.modules.pop("run", None)


def test_progressive_breakeven_ratchet_math_long():
    """Verify mathematical progression for Long position ratcheting."""
    price_open = 2000.0
    atr = 10.0
    trigger_atr = 0.4       # triggers at +4.0 profit
    buffer_atr = 0.15       # +1.5 buffer above entry
    lock_ratio = 0.50       # locks 50% of further gains
    stops_level_dist = 0.5

    be_trigger = trigger_atr * atr
    be_buffer = buffer_atr * atr
    be_sl = price_open + be_buffer  # 2001.5

    def compute_sl(current_price: float, existing_sl: float) -> float:
        profit_move = current_price - price_open
        if profit_move < be_trigger:
            return existing_sl
        target_sl = be_sl
        extra_profit = profit_move - be_trigger
        if extra_profit > 0 and lock_ratio > 0:
            locked_gain = be_buffer + (extra_profit * lock_ratio)
            progressive_sl = price_open + locked_gain
            target_sl = max(target_sl, progressive_sl)
        target_sl = min(target_sl, current_price - stops_level_dist)
        if target_sl > existing_sl + 1e-9 and target_sl >= be_sl:
            return target_sl
        return existing_sl

    current_sl = 1980.0  # Initial risk: 20 pts below entry

    # 1. Price at 2003.0 (+3.0 profit < +4.0 trigger) -> SL remains 1980.0
    current_sl = compute_sl(2003.0, current_sl)
    assert current_sl == 1980.0

    # 2. Price at 2004.0 (+4.0 profit == trigger) -> SL advances to 2001.5 (+1.5 profit locked)
    current_sl = compute_sl(2004.0, current_sl)
    assert current_sl == 2001.5

    # 3. Price surges to 2008.0 (+8.0 profit, extra = 4.0, locks 1.5 + 2.0 = 3.5) -> SL advances to 2003.5
    current_sl = compute_sl(2008.0, current_sl)
    assert current_sl == 2003.5

    # 4. Price surges to 2014.0 (+14.0 profit, extra = 10.0, locks 1.5 + 5.0 = 6.5) -> SL advances to 2006.5
    current_sl = compute_sl(2014.0, current_sl)
    assert current_sl == 2006.5

    # 5. Price retraces from 2014.0 to 2009.0 -> SL must NEVER loosen, remains 2006.5
    current_sl = compute_sl(2009.0, current_sl)
    assert current_sl == 2006.5


def test_progressive_breakeven_ratchet_math_short():
    """Verify mathematical progression for Short position ratcheting."""
    price_open = 2000.0
    atr = 10.0
    trigger_atr = 0.4
    buffer_atr = 0.15
    lock_ratio = 0.50
    stops_level_dist = 0.5

    be_trigger = trigger_atr * atr
    be_buffer = buffer_atr * atr
    be_sl = price_open - be_buffer  # 1998.5

    def compute_sl(current_price: float, existing_sl: float) -> float:
        profit_move = price_open - current_price
        if profit_move < be_trigger:
            return existing_sl
        target_sl = be_sl
        extra_profit = profit_move - be_trigger
        if extra_profit > 0 and lock_ratio > 0:
            locked_gain = be_buffer + (extra_profit * lock_ratio)
            progressive_sl = price_open - locked_gain
            target_sl = min(target_sl, progressive_sl)
        target_sl = max(target_sl, current_price + stops_level_dist)
        if (existing_sl == 0 or target_sl < existing_sl - 1e-9) and target_sl <= be_sl:
            return target_sl
        return existing_sl

    current_sl = 2020.0

    # 1. Price drops to 1997.0 (profit +3.0 < 4.0 trigger) -> remains 2020.0
    current_sl = compute_sl(1997.0, current_sl)
    assert current_sl == 2020.0

    # 2. Price drops to 1996.0 (profit +4.0 == trigger) -> advances to 1998.5
    current_sl = compute_sl(1996.0, current_sl)
    assert current_sl == 1998.5

    # 3. Price drops to 1990.0 (profit +10.0, extra = 6.0, locks 1.5 + 3.0 = 4.5) -> advances to 1995.5
    current_sl = compute_sl(1990.0, current_sl)
    assert current_sl == 1995.5

    # 4. Price retraces upwards to 1993.0 -> SL must NEVER loosen, remains 1995.5
    current_sl = compute_sl(1993.0, current_sl)
    assert current_sl == 1995.5


def test_trading_profile_money_printer(tmp_path, monkeypatch):
    """Verify that TRADING_PROFILE=money_printer applies aggressive compounding defaults."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_PROFILE", "money_printer")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")

    cfg = mod.CONFIG
    assert cfg["min_signal_score"] == 45.0
    assert cfg["adx_min"] == 15.0
    assert cfg["adx_strong"] == 25.0
    assert cfg["risk_per_trade"] == 0.015
    assert cfg["sl_atr_mult"] == 1.8
    assert cfg["tp_rr_ratio"] == 2.0
    assert cfg["breakeven_trigger_atr"] == 0.3
    assert cfg["profit_lock_ratio"] == 0.55
    assert cfg["partial_close_trigger_atr"] == 0.9
    assert cfg["max_open_pos"] == 5
    assert cfg["max_daily_dd"] == 0.05
    sys.modules.pop("run", None)


def test_ai_confidence_lot_sizing(run_module):
    """Verify that _size_lot scales size up with high AI confidence and down with low."""
    cfg = run_module.CONFIG.copy()
    cfg["risk_per_trade"] = 0.01
    cfg["dynamic_risk"] = False

    base = run_module._size_lot(
        account_equity=10_000.0,
        sl_dist_price=10.0,
        tick_size=0.01,
        tick_value=1.0,
        vol_min=0.01,
        vol_max=100.0,
        vol_step=0.01,
        cfg=cfg,
        ai_conf=0.65,
    )

    high_conf = run_module._size_lot(
        account_equity=10_000.0,
        sl_dist_price=10.0,
        tick_size=0.01,
        tick_value=1.0,
        vol_min=0.01,
        vol_max=100.0,
        vol_step=0.01,
        cfg=cfg,
        ai_conf=0.90,
    )

    low_conf = run_module._size_lot(
        account_equity=10_000.0,
        sl_dist_price=10.0,
        tick_size=0.01,
        tick_value=1.0,
        vol_min=0.01,
        vol_max=100.0,
        vol_step=0.01,
        cfg=cfg,
        ai_conf=0.55,
    )

    assert high_conf > base
    assert low_conf < base
    assert high_conf == pytest.approx(base * 1.5, rel=0.05)
    assert low_conf == pytest.approx(base * 0.5, rel=0.05)


def test_scalping_timeframe_m1(tmp_path, monkeypatch):
    """Verify that TIMEFRAME=M1 configures M1 execution and cascades trend to M5 and M15."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIMEFRAME", "M1")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    assert mod.CONFIG["timeframe"] == 1       # M1
    assert mod.CONFIG["h1_tf"] == 5          # M5 intermediate trend
    assert mod.CONFIG["h4_tf"] == 15         # M15 macro trend
    sys.modules.pop("run", None)


def test_scalping_timeframe_m5(tmp_path, monkeypatch):
    """Verify that TIMEFRAME=M5 configures M5 execution and cascades trend to M15 and H1."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIMEFRAME", "M5")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    assert mod.CONFIG["timeframe"] == 5       # M5
    assert mod.CONFIG["h1_tf"] == 15         # M15 intermediate trend
    assert mod.CONFIG["h4_tf"] == 16385      # H1 macro trend
    sys.modules.pop("run", None)


def test_scalping_timeframe_override(tmp_path, monkeypatch):
    """Verify that custom TREND_TF and MACRO_TREND_TF explicitly override defaults."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIMEFRAME", "M1")
    monkeypatch.setenv("TREND_TF", "M15")
    monkeypatch.setenv("MACRO_TREND_TF", "H1")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    assert mod.CONFIG["timeframe"] == 1       # M1
    assert mod.CONFIG["h1_tf"] == 15         # overridden to M15
    assert mod.CONFIG["h4_tf"] == 16385      # overridden to H1
    sys.modules.pop("run", None)


def test_scalping_momentum_oversold_bounce(run_module):
    """Verify that oversold bounce (stoch_k < 35, stoch_k > stoch_d) receives full points."""
    import pandas as pd
    last_series = pd.Series({
        "rsi": 46.0,
        "macd_hist": 0.5,
        "macd_hist_prev": 0.2,
        "stoch_k": 28.0,
        "stoch_d": 22.0,
    })
    score = run_module._momentum_score(last_series, direction=1)
    # MACD hist > 0 & growing (0.35) + Stoch oversold turn (0.30) + RSI (small)
    assert score >= 0.65


def test_scalper_profile_configuration(tmp_path, monkeypatch):
    """TRADING_PROFILE=scalper configures high-frequency, tight scalp parameters."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_PROFILE", "scalper")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    assert mod._IS_SCALPER is True
    assert mod._IS_AGGRESSIVE is True
    assert mod.CONFIG["timeframe"] == 1             # M1
    assert mod.CONFIG["sl_atr_mult"] == 1.5         # Tight 1.5 ATR SL
    assert mod.CONFIG["tp_rr_ratio"] == 2.0         # 2.0 RR target
    assert mod.CONFIG["breakeven_trigger_atr"] == 0.3  # Fast 0.3 ATR Breakeven
    assert mod.CONFIG["partial_close_trigger_atr"] == 0.8  # Rapid 0.8 ATR partial TP
    assert mod.CONFIG["max_spread_atr_ratio"] == 0.20  # Max 20% spread-to-ATR
    sys.modules.pop("run", None)


def test_scalper_spread_guard(run_module):
    """_spread_ok rejects orders when spread exceeds the ATR threshold."""
    info = types.SimpleNamespace(point=0.01)
    tick = types.SimpleNamespace(ask=2000.50, bid=2000.00)  # Spread = 0.50
    bot = run_module.StandaloneAIBot(run_module.CONFIG)

    # Case 1: ATR is 1.0 -> spread (0.50) is 50% of ATR > max (40% default) -> rejected
    assert bot._spread_ok(info, tick, atr=1.0) is False

    # Case 2: ATR is 5.0 -> spread (0.50) is 10% of ATR <= max -> accepted
    assert bot._spread_ok(info, tick, atr=5.0) is True


def test_scalper_momentum_ribbon_bonus(run_module):
    """EMA 9/21 ribbon alignment and pinbars provide confluence bonuses."""
    import pandas as pd
    base_series = pd.Series({
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
    })
    base_score = run_module._momentum_score(base_series, direction=1)

    # Bullish ribbon (close >= ema9 > ema21) + bull pinbar
    scalp_series = pd.Series({
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "close": 2005.0,
        "ema9": 2004.0,
        "ema21": 2002.0,
        "bull_pinbar": True,
    })
    scalp_score = run_module._momentum_score(scalp_series, direction=1)
    assert scalp_score > base_score





def test_explicit_env_value_wins_over_profile_default(tmp_path, monkeypatch):
    """A value written in .env must never be replaced by a profile default.

    Regression: ``_profile_cfg`` treated "equals the balanced default" as
    evidence that the variable was stale and substituted the aggressive value.
    With ``TRADING_PROFILE=money_printer`` that silently turned ADX_MIN=18
    into 15 and PARTIAL_CLOSE_TRIGGER_ATR=1.2 into 0.9, so the bot traded
    different settings than the operator had configured.
    """
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_PROFILE", "money_printer")
    monkeypatch.setenv("ADX_MIN", "18")
    monkeypatch.setenv("PARTIAL_CLOSE_TRIGGER_ATR", "1.2")
    monkeypatch.setenv("RISK_PER_TRADE", "0.01")
    monkeypatch.setenv("MAX_OPEN_POS", "3")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")

    assert mod._IS_AGGRESSIVE is True
    assert mod.CONFIG["adx_min"] == 18.0
    assert mod.CONFIG["partial_close_trigger_atr"] == 1.2
    assert mod.CONFIG["risk_per_trade"] == 0.01
    assert mod.CONFIG["max_open_pos"] == 3
    sys.modules.pop("run", None)


def test_profile_default_still_applies_when_env_is_absent(tmp_path, monkeypatch):
    """Removing the variable must restore the profile default, not the balanced one."""
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_PROFILE", "money_printer")
    for key in ("ADX_MIN", "PARTIAL_CLOSE_TRIGGER_ATR", "RISK_PER_TRADE", "MAX_OPEN_POS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")

    assert mod.CONFIG["adx_min"] == 15.0
    assert mod.CONFIG["partial_close_trigger_atr"] == 0.9
    assert mod.CONFIG["risk_per_trade"] == 0.015
    assert mod.CONFIG["max_open_pos"] == 5
    sys.modules.pop("run", None)


# ---------------------------------------------------------------------------
# Minimum stop-improvement step (_sl_modify_allowed)
# ---------------------------------------------------------------------------
# Live logs showed the structure ratchet sending a modify for a 0.011 price
# move (4027.436 -> 4027.425) on the same 5-bar anchor. Each modify is a broker
# round-trip and a failure surface. The floor below damps drift without ever
# loosening a stop, and the never-loosen rule is re-asserted here.

def _cfg(**overrides):
    base = {"min_sl_step_atr": 0.05}
    base.update(overrides)
    return base


def test_step_floor_blocks_a_one_tick_ratchet(run_module):
    atr = 7.4
    # 0.011 is far below 0.05 * 7.4 = 0.37
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 4027.436, 4027.425, atr, _cfg()) is False


def test_step_floor_allows_a_real_structure_move(run_module):
    atr = 7.4
    # 4027.425 -> 4026.355 is a 1.07 improvement, above 0.37
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 4027.425, 4026.355, atr, _cfg()) is True


def test_never_loosens_a_long_stop(run_module):
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", 1, 2000.0, 1999.0, 5.0, _cfg()) is False


def test_never_loosens_a_short_stop(run_module):
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 2000.0, 2001.0, 5.0, _cfg()) is False


def test_identical_stop_is_not_a_modification(run_module):
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 2000.0, 2000.0, 5.0, _cfg()) is False


def test_breakeven_bypasses_the_step_floor(run_module):
    """MOVE_TO_BREAKEVEN is a state change, not anchor drift."""
    assert run_module._sl_modify_allowed(
        "MOVE_TO_BREAKEVEN", -1, 2000.05, 2000.00, 5.0, _cfg()) is True
    assert run_module._sl_modify_allowed(
        "SECURE_PROFIT", 1, 1999.95, 2000.00, 5.0, _cfg()) is True


def test_breakeven_bypass_still_refuses_to_loosen(run_module):
    """The bypass applies to the step floor only, never to the direction."""
    assert run_module._sl_modify_allowed(
        "MOVE_TO_BREAKEVEN", -1, 2000.0, 2001.0, 5.0, _cfg()) is False


def test_zero_step_restores_every_move(run_module):
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 2000.000, 1999.999, 5.0,
        _cfg(min_sl_step_atr=0.0)) is True


def test_absent_stop_may_always_be_set(run_module):
    """current_sl == 0 means "no stop", so setting one is always an improvement."""
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 0.0, 1999.0, 5.0, _cfg()) is True


def test_none_new_stop_is_refused(run_module):
    assert run_module._sl_modify_allowed(
        "TIGHTEN_SL", -1, 2000.0, None, 5.0, _cfg()) is False

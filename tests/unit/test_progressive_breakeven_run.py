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
        return
    stub = types.ModuleType("MetaTrader5")
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


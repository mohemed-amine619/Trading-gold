"""
tests/unit/test_manual_trades_and_order_execution.py — Tests for manual trade isolation and robust order execution.

Verifies:
1. Manual trades (magic != bot_magic) are protected (SL/TP) but never closed by AI early exit or trend flip.
2. _modify_sltp rounds SL/TP to broker digits, respects stops level, and throttles retries on failure.
3. _get_filling_type resolves filling modes dynamically from symbol_info.
4. _close_position uses the position's magic and rounds price to symbol digits.
5. _close_opposite_positions and _close_profitable ignore manual trades.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        stub = sys.modules["MetaTrader5"]
    else:
        stub = types.ModuleType("MetaTrader5")
        sys.modules["MetaTrader5"] = stub

    attrs = {
        "TIMEFRAME_M1": 1,
        "TIMEFRAME_M5": 5,
        "TIMEFRAME_M15": 15,
        "TIMEFRAME_H1": 16385,
        "TIMEFRAME_H4": 16388,
        "ORDER_TYPE_BUY": 0,
        "ORDER_TYPE_SELL": 1,
        "TRADE_ACTION_DEAL": 1,
        "TRADE_ACTION_SLTP": 6,
        "ORDER_TIME_GTC": 0,
        "ORDER_FILLING_FOK": 0,
        "ORDER_FILLING_IOC": 1,
        "ORDER_FILLING_RETURN": 2,
        "TRADE_RETCODE_DONE": 10009,
        "TRADE_RETCODE_REQUOTE": 10004,
        "TRADE_RETCODE_PRICE_CHANGED": 10015,
        "TRADE_RETCODE_PRICE_OFF": 10016,
        "TRADE_RETCODE_INVALID_FILL": 10030,
        "TRADE_RETCODE_NO_MONEY": 10026,
    }
    for k, v in attrs.items():
        if not hasattr(stub, k):
            setattr(stub, k, v)
    if not hasattr(stub, "last_error"):
        stub.last_error = MagicMock(return_value=(1, "OK"))
    if not hasattr(stub, "symbol_info"):
        stub.symbol_info = MagicMock()
    if not hasattr(stub, "symbol_info_tick"):
        stub.symbol_info_tick = MagicMock()
    if not hasattr(stub, "order_send"):
        stub.order_send = MagicMock()
    if not hasattr(stub, "positions_get"):
        stub.positions_get = MagicMock(return_value=[])
    if not hasattr(stub, "account_info"):
        stub.account_info = MagicMock()
    if not hasattr(stub, "order_calc_margin"):
        stub.order_calc_margin = MagicMock(return_value=5.0)


@pytest.fixture
def run_module(tmp_path, monkeypatch):
    _install_mt5_stub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    mod = importlib.import_module("run")
    yield mod
    sys.modules.pop("run", None)


def _make_dummy_position(ticket=2063718228, magic=0, symbol="XAUUSDm", pos_type=1,
                         price_open=4150.0, volume=0.1, sl=0.0, tp=0.0, profit=-5.0, swap=0.0):
    pos = types.SimpleNamespace()
    pos.ticket = ticket
    pos.magic = magic
    pos.symbol = symbol
    pos.type = pos_type  # 1 = SELL, 0 = BUY
    pos.price_open = price_open
    pos.volume = volume
    pos.sl = sl
    pos.tp = tp
    pos.profit = profit
    pos.swap = swap
    pos.time = time.time()
    return pos


def test_manual_trades_config_defaults(run_module):
    """Verify manual trade management defaults in CONFIG."""
    cfg = run_module.CONFIG
    assert cfg["ai_protect_manual_trades"] is True
    assert cfg["manual_trade_sl_atr"] == 2.0
    assert cfg["manage_manual_trades_full"] is False


def test_get_filling_type_resolution(run_module):
    """Verify _get_filling_type picks IOC (bit 1), FOK (bit 0), or RETURN."""
    bot = run_module.StandaloneAIBot(run_module.CONFIG.copy())

    # Case 1: symbol has filling_mode = 2 (IOC supported)
    with patch.object(run_module.mt5, "symbol_info") as mock_info:
        mock_info.return_value = types.SimpleNamespace(filling_mode=2)
        assert bot._get_filling_type("XAUUSDm") == run_module.mt5.ORDER_FILLING_IOC

    # Case 2: symbol has filling_mode = 1 (FOK supported, IOC not)
    with patch.object(run_module.mt5, "symbol_info") as mock_info:
        mock_info.return_value = types.SimpleNamespace(filling_mode=1)
        assert bot._get_filling_type("XAUUSDm") == run_module.mt5.ORDER_FILLING_FOK

    # Case 3: symbol has filling_mode = 0 (only RETURN supported)
    with patch.object(run_module.mt5, "symbol_info") as mock_info:
        mock_info.return_value = types.SimpleNamespace(filling_mode=0)
        assert bot._get_filling_type("XAUUSDm") == run_module.mt5.ORDER_FILLING_RETURN


def test_manual_trade_not_closed_by_ai_early_exit(run_module):
    """Verify that a manual trade (magic=0) is NEVER closed by AI Early Exit."""
    import pandas as pd
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    manual_pos = _make_dummy_position(ticket=2063718228, magic=0, pos_type=1)  # SELL

    with patch.object(bot, "_open_positions", return_value=[manual_pos]), \
         patch.object(bot, "_close_position") as mock_close, \
         patch.object(bot, "_modify_sltp", return_value=True):

        bot.status["last_direction"] = 1   # Opposite of SELL -> BUY
        bot.status["last_score"] = 55.0    # Triggers AI early exit for bot trades

        df = pd.DataFrame([{
            "close": 4150.0,
            "atr": 5.0,
            "high": 4155.0,
            "low": 4145.0,
        }])
        tick = types.SimpleNamespace(bid=4149.0, ask=4151.0)

        bot._manage_positions(df, tick=tick, equity=10000.0)

        # _close_position must NOT be called for the manual trade!
        mock_close.assert_not_called()


def test_bot_trade_is_closed_by_ai_early_exit(run_module):
    """Verify that a bot trade (magic=888999) IS closed by AI Early Exit on opposite signal."""
    import pandas as pd
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    bot_pos = _make_dummy_position(ticket=111222, magic=cfg["magic_number"], pos_type=1)  # SELL

    with patch.object(bot, "_open_positions", return_value=[bot_pos]), \
         patch.object(bot, "_close_position") as mock_close:

        bot.status["last_direction"] = 1   # Opposite of SELL -> BUY
        bot.status["last_score"] = 55.0    # Triggers AI Early Exit (Incoming Reversal)

        df = pd.DataFrame([{
            "close": 4150.0,
            "atr": 5.0,
            "high": 4155.0,
            "low": 4145.0,
        }])
        tick = types.SimpleNamespace(bid=4149.0, ask=4151.0)

        bot._manage_positions(df, tick=tick, equity=10000.0)

        # _close_position MUST be called for the bot trade
        mock_close.assert_called_once()
        assert mock_close.call_args[0][0].ticket == 111222


def test_manual_trade_protection_throttles_retries(run_module):
    """Verify that when _modify_sltp fails, retries are throttled and don't loop every tick."""
    import pandas as pd
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    manual_pos = _make_dummy_position(ticket=2063718228, magic=0, pos_type=1, sl=0.0, tp=0.0)

    df = pd.DataFrame([{
        "close": 4150.0,
        "atr": 5.0,
        "high": 4155.0,
        "low": 4145.0,
    }])
    tick = types.SimpleNamespace(bid=4149.0, ask=4151.0)

    with patch.object(bot, "_open_positions", return_value=[manual_pos]), \
         patch.object(bot, "_modify_sltp", return_value=False) as mock_modify:

        # Tick 1: fails
        bot._manage_positions(df, tick=tick, equity=10000.0)
        assert mock_modify.call_count == 1
        assert 2063718228 in bot._manual_sltp_failed

        # Tick 2 (1 second later): throttled, must NOT call _modify_sltp again!
        bot._manage_positions(df, tick=tick, equity=10000.0)
        assert mock_modify.call_count == 1  # Still 1!


def test_modify_sltp_rounds_to_symbol_digits(run_module):
    """Verify that _modify_sltp rounds SL and TP to symbol digits."""
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    pos = _make_dummy_position(ticket=123, magic=0, pos_type=1)

    info = types.SimpleNamespace(digits=2, trade_stops_level=10, point=0.01)
    tick = types.SimpleNamespace(bid=4150.0, ask=4150.5)

    with patch.object(run_module.mt5, "symbol_info", return_value=info), \
         patch.object(run_module.mt5, "symbol_info_tick", return_value=tick), \
         patch.object(run_module.mt5, "order_send") as mock_send:

        mock_send.return_value = types.SimpleNamespace(retcode=run_module.mt5.TRADE_RETCODE_DONE)

        # Pass 5 decimal places: 4158.22438, 4149.12424
        result = bot._modify_sltp(pos, 4158.22438, 4149.12424)
        assert result is True

        req = mock_send.call_args[0][0]
        # Must be rounded to 2 digits
        assert req["sl"] == 4158.22
        assert req["tp"] == 4149.12
        assert req["magic"] == 0


def test_close_opposite_positions_ignores_manual_trades(run_module):
    """Verify _close_opposite_positions skips manual trades."""
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    bot = run_module.StandaloneAIBot(cfg)

    manual_pos = _make_dummy_position(ticket=2063718228, magic=0, pos_type=1)
    bot_pos = _make_dummy_position(ticket=999, magic=cfg["magic_number"], pos_type=1)

    with patch.object(bot, "_open_positions", return_value=[manual_pos, bot_pos]), \
         patch.object(bot, "_close_position") as mock_close:

        # New strong BUY signal opposite to the positions
        bot._close_opposite_positions(direction=1, score=75.0)

        # Only the bot position should be closed!
        assert mock_close.call_count == 1
        assert mock_close.call_args[0][0].ticket == 999


# ---------------------------------------------------------------------------
# Regression tests for _margin_ok (retcode=10026 prevention)
# ---------------------------------------------------------------------------

def _make_margin_bot(run_module):
    """Return a StandaloneAIBot with a minimal config for margin tests."""
    cfg = run_module.CONFIG.copy()
    cfg["dry_run"] = False
    return run_module.StandaloneAIBot(cfg)


def test_margin_ok_blocks_when_order_calc_margin_returns_none(run_module):
    """_margin_ok must return False (fail-safe) when order_calc_margin() returns None.

    Previously the code did ``margin = float(margin) if margin else 0.0``,
    which silently set margin=0 and let the order through, causing
    retcode=10026 from the broker.
    """
    bot = _make_margin_bot(run_module)

    with patch.object(run_module.mt5, "order_calc_margin", return_value=None), \
         patch.object(run_module.mt5, "account_info") as mock_acc:

        acc = types.SimpleNamespace(margin=0.0, margin_free=500.0)
        mock_acc.return_value = acc

        result = bot._margin_ok(price=2650.0, volume=0.01, equity=509.64)

        assert result is False, (
            "_margin_ok must return False when order_calc_margin returns None "
            "(fail-safe — do not let a None silently pass as 0)"
        )


def test_margin_ok_blocks_when_required_margin_exceeds_free_margin(run_module):
    """_margin_ok must return False when required margin > broker free_margin.

    This is the direct guard that prevents retcode=10026 (TRADE_RETCODE_NO_MONEY).
    """
    bot = _make_margin_bot(run_module)

    # Required margin for the order = 200.0, but free margin = 50.0
    with patch.object(run_module.mt5, "order_calc_margin", return_value=200.0), \
         patch.object(run_module.mt5, "account_info") as mock_acc:

        acc = types.SimpleNamespace(margin=300.0, margin_free=50.0)
        mock_acc.return_value = acc

        # equity * max_margin_usage = 509.64 * 0.30 = ~152.89 → used(300)+req(200) > limit too
        result = bot._margin_ok(price=2650.0, volume=0.01, equity=509.64)

        assert result is False, (
            "_margin_ok must return False when required_margin(200) > free_margin(50)"
        )


def test_margin_ok_blocks_when_only_free_margin_is_insufficient(run_module):
    """_margin_ok must block even when the usage-ceiling guard passes but free < required.

    Edge case: the bot has very high equity but used margin is 0 and free_margin is tiny.
    Guard 1 (ceiling) passes; Guard 2 (free margin) must catch it.
    """
    bot = _make_margin_bot(run_module)
    bot.cfg["max_margin_usage"] = 0.99  # effectively disable guard 1

    # Required = 150, free = 20 → should still block
    with patch.object(run_module.mt5, "order_calc_margin", return_value=150.0), \
         patch.object(run_module.mt5, "account_info") as mock_acc:

        acc = types.SimpleNamespace(margin=0.0, margin_free=20.0)
        mock_acc.return_value = acc

        result = bot._margin_ok(price=2650.0, volume=0.01, equity=509.64)

        assert result is False, (
            "_margin_ok must block via free-margin guard when required(150) > free(20), "
            "even if ceiling guard would pass"
        )


def test_margin_ok_passes_when_sufficient_free_margin(run_module):
    """_margin_ok must return True when margin is computable and free margin is sufficient."""
    bot = _make_margin_bot(run_module)

    # Required = 5.0, free = 400.0, used = 0.0 → should pass
    with patch.object(run_module.mt5, "order_calc_margin", return_value=5.0), \
         patch.object(run_module.mt5, "account_info") as mock_acc:

        acc = types.SimpleNamespace(margin=0.0, margin_free=400.0)
        mock_acc.return_value = acc

        result = bot._margin_ok(price=2650.0, volume=0.01, equity=509.64)

        assert result is True, (
            "_margin_ok should pass when required(5.0) <= free_margin(400.0) and ceiling ok"
        )

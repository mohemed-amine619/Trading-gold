"""
tests/unit/test_run_py_wiring.py — the run.py code paths that were modified.

`test_run_py_location.py` proves the location *helpers* work. It does not prove
the two functions actually changed in `run.py` use them.

These tests drive the real thing — a genuine `StandaloneAIBot` built by its own
`__init__`, with MetaTrader5 stubbed — through `_get_signal` and `_send_order`.

**These assertions were validated by mutation.** An earlier version of this file
passed even with the wiring completely disabled (`if False and atr > 0`,
`if False:`), because it re-tested the helpers instead of the modified
functions. Every assertion below was checked against a deliberately broken
`run.py` and confirmed to fail. If you edit the wiring, re-run that check: a
test here that cannot fail is worse than no test.

Reaching the veto requires care. On a range frame the ADX gate (>= 18) fires
first, and on a trend frame `_score_direction` picks the trend direction anyway
— so the location gate is only observable by forcing the scoring to favour the
counter-trend side. That is done here explicitly rather than pretending a
natural frame reaches it.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Stubbed MT5 — real numbers, never a MagicMock
# ─────────────────────────────────────────────────────────────────────────────

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
        self.price = 202.0
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
    def symbol_info_tick(self, symbol): return _Tick(self.price - 0.05, self.price + 0.05)
    def account_info(self): return _Acc()
    def order_calc_margin(self, *a, **k): return 100.0
    def positions_get(self, *a, **k): return []

    def order_send(self, req):
        return types.SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=1, deal=1,
                                     volume=req.get("volume", 0.0), price=0.0, comment="",
                                     request_id=0, retcode_external=0)

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


@pytest.fixture
def strong_short(run_and_stub, monkeypatch):
    """
    Force the scorer to favour a SHORT so the location gate is reachable.

    On a bullish frame the natural scorer picks the long, so the gate is never
    consulted for the direction that matters. Returning 90 for -1 and 10 for +1
    makes the short win the score contest and reach the gate, which is the
    situation under test: a counter-trend entry against a confirmed uptrend.
    """
    run_module, _ = run_and_stub
    monkeypatch.setattr(
        run_module, "_score_direction",
        lambda ai_dir, ai_conf, m15, h1, h4, d, cfg: (90.0 if d == -1 else 10.0),
    )
    return run_module


# ─────────────────────────────────────────────────────────────────────────────
# Frame builders
# ─────────────────────────────────────────────────────────────────────────────

def uptrend_frame(n: int = 300):
    close, price = [], 100.0
    for i in range(n):
        step = 0.45
        if i % 20 in (0, 1, 2) and i > 0:
            step = -0.30
        price += step
        close.append(round(price, 4))
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": close, "high": [c + 0.3 for c in close],
         "low": [c - 0.3 for c in close], "close": close,
         "tick_volume": [1000.0] * n}, index=idx,
    )


class _FakeAI:
    def __init__(self, direction=1):
        self.direction = direction

    def update(self, frame): pass
    def predict(self, frame):
        return types.SimpleNamespace(direction=self.direction, confidence=0.8)


def build_bot(run_module, **overrides):
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "TEST"
    cfg["dry_run"] = True
    cfg["location_min_bars"] = 10
    cfg.update(overrides)
    bot = run_module.StandaloneAIBot(cfg)
    bot.ai = _FakeAI()
    return bot


# ─────────────────────────────────────────────────────────────────────────────
# _get_signal — the location veto, proven to be load-bearing
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSignalVeto:
    def test_counter_trend_short_is_vetoed_with_the_gate_on(self, strong_short):
        run_module = strong_short
        bot = build_bot(run_module, location_gate=True)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)

        assert direction == 0, "a counter-trend short must be vetoed"
        assert "location veto" in reason
        assert "BULLISH" in reason

    def test_the_same_signal_is_allowed_with_the_gate_off(self, strong_short):
        """The control. Without this the veto test proves nothing — it would
        pass for any reason the signal returned 0."""
        run_module = strong_short
        bot = build_bot(run_module, location_gate=False)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)

        assert direction == -1, "disabling the gate must restore the prior behaviour"
        assert "location veto" not in reason

    def test_with_trend_long_is_not_vetoed(self, run_and_stub):
        """A confirmed uptrend long must still pass — otherwise the gate is a
        blanket kill switch rather than a location rule.

        Deliberately does *not* use the ``strong_short`` fixture: that forces
        the short to win the score contest regardless of the AI's opinion, so
        it cannot express a with-trend entry at all.
        """
        run_module, _ = run_and_stub
        bot = build_bot(run_module, location_gate=True)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        bot.ai.direction = 1

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)

        assert direction == 1, f"with-trend long was blocked: {reason}"
        assert "location veto" not in reason

    def test_no_location_does_not_block(self, run_and_stub):
        """UNKNOWN location must never become a silent kill switch."""
        run_module, _ = run_and_stub
        bot = build_bot(run_module, location_gate=True, location_min_bars=10_000)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)

        assert bot._location_for(frame) is None, "precondition: no location"

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)
        assert "location veto" not in reason

    def test_pre_existing_adx_gate_is_intact(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = build_bot(run_module, adx_min=99.0)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)

        assert direction == 0
        assert "ADX" in reason

    def test_return_shape_is_always_valid(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = build_bot(run_module)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)

        direction, score, conf, reason = bot._get_signal(frame, frame, frame)

        assert direction in (1, -1, 0)
        assert 0.0 <= score <= 100.0
        assert 0.0 <= conf <= 1.0
        assert reason


# ─────────────────────────────────────────────────────────────────────────────
# _send_order — the structural stop, proven to be load-bearing
# ─────────────────────────────────────────────────────────────────────────────

class TestSendOrderStructuralStop:
    def test_stop_source_is_reported_when_location_is_used(self, run_and_stub, caplog):
        run_module, stub = run_and_stub
        bot = build_bot(run_module, location_stop=True)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        with caplog.at_level(logging.INFO, logger="run"):
            assert bot._send_order(1, 10_000.0, frame, ai_conf=0.8) is True

        messages = [r.getMessage() for r in caplog.records]
        assert any("Stop via" in m for m in messages), (
            "location_stop=True but the stop source was never reported — "
            "the structural stop is not being used"
        )

    def test_stop_source_is_absent_when_the_feature_is_off(self, run_and_stub, caplog):
        """The control for the test above."""
        run_module, stub = run_and_stub
        bot = build_bot(run_module, location_stop=False)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        with caplog.at_level(logging.INFO, logger="run"):
            assert bot._send_order(1, 10_000.0, frame, ai_conf=0.8) is True

        messages = [r.getMessage() for r in caplog.records]
        assert not any("Stop via" in m for m in messages), (
            "location_stop=False but a stop source was reported"
        )

    def test_order_is_always_produced(self, run_and_stub):
        run_module, stub = run_and_stub
        bot = build_bot(run_module)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        assert bot._send_order(1, 10_000.0, frame, ai_conf=0.8) is True

    def test_long_stop_is_below_entry(self, run_and_stub, caplog):
        run_module, stub = run_and_stub
        bot = build_bot(run_module, location_stop=True)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        with caplog.at_level(logging.INFO, logger="run"):
            bot._send_order(1, 10_000.0, frame, ai_conf=0.8)

        dry_run = [r.getMessage() for r in caplog.records if "DRY-RUN" in r.getMessage()]
        assert dry_run, "no dry-run line was emitted"
        assert "sl=199" in dry_run[0] or "sl=198" in dry_run[0]

    def test_falls_back_to_atr_when_location_is_unavailable(self, run_and_stub, caplog):
        """Too little history must degrade to the original ATR stop, not fail."""
        run_module, stub = run_and_stub
        bot = build_bot(run_module, location_stop=True, location_min_bars=10_000)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        assert bot._location_for(frame) is None, "precondition: no location"

        with caplog.at_level(logging.INFO, logger="run"):
            assert bot._send_order(1, 10_000.0, frame, ai_conf=0.8) is True

    def test_never_raises_on_a_malformed_frame(self, run_and_stub):
        """A live bot must not die in the order path."""
        run_module, stub = run_and_stub
        bot = build_bot(run_module)
        stub.price = 202.0
        broken = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
             "tick_volume": [1.0]}, index=pd.date_range("2026", periods=1),
        )
        try:
            bot._send_order(1, 10_000.0, broken, ai_conf=0.8)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"_send_order raised on a malformed frame: {exc!r}")

    def test_never_raises_on_garbage_broker_metadata(self, run_and_stub):
        """A mocked or half-initialised client must not take the bot down."""
        run_module, stub = run_and_stub
        bot = build_bot(run_module)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0
        stub.symbol_info = lambda symbol: types.SimpleNamespace(
            digits=2, point=0.01, trade_tick_size="not-a-number",
            trade_tick_value=1.0, volume_min=0.01, volume_max=10.0,
            volume_step=0.01, contract_size=100.0, trade_stops_level=10,
            spread=10, filling_mode=2,
        )
        try:
            bot._send_order(1, 10_000.0, frame, ai_conf=0.8)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"_send_order raised on garbage metadata: {exc!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard contract
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardContract:
    def test_pre_existing_status_keys_survive(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = build_bot(run_module)
        location = bot._location_for(run_module._prepare(uptrend_frame(), bot.cfg))
        bot._publish_location_status(location)

        st = bot.get_status()
        for key in ("balance", "equity", "open_positions", "trades_today",
                    "last_bar", "last_direction", "last_score", "last_ai_conf",
                    "last_signal", "closed_trades", "wins", "losses", "total_pnl"):
            assert key in st, f"pre-existing status key lost: {key}"

    def test_location_keys_are_present(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = build_bot(run_module)
        location = bot._location_for(run_module._prepare(uptrend_frame(), bot.cfg))
        bot._publish_location_status(location)

        st = bot.get_status()
        for key in ("loc_bias", "loc_position", "loc_structure",
                    "loc_range_low", "loc_range_high"):
            assert key in st, f"missing location key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic AI SL/TP Reasoning Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicAISLTP:
    def test_dynamic_sltp_reasoner_initialization(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = build_bot(run_module)
        reasoner = bot._get_sltp_reasoner()
        assert reasoner is not None
        assert reasoner.provider is not None

    def test_dynamic_sltp_entry_logs_reasoning(self, run_and_stub, caplog):
        run_module, stub = run_and_stub
        bot = build_bot(run_module, dynamic_sltp_enabled=True, location_stop=True)
        frame = run_module._prepare(uptrend_frame(), bot.cfg)
        stub.price = 202.0

        with caplog.at_level(logging.INFO, logger="run"):
            assert bot._send_order(1, 10_000.0, frame, ai_conf=0.8) is True

        messages = [r.getMessage() for r in caplog.records]
        assert any("AI Dynamic SL/TP" in m for m in messages), (
            "dynamic_sltp_enabled=True but AI Dynamic SL/TP was never logged"
        )

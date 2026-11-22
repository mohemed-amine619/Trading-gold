"""
tests/unit/test_run_py_withdrawal_peak.py — prove run.py lowers the drawdown
peak after an external withdrawal, and NEVER lowers it for a losing trade.

Observed live (2026-11-22): a 1001.30 withdrawal left peak=1504.37 against
equity=503.07, which reads as a 66.6% total drawdown against a 20% ceiling and
hard-halted all four symbols until a manual POST /api/config/reset-halt.

The safety-critical direction is the second one. A loss and a withdrawal both
lower equity; only the broker's balance ledger tells them apart. Rebasing the
peak for a loss would erase the drawdown the ceiling exists to catch.

These tests drive the real StandaloneAIBot.connect() with MetaTrader5 stubbed.
The pure helper is covered in test_risk_baseline.py; this file guards the wiring.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from datetime import date, datetime, timedelta

import pytest

from core.risk.state import RiskState, load_state, save_state, state_path


DEAL_TYPE_BUY = 0
DEAL_TYPE_BALANCE = 2


class _Acc:
    def __init__(self, balance=503.07):
        self.login = 261009880
        self.balance = balance
        self.equity = balance
        self.margin = 0.0
        self.margin_free = balance


class _Deal:
    """Minimal stand-in for an MT5 deal record."""

    def __init__(self, deal_type, profit, when):
        self.type = deal_type
        self.profit = profit
        self.time = int(when.timestamp())


class StubMT5:
    def __init__(self, balance=503.07, deals=None):
        self.balance = balance
        self._deals = deals or []
        self.TRADE_ACTION_DEAL = 1
        self.DEAL_TYPE_BALANCE = DEAL_TYPE_BALANCE

    def initialize(self, **kwargs):
        return True

    def account_info(self):
        return _Acc(self.balance)

    def positions_get(self, *args, **kwargs):
        return []

    def history_deals_get(self, *args, **kwargs):
        return list(self._deals)

    def last_error(self):
        return 0

    def shutdown(self):
        pass


@pytest.fixture
def run_module(tmp_path, monkeypatch):
    snapshot = dict(os.environ)

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv_stub

    def _load(balance=503.07, deals=None):
        stub = StubMT5(balance, deals)
        mt5_module = types.ModuleType("MetaTrader5")
        for name in dir(stub):
            if not name.startswith("_"):
                setattr(mt5_module, name, getattr(stub, name))
        mt5_module.DEAL_TYPE_BALANCE = DEAL_TYPE_BALANCE
        mt5_module.__getattr__ = lambda n: 0
        sys.modules["MetaTrader5"] = mt5_module
        sys.modules.pop("run", None)
        module = importlib.import_module("run")
        module.mt5 = mt5_module
        return module

    monkeypatch.setenv("RISK_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("TRADING_MODE", "paper")
    for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        monkeypatch.setenv(key, "")

    yield _load, tmp_path

    sys.modules.pop("run", None)
    sys.modules.pop("MetaTrader5", None)
    sys.modules["dotenv"] = importlib.import_module("dotenv")
    os.environ.clear()
    os.environ.update(snapshot)


def _bot(run):
    return run.StandaloneAIBot({
        "symbol": "EURUSDm",
        "magic_number": 888999,
        "risk_per_trade": 0.01,
        "enable_notifications": False,
    })


def _write_state(folder, *, peak, marker_age_hours=24, halted=None):
    """Risk state with an already-established balance-deal marker."""
    marker = (datetime.now() - timedelta(hours=marker_age_hours)).isoformat(timespec="seconds")
    save_state(state_path(str(folder), "EURUSDm"), RiskState(
        daily_date=date.today().isoformat(),
        daily_start_equity=503.07,
        trade_date=date.today().isoformat(),
        trades_today=0,
        peak_equity=peak,
        halted_reason=halted,
        last_balance_deal_ts=marker,
    ))


def _withdrawal(amount, hours_ago=1):
    return _Deal(DEAL_TYPE_BALANCE, -abs(amount), datetime.now() - timedelta(hours=hours_ago))


def _deposit(amount, hours_ago=1):
    return _Deal(DEAL_TYPE_BALANCE, abs(amount), datetime.now() - timedelta(hours=hours_ago))


def _losing_trade(amount, hours_ago=1):
    return _Deal(DEAL_TYPE_BUY, -abs(amount), datetime.now() - timedelta(hours=hours_ago))


# ─────────────────────────────────────────────────────────────────────────────


def test_the_live_case_a_withdrawal_rebases_the_peak_and_clears_the_halt(run_module):
    load, folder = run_module
    run = load(deals=[_withdrawal(1001.30)])
    _write_state(folder, peak=1504.37,
                 halted="total DD 66.6% >= 20.0% (reset required)")
    bot = _bot(run)

    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(503.07, abs=0.01)
    assert bot._halt_reason is None, "a withdrawal-caused halt must clear itself"


def test_a_losing_trade_never_moves_the_peak(run_module):
    """THE safety test. A loss lowers equity exactly like a withdrawal does,
    but it is not a balance operation, so the drawdown must survive."""
    load, folder = run_module
    run = load(deals=[_losing_trade(1001.30)])
    _write_state(folder, peak=1504.37)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(1504.37), "a loss must not relieve drawdown"


def test_a_deposit_never_moves_the_peak(run_module):
    load, folder = run_module
    run = load(deals=[_deposit(1000.0)])
    _write_state(folder, peak=1504.37)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(1504.37)


def test_the_first_run_establishes_the_marker_without_rebasing(run_module):
    """No stored marker means the whole balance history is unknown. Applying
    it would walk the peak down to today's equity in one step."""
    load, folder = run_module
    run = load(deals=[_withdrawal(1001.30)])
    save_state(state_path(str(folder), "EURUSDm"), RiskState(
        daily_date=date.today().isoformat(),
        daily_start_equity=503.07,
        peak_equity=1504.37,
        last_balance_deal_ts=None,
    ))
    bot = _bot(run)

    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(1504.37), "no retroactive rebase"
    saved = load_state(state_path(str(folder), "EURUSDm"))
    assert saved.last_balance_deal_ts is not None, "marker must be established"


def test_the_same_withdrawal_is_applied_only_once(run_module):
    """Without a persisted marker every restart would re-apply it and walk the
    peak down to equity."""
    load, folder = run_module
    run = load(deals=[_withdrawal(400.0)])
    _write_state(folder, peak=1000.0)

    bot = _bot(run)
    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(600.0)

    bot2 = _bot(run)
    assert bot2.connect() is True
    assert bot2._peak_equity == pytest.approx(600.0), "must not apply twice"


def test_an_unreadable_history_leaves_the_peak_alone(run_module):
    """Fail closed: a terminal that cannot answer must not relieve drawdown."""
    load, folder = run_module
    run = load()
    run.mt5.history_deals_get = lambda *a, **k: None
    _write_state(folder, peak=1504.37)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot._peak_equity == pytest.approx(1504.37)

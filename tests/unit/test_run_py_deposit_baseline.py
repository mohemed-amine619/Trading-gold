"""
tests/unit/test_run_py_deposit_baseline.py — prove run.py actually rebases the
daily baseline on a deposit, and does NOT rebase when a trade was made.

These tests drive the real StandaloneAIBot.connect() with MetaTrader5 stubbed.
The pure helper is covered separately; this file guards the wiring.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from datetime import date

import pytest

from core.risk.state import RiskState, save_state, state_path


class _Acc:
    def __init__(self, balance=500.0):
        self.login = 261009880
        self.balance = balance
        self.equity = balance
        self.margin = 0.0
        self.margin_free = balance


class StubMT5:
    def __init__(self, balance=500.0, open_positions=0):
        self.balance = balance
        self.open_positions = open_positions
        self.TRADE_ACTION_DEAL = 1

    def initialize(self, **kwargs):
        return True

    def account_info(self):
        return _Acc(self.balance)

    def positions_get(self, *args, **kwargs):
        return [object()] * self.open_positions

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

    def _load(balance=500.0, open_positions=0):
        stub = StubMT5(balance, open_positions)
        mt5_module = types.ModuleType("MetaTrader5")
        for name in dir(stub):
            if not name.startswith("_"):
                setattr(mt5_module, name, getattr(stub, name))
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


def _bot(run, cfg_overrides=None):
    cfg = {
        "symbol": "EURUSDm",
        "magic_number": 888999,
        "risk_per_trade": 0.01,
        "enable_notifications": False,
    }
    cfg.update(cfg_overrides or {})
    return run.StandaloneAIBot(cfg)


def _write_state(folder, *, start=390.04, trades=0):
    st = RiskState(
        daily_date=date.today().isoformat(),
        daily_start_equity=start,
        trade_date=date.today().isoformat(),
        trades_today=trades,
    )
    save_state(state_path(str(folder), "EURUSDm"), st)


def test_deposit_on_a_flat_account_rebases_the_baseline(run_module):
    load, folder = run_module
    run = load()
    _write_state(folder)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot.daily_start == pytest.approx(500.0)
    # The rebase is persisted, so a second restart cannot add it again.
    bot2 = _bot(run)
    assert bot2.connect() is True
    assert bot2.daily_start == pytest.approx(500.0)


def test_a_trade_today_blocks_the_rebase(run_module):
    load, folder = run_module
    run = load()
    _write_state(folder, trades=1)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot.daily_start == pytest.approx(390.04)


def test_without_the_helper_no_rebase(run_module, monkeypatch):
    """Mutation guard: the assertion above must fail if the wiring is removed."""
    load, folder = run_module
    run = load()
    monkeypatch.setattr(run, "_deposit_baseline_adjustment", None)
    _write_state(folder)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot.daily_start == pytest.approx(390.04)


def test_an_open_position_blocks_the_rebase(run_module):
    load, folder = run_module
    run = load(open_positions=1)
    _write_state(folder)
    bot = _bot(run)

    assert bot.connect() is True
    assert bot.daily_start == pytest.approx(390.04)

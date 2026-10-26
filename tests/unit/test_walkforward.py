"""Tests for core/backtesting/walkforward.py.

All data is synthetic and deterministic. Nothing here shows that a strategy
has an edge on real markets; it shows the harness counts and gates correctly.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.backtesting import walkforward as wf
from core.backtesting.engine import BacktestEngine
from core.strategy.base import BaseStrategy, MarketRegime, Signal


class AlwaysBuy(BaseStrategy):
    def __init__(self):
        super().__init__(name="always_buy")

    def evaluate(self, df, regime):
        return Signal(direction=1, score=100.0, reason="buy", invalidation="")


class AlwaysSell(BaseStrategy):
    def __init__(self):
        super().__init__(name="always_sell")

    def evaluate(self, df, regime):
        return Signal(direction=-1, score=100.0, reason="sell", invalidation="")


def _uptrend(n: int) -> pd.DataFrame:
    """Steady rise of 0.1 per bar. Long wins reach TP; shorts reach SL."""
    close = 1000.0 + 0.1 * np.arange(n)
    index = [datetime(2024, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=pd.DatetimeIndex(index),
    )


# ── window splitting ──────────────────────────────────────────────────────────

def test_windows_are_consecutive_and_cover_every_bar():
    windows = wf.out_of_sample_windows(1003, 4, 100)

    assert windows[0][0] == 0
    assert windows[-1][1] == 1003
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end == next_start, "windows must neither overlap nor leave gaps"
    assert all(end - start >= 100 for start, end in windows)


def test_too_little_data_is_refused_not_thinned():
    with pytest.raises(ValueError):
        wf.out_of_sample_windows(350, 4, 100)


def test_window_count_must_be_positive():
    with pytest.raises(ValueError):
        wf.out_of_sample_windows(1000, 0, 10)


# ── per-window evaluation ─────────────────────────────────────────────────────

def test_profitable_strategy_passes_and_losing_strategy_fails():
    df = _uptrend(1200)
    gate = wf.GateRule(min_trades=1, min_profit_factor=1.2)
    windows = wf.out_of_sample_windows(len(df), 4, 300)

    good = [wf.evaluate_window(df, AlwaysBuy(), s, e, gate) for s, e in windows]
    bad = [wf.evaluate_window(df, AlwaysSell(), s, e, gate) for s, e in windows]

    assert all(w.passed for w in good)
    assert not any(w.passed for w in bad)
    assert all(w.net_profit < 0 for w in bad)


def test_trades_entering_during_warmup_are_not_counted():
    df = _uptrend(1200)
    gate = wf.GateRule(min_trades=0, min_profit_factor=0.0)
    start, end = 600, 900
    warmup = 200

    result = wf.evaluate_window(df, AlwaysBuy(), start, end, gate, warmup=warmup)

    # Reproduce the engine run. Only entries inside the window count; the one
    # still open at the window's end is closed there and counted too.
    engine = BacktestEngine()
    engine.run(df.iloc[start - warmup:end], AlwaysBuy())
    expected = [t for t in engine.trades if t.entry_time >= df.index[start]]
    assert result.trades == len(expected)
    assert result.trades >= 1


def test_position_open_at_window_end_is_counted_not_dropped():
    df = _uptrend(1200)
    gate = wf.GateRule(min_trades=0, min_profit_factor=0.0)

    # A 200-bar hold cannot finish inside a 300-bar window that starts at 300,
    # so every window must still report at least one trade.
    windows = wf.out_of_sample_windows(len(df), 4, 300)
    results = [wf.evaluate_window(df, AlwaysBuy(), s, e, gate) for s, e in windows]

    assert all(r.trades >= 1 for r in results), [r.trades for r in results]


def test_window_outside_the_data_is_rejected():
    df = _uptrend(500)
    with pytest.raises(ValueError):
        wf.evaluate_window(df, AlwaysBuy(), 0, 900, wf.GateRule())


# ── report verdict ────────────────────────────────────────────────────────────

def _result(passed: bool) -> wf.WindowResult:
    return wf.WindowResult(
        start=0, end=10, trades=10, win_rate=0.5,
        net_profit=1.0, profit_factor=1.5, passed=passed,
    )


def test_report_passes_only_when_every_window_passes():
    assert wf.StrategyReport("s", [_result(True), _result(True)]).passed
    assert not wf.StrategyReport("s", [_result(True), _result(False)]).passed


def test_report_with_no_windows_never_passes():
    assert not wf.StrategyReport("s", []).passed


# ── ensemble adapter ──────────────────────────────────────────────────────────

class _FakeEnsemble:
    def __init__(self, signal):
        self.signal = signal
        self.calls = 0

    def evaluate_all(self, df):
        self.calls += 1
        return self.signal


def test_ensemble_adapter_delegates_to_the_ensemble():
    expected = Signal(direction=-1, score=77.0, reason="x", invalidation="")
    fake = _FakeEnsemble(expected)
    adapter = wf.EnsembleStrategy(fake)

    got = adapter.evaluate(_uptrend(5), MarketRegime.RANGING)

    assert got is expected
    assert fake.calls == 1
    assert adapter.name == "ensemble"

"""
core/backtesting/walkforward.py — Out-of-sample evaluation of strategies.

Why this exists
---------------
A strategy that looks profitable over the whole history has usually been
fitted to it. This module splits history into consecutive windows and reports
each window separately, so a result counts only where it was not tuned.

Each window is traded with ``BacktestEngine`` on a slice that starts
``warmup`` bars early, so indicators such as ``ema200`` are ready when the
window begins. Only trades that *enter* inside the window are counted.

Limits (also stated in docs/PROJECT_STATUS.md)
----------------------------------------------
* ``BacktestEngine`` passes a fixed ``TRENDING`` regime to single strategies,
  and uses fixed sizing and a fixed 2×ATR stop. A result here measures the
  signal logic under those assumptions, not the live bot's risk management.
* Nothing is learned from the windows, so there is no training split. Any
  future model that *is* trained must be fitted before each window only.
* A pass means the gate's thresholds were met on the data supplied. It is
  not evidence of future profit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from core.backtesting.engine import BacktestEngine
from core.strategy.base import BaseStrategy, MarketRegime, Signal
from core.strategy.ensemble import StrategyEnsemble


class EnsembleStrategy(BaseStrategy):
    """Adapts ``StrategyEnsemble`` to the single-strategy backtest interface."""

    def __init__(self, ensemble: StrategyEnsemble, name: str = "ensemble"):
        super().__init__(name=name)
        self._ensemble = ensemble

    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        return self._ensemble.evaluate_all(df)


@dataclass(frozen=True)
class GateRule:
    """Thresholds every window must meet for a strategy to pass."""

    min_trades: int = 10
    min_profit_factor: float = 1.2
    min_win_rate: float = 0.0


@dataclass(frozen=True)
class WindowResult:
    start: int
    end: int
    trades: int
    win_rate: float
    net_profit: float
    profit_factor: float
    passed: bool


@dataclass(frozen=True)
class StrategyReport:
    name: str
    windows: List[WindowResult]

    @property
    def passed(self) -> bool:
        """True only when there is at least one window and every window passed."""
        return bool(self.windows) and all(w.passed for w in self.windows)


def out_of_sample_windows(n_bars: int, n_windows: int, min_window: int) -> List[Tuple[int, int]]:
    """Split ``n_bars`` into ``n_windows`` consecutive half-open ranges.

    Ranges do not overlap and together cover every bar. Raises ``ValueError``
    when the data cannot give each window ``min_window`` bars, rather than
    quietly producing thin windows.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be at least 1")
    if n_bars < n_windows * min_window:
        raise ValueError(
            f"{n_bars} bars cannot give {n_windows} windows of at least {min_window} bars"
        )
    size = n_bars // n_windows
    bounds = [i * size for i in range(n_windows)] + [n_bars]
    # The last window absorbs the remainder so no bar is dropped.
    return [(bounds[i], bounds[i + 1]) for i in range(n_windows)]


def _window_metrics(trades: Sequence, gate: GateRule) -> Dict[str, float]:
    closed = [t for t in trades if not t.is_open]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(abs(t.pnl) for t in losses)
    count = len(closed)
    win_rate = len(wins) / count if count else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    return {
        "trades": count,
        "win_rate": win_rate,
        "net_profit": gross_profit - gross_loss,
        "profit_factor": profit_factor,
    }


def evaluate_window(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    start: int,
    end: int,
    gate: GateRule,
    warmup: int = 200,
) -> WindowResult:
    """Trade one window and apply the gate. Only entries inside it count."""
    if not 0 <= start < end <= len(df):
        raise ValueError(f"window [{start}, {end}) is outside {len(df)} bars")
    lo = max(0, start - warmup)
    engine = BacktestEngine()
    engine.run(df.iloc[lo:end], strategy)

    window_start = df.index[start]
    inside = [t for t in engine.trades if t.entry_time >= window_start]
    # A position still open at the window's end is closed at its last close.
    # Left open, it would be dropped from the count, so a slow-holding strategy
    # would show zero trades in every window. Closing here keeps each window
    # judged on its own entries and independent of the next window.
    last_time, last_close = df.index[end - 1], float(df["close"].iloc[end - 1])
    for trade in inside:
        if trade.is_open:
            trade.close(last_time, last_close)
    m = _window_metrics(inside, gate)
    passed = (
        m["trades"] >= gate.min_trades
        and m["profit_factor"] >= gate.min_profit_factor
        and m["win_rate"] >= gate.min_win_rate
    )
    return WindowResult(
        start=start,
        end=end,
        trades=m["trades"],
        win_rate=m["win_rate"],
        net_profit=m["net_profit"],
        profit_factor=m["profit_factor"],
        passed=passed,
    )


def evaluate_strategies(
    df: pd.DataFrame,
    strategies: Sequence[BaseStrategy],
    n_windows: int = 4,
    min_window: int = 300,
    gate: GateRule = GateRule(),
    warmup: int = 200,
) -> List[StrategyReport]:
    """Run every strategy on the same out-of-sample windows of ``df``.

    ``df`` must already carry indicators (see ``core.market.pipeline``) and
    have a datetime index. Strategies are evaluated on identical windows so the
    reports are comparable.
    """
    windows = out_of_sample_windows(len(df), n_windows, min_window)
    reports = []
    for strategy in strategies:
        results = [
            evaluate_window(df, strategy, s, e, gate, warmup=warmup)
            for s, e in windows
        ]
        reports.append(StrategyReport(name=strategy.name, windows=results))
    return reports

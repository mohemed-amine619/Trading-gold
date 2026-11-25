"""
core/strategy/panel.py — every strategy of the library, read on one timeframe.

The panel is OBSERVABILITY, not a trade signal. For each strategy it reports the
call on the newest bar of the frame it is given, how many independent calls the
strategy made over the replay window, their edge under the barrier rule the live
vote uses, and whether that edge clears the proof gate *for the number of
strategies on the panel* (Bonferroni over all of them, so ``proven`` here is
stricter than the live vote's, which only tests ``SCALP_STRATEGIES``).

Nothing here adds points to the confluence score. Only the gated vote in
``core/strategy/scalp.py`` (``scalp_support``) may do that.

Frames are expected to hold CLOSED bars only (the bot's fetch drops the forming
candle); the reading is taken on the last row.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.strategy.library import CATALOGUE, CATEGORIES, CLASSIC
from core.strategy.scalp import GENERATORS, ScalpConfig, ScalpEngine, prove

DEFAULT_CHECK_BARS = 4000

__all__ = ["DEFAULT_CHECK_BARS", "StrategyReading", "StrategyPanel", "PanelBuilder", "build_panel",
           "strategy_category", "unavailable_entries"]


def _category_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in CATALOGUE:
        if e.kind == "signal" and e.target and e.target not in out:
            out[e.target] = e.category
    for name in CLASSIC:
        out.setdefault(name, "classic")
    return out


_CATEGORY = _category_map()


def strategy_category(name: str) -> str:
    return _CATEGORY.get(name, "other")


def unavailable_entries() -> Tuple[Tuple[str, str, str], ...]:
    """(category, label, note) for catalogue entries that cannot be computed from MT5 bars."""
    return tuple((e.category, e.label, e.note) for e in CATALOGUE if e.kind == "unavailable")


@dataclass
class StrategyReading:
    name: str
    category: str
    timeframe: str
    signal: int = 0                       # -1 / 0 / +1 on the newest bar
    bars_since_signal: Optional[int] = None
    calls: int = 0                        # independent (horizon-thinned) calls in the replay window
    edge: Optional[float] = None          # (wins - losses) / calls
    z: Optional[float] = None             # edge * sqrt(independent calls)
    proven: bool = False
    available: bool = True
    error: Optional[str] = None


@dataclass
class StrategyPanel:
    timeframe: str
    as_of: Optional[str] = None
    readings: List[StrategyReading] = field(default_factory=list)
    long_count: int = 0
    short_count: int = 0
    flat_count: int = 0
    net_breadth: float = 0.0
    weighted_breadth: float = 0.0
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    required_z: float = 0.0
    n_tested: int = 0
    note: str = ("information, not a trade signal: only the gated vote "
                 "(SCALP_STRATEGIES) can add score points")

    @property
    def proven(self) -> List[str]:
        return [r.name for r in self.readings if r.proven]

    def summary(self) -> str:
        return (f"strategies {self.timeframe}: {self.long_count} long / {self.short_count} short / "
                f"{self.flat_count} flat, breadth {self.net_breadth:+.2f}, "
                f"proven: {', '.join(self.proven) or 'none'}")

    def top(self, n: int = 15) -> List[StrategyReading]:
        avail = [r for r in self.readings if r.available and r.error is None]
        return sorted(avail, key=lambda r: -abs(r.z or 0.0))[:n]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proven"] = self.proven
        return d


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class PanelBuilder:
    """Holds one engine per configuration and caches the last panel per timeframe
    (keyed on the OHLC data), so the live loop can call it every second."""

    def __init__(self, cfg: Optional[ScalpConfig] = None, strategies: Tuple[str, ...] = ("all",)):
        # Default replay window = what the bot sees for gold (HISTORY_CANDLES 4500 - margin).
        base = cfg or ScalpConfig(check_bars=DEFAULT_CHECK_BARS)
        self.cfg = replace(base, strategies=tuple(strategies), auto_z=True)
        self.engine = ScalpEngine(self.cfg)
        self._cache: Dict[str, Tuple[tuple, StrategyPanel]] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        return self.engine.names

    def signals(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Signals per strategy plus the error text of any generator that raised."""
        from core.strategy.analog import _atr
        atr = _atr(df)
        sigs: Dict[str, np.ndarray] = {}
        errors: Dict[str, str] = {}
        for name in self.names:
            try:
                sigs[name] = np.asarray(GENERATORS[name](df, atr, self.cfg), dtype=np.int8)
            except Exception as exc:              # one broken rule never stops the panel
                sigs[name] = np.zeros(len(df), dtype=np.int8)
                errors[name] = repr(exc)[:200]
        return sigs, errors

    def build(self, df: pd.DataFrame, timeframe: str) -> StrategyPanel:
        panel = StrategyPanel(timeframe=timeframe, required_z=float(self.engine.min_z),
                              n_tested=len(self.names))
        if not isinstance(df, pd.DataFrame) or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return panel
        need = [c for c in ("open", "high", "low", "close") if c not in df.columns]
        if need:
            return panel
        key = (len(df), df.index[0], df.index[-1],
               hash(df[["open", "high", "low", "close"]].to_numpy(dtype=float).tobytes()))
        hit = self._cache.get(timeframe)
        if hit is not None and hit[0] == key:
            return hit[1]

        panel.as_of = str(df.index[-1])
        sigs, errors = self.signals(df)
        outcomes = self.engine.outcomes(df)
        gate = replace(self.cfg, min_z=self.engine.min_z)
        n = len(df)
        enough = n >= gate.horizon + 60
        weights_long = weights_short = 0.0
        cats: Dict[str, Dict[str, int]] = {c: {"long": 0, "short": 0} for c in ("classic",) + CATEGORIES}
        for name in self.names:
            sig = sigs[name]
            r = StrategyReading(name=name, category=strategy_category(name), timeframe=timeframe)
            r.error = errors.get(name)
            r.signal = int(sig[-1]) if n else 0
            nz = np.flatnonzero(sig)
            r.bars_since_signal = int(n - 1 - nz[-1]) if nz.size else None
            if enough and r.error is None:
                proof = prove(sig, outcomes, gate)
                r.calls = int(proof.get("n_independent") or 0)
                r.edge = _finite(proof.get("edge"))
                r.z = (r.edge * math.sqrt(r.calls)) if (r.edge is not None and r.calls) else None
                r.proven = bool(proof.get("proven"))
            panel.readings.append(r)
            if r.error is not None:
                continue
            bucket = cats.setdefault(r.category, {"long": 0, "short": 0})
            if r.signal > 0:
                panel.long_count += 1
                bucket["long"] += 1
                weights_long += max(0.0, r.z or 0.0)
            elif r.signal < 0:
                panel.short_count += 1
                bucket["short"] += 1
                weights_short += max(0.0, r.z or 0.0)
            else:
                panel.flat_count += 1
        for cat, label, note in unavailable_entries():
            panel.readings.append(StrategyReading(name=label, category=cat, timeframe=timeframe,
                                                  available=False, error=None))
        active = panel.long_count + panel.short_count
        panel.net_breadth = (panel.long_count - panel.short_count) / max(1, active)
        wt = weights_long + weights_short
        panel.weighted_breadth = (weights_long - weights_short) / wt if wt > 0 else 0.0
        panel.by_category = {k: v for k, v in cats.items() if v["long"] or v["short"]}
        self._cache[timeframe] = (key, panel)
        return panel


_DEFAULT_BUILDERS: Dict[tuple, PanelBuilder] = {}


def build_panel(df: pd.DataFrame, timeframe: str, cfg: Optional[ScalpConfig] = None,
                strategies: Tuple[str, ...] = ("all",)) -> StrategyPanel:
    """One-shot helper; reuses a builder per (config, strategies) so repeated calls are cached."""
    key = (cfg, tuple(strategies))
    b = _DEFAULT_BUILDERS.get(key)
    if b is None:
        b = _DEFAULT_BUILDERS[key] = PanelBuilder(cfg, strategies)
    return b.build(df, timeframe)

"""
core/market/series.py — Shared OHLCV helpers.

Small, deliberately. Every function here exists because the same logic was
being written three times with three slightly different bugs.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np
import pandas as pd


def valid_tick_size(value: Any) -> Optional[float]:
    """
    Return ``tick_size`` only if it is a usable positive number, else ``None``.

    Broker metadata is untrusted input. A mocked adapter, a partially
    initialised stub, or a genuinely odd broker value can hand back a
    ``MagicMock``, ``None``, ``0``, ``NaN``, or a negative number, and that
    value flows straight into arithmetic — ``price / tick_size`` raises
    ``TypeError``, and every ``max(tolerance, tick_size)`` silently becomes
    meaningless.

    Failing closed to ``None`` is the correct outcome: prices are then left
    unsnapped, which is honest and slightly less precise, rather than raising
    inside the analysis or corrupting a tolerance.

    Note the explicit type check. ``float(obj)`` is not a validity test:
    ``unittest.mock.MagicMock`` implements ``__float__`` and happily returns
    ``1.0``, so a mocked MT5 client would otherwise pass as a legitimate
    increment and snap every level to a whole unit. Only real numeric types
    (including numpy scalars) and parseable strings are accepted.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None

    if not math.isfinite(number) or number <= 0:
        return None
    return number


def resolve_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Current ATR, preferring the pipeline's own value.

    ``market.pipeline.add_indicators`` already attaches a proper Wilder ATR, so
    that is used when present. The fallback for un-enriched frames averages the
    last ``period`` true ranges rather than reading the most recent one.

    That averaging is not cosmetic. A single spike bar has an outsized true
    range, and taking only the latest value lets one bar redefine "the
    volatility of this market". The consequence is not a cosmetic one either:
    range-width and cluster-tolerance checks are all expressed as multiples of
    ATR, so a one-bar inflation can make a perfectly tradable range fail the
    "is this wide enough to be a range" test and silently return
    ``UNDEFINED`` location — refusing to trade a market that is fine.

    Returns:
        A positive ATR, or ``None`` when no scale can be established. ``None``
        means "unknown", and callers must abstain rather than substitute.
    """
    if df is None or df.empty:
        return None

    if "atr" in df.columns:
        series = df["atr"].dropna()
        if not series.empty:
            last = float(series.iloc[-1])
            return last if last > 0 else None

    if not {"high", "low", "close"}.issubset(df.columns) or len(df) < 2:
        return None

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    window = tr.dropna().tail(period)
    if window.empty:
        return None
    value = float(window.mean())
    return value if value > 0 else None
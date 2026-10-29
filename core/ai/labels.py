"""
core/ai/labels.py — first-barrier (triple-barrier style) direction labels.

The question these labels answer
--------------------------------
"Starting from this bar, which side does price break first by a full stop
distance?"  The barrier is ``barrier_mult × ATR`` above and below the close,
which is the same distance the bot's stop uses. A label is therefore about a
move big enough to matter to a trade, and not about a few cents of drift.

How this differs from the close-only label it can replace
---------------------------------------------------------
* The window is first-touch, not best-close. A bar that drops 1.3 ATR and then
  rallies 0.4 ATR is labelled DOWN, because the stop side was touched first.
  The old label called it UP.
* Wicks count. Touches use ``high`` and ``low``, so an intrabar excursion that
  a trader's stop would have hit is seen.
* A bar where both barriers are touched inside one candle is dropped (NaN).
  The order of touches inside a single bar is unknowable from OHLC, so any
  label would be a guess.
* A bar where neither barrier is touched within the horizon is dropped (NaN).
  It is not a direction.

Pure numpy. No pandas, no broker SDK.
"""

from __future__ import annotations

import numpy as np


def first_barrier_labels(high, low, close, atr, barrier_mult: float,
                         horizon: int) -> np.ndarray:
    """Return a float array: 1.0 = up barrier first, 0.0 = down barrier first,
    NaN = unresolved (timeout, ambiguous candle, missing data).

    All four inputs must have the same length. ``atr`` values that are NaN or
    not strictly positive produce NaN for that bar.
    """
    if barrier_mult <= 0:
        raise ValueError("barrier_mult must be positive")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    atr = np.asarray(atr, dtype=float)
    n = len(close)
    if not (len(high) == len(low) == len(atr) == n):
        raise ValueError("high, low, close and atr must have the same length")

    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        a = atr[i]
        c = close[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(c)):
            continue
        barrier = barrier_mult * a
        up, down = c + barrier, c - barrier
        for j in range(i + 1, i + 1 + horizon):
            hit_up = high[j] >= up
            hit_down = low[j] <= down
            if hit_up and hit_down:
                break                      # ambiguous single candle: drop
            if hit_up:
                labels[i] = 1.0
                break
            if hit_down:
                labels[i] = 0.0
                break
    return labels

"""
core/market/swings.py — Swing point detection.

The foundation of every location question. Before the bot can say "price is at
a discount", "below resistance" or "that trend is broken" it needs an
unambiguous list of the pivots the market actually turned from.

A swing point is a bar whose extreme is the most extreme of a symmetric window
around it (a fractal). Two properties matter and are enforced here:

**Confirmation.** A swing at bar *i* can only be known once *k* bars to its
right have closed. The final *k* bars of any series are therefore
*unconfirmed* and may repaint. They are reported with ``confirmed=False`` and
are excluded from structural decisions by default, because a level that can
move is not a level you can place a stop behind.

**Determinism on ties.** Flat markets produce runs of equal highs. "Is this the
max" alone would tag every bar in the run. The earliest bar achieving the
window maximum wins, so a plateau yields exactly one swing, always the same one.

This module is pure: no broker SDK, no I/O, no global state. It is the easiest
surface in the codebase to test, so it should stay that way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SwingKind(str, Enum):
    """Which side of the market a swing point is on."""

    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class SwingPoint:
    """
    A confirmed (or explicitly unconfirmed) pivot.

    Attributes:
        index: Positional index into the source frame. Use this, not the label,
            to align against other positional data.
        time: Bar timestamp, for display and for aligning to other timeframes.
        price: The extreme — ``high`` for a swing high, ``low`` for a swing low.
        kind: ``HIGH`` or ``LOW``.
        confirmed: ``False`` for the trailing bars that lack enough right-hand
            context to be final. Never trade against an unconfirmed swing.
        width: Bars between the pivot and the confirming bars either side.
            A wider swing is a more significant one; callers may weight by it.
    """

    index: int
    time: pd.Timestamp
    price: float
    kind: SwingKind
    confirmed: bool = True
    width: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "time": self.time,
            "price": self.price,
            "kind": self.kind.value,
            "confirmed": self.confirmed,
            "width": self.width,
        }


def find_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    include_unconfirmed: bool = True,
) -> List[SwingPoint]:
    """
    Detect fractal swing highs and lows.

    Args:
        df: OHLCV frame. Requires ``high`` and ``low`` columns.
        left: Bars required to the *left* of a pivot. The leading ``left``
            bars therefore cannot host a confirmed pivot.
        right: Bars required to the *right* of a pivot. The trailing ``right``
            bars cannot host a confirmed pivot.
        include_unconfirmed: Include trailing pivots lacking right-hand
            confirmation. Useful for display; unsafe for stops.

    Returns:
        Swings in chronological order (ascending ``index``).

    Notes:
        ``left`` and ``right`` are kept separate on purpose. Symmetric values
        are conventional, but asymmetric values are meaningful: a large ``left``
        with a small ``right`` finds major pivots that are confirmed quickly,
        which suits a position stop. Widening ``left`` and narrowing ``right``
        is the opposite trade-off and suits structural analysis.
    """
    if df is None or df.empty:
        return []
    if left < 0 or right < 0:
        raise ValueError(f"left/right must be non-negative, got left={left}, right={right}")
    if left == 0 and right == 0:
        # A zero-width window makes every bar its own pivot. Refuse rather than
        # return a meaningless list that downstream code would treat as signal.
        raise ValueError("left and right cannot both be zero: window is empty")

    for col in ("high", "low"):
        if col not in df.columns:
            raise KeyError(f"swings require a '{col}' column; got {list(df.columns)}")

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)
    times = df.index

    swings: List[SwingPoint] = []

    # A pivot at i needs bars [i-left, i+right] to exist and be closed.
    lo_bound = left
    hi_bound = n - 1 - right

    for i in range(lo_bound, hi_bound + 1):
        w_hi = highs[i - left : i + right + 1]
        w_lo = lows[i - left : i + right + 1]

        # np.argmax returns the FIRST maximal index, which resolves plateaus to
        # a single deterministic pivot instead of tagging every equal bar.
        hi_pos = i - left + int(np.argmax(w_hi))
        lo_pos = i - left + int(np.argmin(w_lo))

        if hi_pos == i and highs[i] > w_hi.min():
            # > min (not >= max) rejects a flat plateau where every bar ties:
            # the whole run then shares max==min and no pivot is invented.
            swings.append(
                SwingPoint(
                    index=i,
                    time=times[i],
                    price=float(highs[i]),
                    kind=SwingKind.HIGH,
                    confirmed=True,
                    width=left + right,
                )
            )

        if lo_pos == i and lows[i] < w_lo.max():
            swings.append(
                SwingPoint(
                    index=i,
                    time=times[i],
                    price=float(lows[i]),
                    kind=SwingKind.LOW,
                    confirmed=True,
                    width=left + right,
                )
            )

    if include_unconfirmed:
        swings.extend(_tail_swings(df, left=left, right=right, already=swings, n=n))

    swings.sort(key=lambda s: (s.index, s.kind.value))
    return swings


def _tail_swings(
    df: pd.DataFrame,
    left: int,
    right: int,
    already: Sequence[SwingPoint],
    n: int,
) -> List[SwingPoint]:
    """
    Provisional pivots in the trailing bars that lack right-hand confirmation.

    These are computed with whatever right context exists, so they are the best
    current guess and may revise. They carry ``confirmed=False``.
    """
    if n <= left:
        return []

    seen = {(s.index, s.kind) for s in already}
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df.index

    out: List[SwingPoint] = []
    for i in range(max(left, n - right), n):
        w_hi = highs[i - left : n]
        w_lo = lows[i - left : n]
        if len(w_hi) == 0 or len(w_lo) == 0:
            continue

        if (i, SwingKind.HIGH) not in seen:
            hi_pos = i - left + int(np.argmax(w_hi))
            if hi_pos == i and highs[i] > w_hi.min():
                out.append(
                    SwingPoint(
                        index=i,
                        time=times[i],
                        price=float(highs[i]),
                        kind=SwingKind.HIGH,
                        confirmed=False,
                        width=left + (n - 1 - i),
                    )
                )
        if (i, SwingKind.LOW) not in seen:
            lo_pos = i - left + int(np.argmin(w_lo))
            if lo_pos == i and lows[i] < w_lo.max():
                out.append(
                    SwingPoint(
                        index=i,
                        time=times[i],
                        price=float(lows[i]),
                        kind=SwingKind.LOW,
                        confirmed=False,
                        width=left + (n - 1 - i),
                    )
                )
    return out


def last_confirmed(swings: Sequence[SwingPoint], kind: Optional[SwingKind] = None) -> Optional[SwingPoint]:
    """Most recent confirmed swing, optionally of one kind."""
    for s in reversed(swings):
        if not s.confirmed:
            continue
        if kind is None or s.kind is kind:
            return s
    return None


def alternating(swings: Sequence[SwingPoint]) -> List[SwingPoint]:
    """
    Collapse a swing list to true alternation: HIGH, LOW, HIGH, LOW...

    Raw fractals can emit consecutive highs when a plateau or a double top
    produces near-identical pivots. Structure maths assumes alternation —
    "higher high than the previous high" is meaningless if the previous swing
    is also a high — so consumers should pass through here first.

    Within a run of the same kind, the most extreme pivot is kept: the highest
    high of consecutive highs, the lowest low of consecutive lows. That is the
    level that actually mattered.
    """
    ordered = sorted(swings, key=lambda s: s.index)
    out: List[SwingPoint] = []
    for s in ordered:
        if out and out[-1].kind is s.kind:
            keep_new = (
                s.kind is SwingKind.HIGH and s.price > out[-1].price
            ) or (s.kind is SwingKind.LOW and s.price < out[-1].price)
            if keep_new:
                out[-1] = s
            continue
        out.append(s)
    return out
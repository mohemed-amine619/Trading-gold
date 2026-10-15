"""
core/market/structure.py — Market structure: what the market is doing.

Swing points tell us *where* the turns were. Structure tells us *what the
market is doing with those turns*, and that is a different question from what a
moving average says.

The vocabulary, defined precisely here because it is used loosely everywhere:

- **Bias** — the direction implied by the sequence of swing extremes.
  Two higher highs and two higher lows is BULLISH. Two lower highs and two
  lower lows is BEARISH. Anything mixed is RANGING: a genuine "nobody knows",
  never silently upgraded to a directional bias.
- **BOS (Break of Structure)** — a close *beyond the last swing in the
  direction the trend was already going*. Continuation. In an uptrend, price
  closing above the most recent swing high.
- **CHoCH (Change of Character)** — the *first* close against the prevailing
  bias. This is the warning that the prevailing trend is over. Its importance
  is that it is the earliest structural evidence available, earlier than any
  cross, divergence or level break on a chart.

Breaks are judged on **body close, not wick**. A wick through a level is
liquidity being taken (see ``core/market/liquidity.py``), which is a different
event and is reported there. Conflating the two is how structure-based systems
end up taking a "bullish break" on a bar that closed 40 points lower.

Everything here is derived from confirmed swings only. A break through a pivot
that can still repaint is not a break.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from core.market.swings import SwingKind, SwingPoint, alternating, find_swings

logger = logging.getLogger(__name__)


class StructureBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    # Fewer than two confirmed swings of each kind. Refuse to guess.
    UNDEFINED = "UNDEFINED"


class BreakKind(str, Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"


@dataclass(frozen=True)
class StructureEvent:
    """A confirmed close through a structural level."""

    kind: BreakKind
    direction: int          # +1 bullish break, -1 bearish break
    level: float            # the swing price that was broken
    break_index: int
    break_time: pd.Timestamp
    break_close: float
    swing_index: int
    swing_time: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "direction": self.direction,
            "level": self.level,
            "break_index": self.break_index,
            "break_time": self.break_time,
            "break_close": self.break_close,
            "swing_index": self.swing_index,
            "swing_time": self.swing_time,
        }


@dataclass(frozen=True)
class StructureState:
    """The full structural read of a market at one point in time."""

    bias: StructureBias
    confidence: float
    events: List[StructureEvent] = field(default_factory=list)
    swings: List[SwingPoint] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def last_bos(self) -> Optional[StructureEvent]:
        for e in reversed(self.events):
            if e.kind is BreakKind.BOS:
                return e
        return None

    @property
    def last_choch(self) -> Optional[StructureEvent]:
        for e in reversed(self.events):
            if e.kind is BreakKind.CHOCH:
                return e
        return None

    @property
    def last_event(self) -> Optional[StructureEvent]:
        return self.events[-1] if self.events else None

    def to_dict(self) -> dict:
        return {
            "bias": self.bias.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "events": [e.to_dict() for e in self.events],
            "swings": [s.to_dict() for s in self.swings],
        }


def analyse_structure(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    lookback: int = 6,
) -> StructureState:
    """
    Derive structural bias and the sequence of BOS/CHoCH events.

    Args:
        df: OHLCV frame with a datetime index.
        left/right: Fractal window for swing detection.
        lookback: Maximum events retained, newest last.

    Returns:
        A ``StructureState``. Bias is ``UNDEFINED`` when there is not enough
        confirmed structure to say anything — never a defaulted guess.
    """
    if df is None or df.empty or "close" not in df.columns:
        return StructureState(StructureBias.UNDEFINED, 0.0, reasons=["No data"])

    swings = alternating(find_swings(df, left=left, right=right, include_unconfirmed=False))
    if len(swings) < 2:
        return StructureState(
            StructureBias.UNDEFINED,
            0.0,
            swings=swings,
            reasons=[f"Only {len(swings)} confirmed swing(s); structure undefined"],
        )

    highs = [s for s in swings if s.kind is SwingKind.HIGH]
    lows = [s for s in swings if s.kind is SwingKind.LOW]

    bias, bias_conf, bias_reasons = _bias_from_swings(highs, lows)

    events = _detect_breaks(df, swings, bias)

    return StructureState(
        bias=bias,
        confidence=bias_conf,
        events=events[-lookback:],
        swings=swings,
        reasons=bias_reasons,
    )


def _bias_from_swings(
    highs: Sequence[SwingPoint], lows: Sequence[SwingPoint]
) -> tuple[StructureBias, float, List[str]]:
    """
    Bias from the last two highs and last two lows.

    Confidence is the fraction of the four comparisons that agree, so a clean
    HH/HL reads high and a one-sided sequence reads low rather than picking a
    side and pretending.
    """
    reasons: List[str] = []
    if len(highs) < 2 or len(lows) < 2:
        return StructureBias.UNDEFINED, 0.0, ["Fewer than two confirmed swings of each kind"]

    bias, conf, reasons = _bias_and_confidence(highs, lows)
    return bias, conf, reasons


def _bias_and_confidence(
    highs: Sequence[SwingPoint], lows: Sequence[SwingPoint]
) -> tuple[StructureBias, float, List[str]]:
    """
    Bias plus the *agreement* of the two comparisons that determine it.

    There are exactly two independent comparisons — the high sequence and the
    low sequence. Dividing by four (the number of booleans) would score a
    textbook HH/HL uptrend at 0.5, since two of the four comparisons are the
    mutually exclusive opposite and are always false. Confidence is therefore
    agreement among the determinate comparisons only.

    A flat pair (previous == current) is *indeterminate*, not bearish, and is
    excluded from the denominator instead of counting against the read.
    """
    if len(highs) < 2 or len(lows) < 2:
        return StructureBias.UNDEFINED, 0.0, ["Fewer than two confirmed swings of each kind"]

    reasons: List[str] = []

    hi_prev, hi_now = highs[-2].price, highs[-1].price
    lo_prev, lo_now = lows[-2].price, lows[-1].price

    hi_dir = 1 if hi_now > hi_prev else (-1 if hi_now < hi_prev else 0)
    lo_dir = 1 if lo_now > lo_prev else (-1 if lo_now < lo_prev else 0)

    determinate = [d for d in (hi_dir, lo_dir) if d != 0]
    if not determinate:
        reasons.append("Highs and lows are unchanged: no structural direction")
        return StructureBias.RANGING, 0.0, reasons

    if hi_dir == 1 and lo_dir == 1:
        reasons.append(
            f"Higher high ({hi_prev:.5f} -> {hi_now:.5f}) and higher low "
            f"({lo_prev:.5f} -> {lo_now:.5f})"
        )
        bias = StructureBias.BULLISH
    elif hi_dir == -1 and lo_dir == -1:
        reasons.append(
            f"Lower high ({hi_prev:.5f} -> {hi_now:.5f}) and lower low "
            f"({lo_prev:.5f} -> {lo_now:.5f})"
        )
        bias = StructureBias.BEARISH
    else:
        parts = []
        if hi_dir == 1:
            parts.append("higher high")
        elif hi_dir == -1:
            parts.append("lower high")
        else:
            parts.append("equal highs")
        if lo_dir == 1:
            parts.append("higher low")
        elif lo_dir == -1:
            parts.append("lower low")
        else:
            parts.append("equal lows")
        reasons.append(f"Mixed structure ({', '.join(parts)}): ranging, no directional edge")
        bias = StructureBias.RANGING

    # Confidence is agreement with the bias actually being reported. Hardcoding
    # "agreement with bullish" would score every clean downtrend 0.0.
    bias_dir = 1 if bias is StructureBias.BULLISH else (-1 if bias is StructureBias.BEARISH else 0)
    if bias_dir == 0:
        return bias, 0.0, reasons

    agree = sum(1 for d in determinate if d == bias_dir)
    return bias, agree / len(determinate), reasons


def _detect_breaks(
    df: pd.DataFrame,
    swings: Sequence[SwingPoint],
    initial_bias: StructureBias,
) -> List[StructureEvent]:
    """
    Walk swings chronologically and find the close that broke each one.

    Bias is recomputed **causally** at each step from the swings seen so far,
    never seeded from the finished series. Seeding with the final bias would
    retro-label the opening breaks of a move using information from its end: an
    uptrend that later reversed would have its first bullish break filed as a
    bearish CHoCH, which is simply wrong.

    ``initial_bias`` is accepted only to document the bias of the whole series;
    classification inside the walk is causal and does not use it.

    Until two highs and two lows have formed, bias is UNDEFINED and a break is
    recorded as BOS — there is no prior character to have changed.
    """
    closes = df["close"].to_numpy(dtype=float)
    times = df.index
    n = len(closes)

    events: List[StructureEvent] = []
    seen_highs: List[SwingPoint] = []
    seen_lows: List[SwingPoint] = []
    bias = StructureBias.UNDEFINED

    for swing in swings:
        # Only look forward. A break must happen after the pivot formed, or the
        # bar that makes the pivot is trivially "beyond" it.
        start = swing.index + 1
        if start >= n:
            if swing.kind is SwingKind.HIGH:
                seen_highs.append(swing)
            else:
                seen_lows.append(swing)
            continue

        if swing.kind is SwingKind.HIGH:
            hit = _first_index_above(closes, start, swing.price)
            direction = 1
        else:
            hit = _first_index_below(closes, start, swing.price)
            direction = -1

        if hit is not None:
            # Classify against the bias in force *before* this break, derived
            # only from swings that already existed at that point in the series.
            if bias is StructureBias.UNDEFINED:
                kind = BreakKind.BOS
            elif direction == 1 and bias is StructureBias.BEARISH:
                kind = BreakKind.CHOCH
            elif direction == -1 and bias is StructureBias.BULLISH:
                kind = BreakKind.CHOCH
            else:
                kind = BreakKind.BOS

            events.append(
                StructureEvent(
                    kind=kind,
                    direction=direction,
                    level=float(swing.price),
                    break_index=hit,
                    break_time=times[hit],
                    break_close=float(closes[hit]),
                    swing_index=swing.index,
                    swing_time=swing.time,
                )
            )

            # Either event flips the working bias, so the next break in the new
            # direction is a continuation (BOS) rather than another CHoCH.
            bias = StructureBias.BULLISH if direction == 1 else StructureBias.BEARISH

        if swing.kind is SwingKind.HIGH:
            seen_highs.append(swing)
        else:
            seen_lows.append(swing)

        # Re-derive bias from structure only, never from a lone break, so a
        # single break of an early pivot cannot manufacture a trend.
        if len(seen_highs) >= 2 and len(seen_lows) >= 2:
            structural_bias, _, _ = _bias_and_confidence(seen_highs, seen_lows)
            if structural_bias is not StructureBias.RANGING:
                bias = structural_bias

    events.sort(key=lambda e: (e.break_index, e.direction))
    return events


def _first_index_above(closes: np.ndarray, start: int, level: float) -> Optional[int]:
    tail = closes[start:]
    hits = np.flatnonzero(tail > level)
    return int(start + hits[0]) if hits.size else None


def _first_index_below(closes: np.ndarray, start: int, level: float) -> Optional[int]:
    tail = closes[start:]
    hits = np.flatnonzero(tail < level)
    return int(start + hits[0]) if hits.size else None


def active_swing(
    state: StructureState, direction: int, before_index: Optional[int] = None
) -> Optional[SwingPoint]:
    """
    The swing that defines invalidation for a trade in ``direction``.

    For a long, that is the most recent confirmed swing *low* — the level whose
    loss says the buyer was wrong. For a short, the most recent swing *high*.

    This is the structural answer to "where does this idea stop being true",
    which is a question the existing ``Signal.invalidation`` string can only
    describe in English and never prices.
    """
    kind = SwingKind.LOW if direction > 0 else SwingKind.HIGH
    for s in reversed(state.swings):
        if s.kind is not kind:
            continue
        if before_index is not None and s.index >= before_index:
            continue
        return s
    return None
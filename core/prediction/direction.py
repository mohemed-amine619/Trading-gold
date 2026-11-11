"""
core/prediction/direction.py — standalone directional bias for a symbol.

This module answers "which way does the market look right now?" using only
price history. It is deliberately broker-free and stateless: it never submits
an order, never touches risk state, and never imports MetaTrader5. That makes
it safe to run off-market (weekends, after hours) or in CI.

Inputs are validated rather than defaulted: missing or unusable data yields an
honest ``NEUTRAL`` verdict with a reason, never a fabricated direction.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from core.market.location import analyse_location
from core.market.pipeline import add_indicators
from core.strategy.ensemble import get_default_ensemble


@dataclass
class DirectionPrediction:
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    as_of: Optional[str] = None
    direction: int = 0            # +1 LONG, -1 SHORT, 0 NEUTRAL
    label: str = "NEUTRAL"
    score: float = 0.0            # 0..100
    confidence: float = 0.0       # 0..1
    location_bias: str = "UNDEFINED"
    range_position: Optional[float] = None
    structure_bias: str = "UNDEFINED"
    reasons: List[str] = field(default_factory=list)
    invalidation: str = ""
    nn: Optional[Dict[str, Any]] = None
    consensus: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}


def _has_indicators(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ("ema50", "rsi", "macd_hist"))


def _structure_bias(loc) -> str:
    try:
        bias = loc.structure.bias
        return getattr(bias, "value", str(bias))
    except Exception:
        return "UNDEFINED"


def _as_of(df: pd.DataFrame) -> Optional[str]:
    try:
        idx = df.index
        if len(idx) == 0:
            return None
        last = idx[-1]
        return str(last)
    except Exception:
        return None


def predict_direction(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    model_path: Optional[str] = None,
    min_oos_edge: float = 0.0,
) -> DirectionPrediction:
    """Return a directional bias for ``df`` (OHLCV, oldest->newest)."""

    base = DirectionPrediction(symbol=symbol, timeframe=timeframe)

    # --- validate input ------------------------------------------------
    if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
        base.reasons = ["No usable price data: empty frame or missing 'close' column"]
        base.as_of = _as_of(df) if isinstance(df, pd.DataFrame) else None
        return base

    base.as_of = _as_of(df)

    # --- indicators ----------------------------------------------------
    work = df
    if not _has_indicators(work):
        try:
            work = add_indicators(work)
        except Exception as exc:  # never crash the caller on a bad frame
            base.reasons = [f"indicator pipeline failed: {exc!r}"]
            return base

    # --- location ------------------------------------------------------
    try:
        loc = analyse_location(work)
    except Exception as exc:
        base.reasons = [f"location analysis failed: {exc!r}"]
        return base

    base.location_bias = getattr(loc.bias, "value", str(loc.bias))
    base.range_position = loc.range_position
    base.structure_bias = _structure_bias(loc)

    # --- ensemble ------------------------------------------------------
    try:
        signal = get_default_ensemble().evaluate_all(work)
    except Exception as exc:
        base.reasons = [f"strategy ensemble failed: {exc!r}"]
        return base

    direction = int(signal.direction or 0)
    score = float(signal.score or 0.0)
    reasons: List[str] = []
    if signal.reason:
        reasons.append(signal.reason)
    invalidation = signal.invalidation or ""

    # --- location veto (location may only remove trades) ---------------
    if direction != 0:
        try:
            gate = loc.entry_gate(direction)
        except Exception as exc:
            gate = None
            reasons.append(f"location gate error: {exc!r}")
        if gate is not None and not gate.allowed:
            direction = 0
            score = 0.0
            reasons.append(f"location veto: {gate.reason}")
        elif gate is not None and getattr(gate, "cautions", None):
            reasons.extend(gate.cautions)

    # --- confidence ----------------------------------------------------
    confidence = (score / 100.0) if direction != 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    base.direction = direction
    base.label = _LABELS.get(direction, "NEUTRAL")
    base.score = round(score, 2)
    base.confidence = round(confidence, 4)
    base.reasons = reasons
    base.invalidation = invalidation

    # --- optional NN challenger ---------------------------------------
    nn_info: Optional[Dict[str, Any]] = None
    if model_path:
        try:
            from core.ai.nn_predictor import ModelPredictor

            predictor = ModelPredictor(model_path, min_oos_edge=min_oos_edge)
            nn_sig = predictor.predict_signal(work)
            nn_info = {
                "status": predictor.status,
                "direction": int(getattr(nn_sig, "direction", 0) or 0),
                "confidence": float(getattr(nn_sig, "confidence", 0.0) or 0.0),
                "reason": getattr(nn_sig, "reason", ""),
                "oos_edge": predictor.oos_edge,
            }
        except Exception as exc:
            nn_info = {
                "status": "error",
                "direction": 0,
                "confidence": 0.0,
                "reason": f"predictor error: {exc!r}",
                "oos_edge": None,
            }
    base.nn = nn_info

    # --- consensus -----------------------------------------------------
    if nn_info is not None:
        nn_dir = int(nn_info.get("direction", 0) or 0)
        if direction != 0 and nn_dir != 0:
            base.consensus = "agree" if direction == nn_dir else "conflict"
        elif direction != 0:
            base.consensus = "base-only"
        elif nn_dir != 0:
            base.consensus = "nn-only"
        else:
            base.consensus = "none"

    return base


_TF_RANK = {
    "M1": 1, "M2": 2, "M3": 3, "M5": 4, "M15": 5, "M30": 6,
    "H1": 7, "H2": 8, "H4": 9, "D1": 10, "W1": 11,
}


@dataclass
class TimeframePrediction:
    timeframe: str
    prediction: DirectionPrediction


@dataclass
class MultiTimeframePrediction:
    symbol: Optional[str] = None
    timeframes: Dict[str, DirectionPrediction] = field(default_factory=dict)
    higher_timeframe: Optional[str] = None
    consensus_direction: int = 0
    consensus_label: str = "NEUTRAL"
    alignment: int = 0
    conflicting: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframes": {k: v.to_dict() for k, v in self.timeframes.items()},
            "higher_timeframe": self.higher_timeframe,
            "consensus_direction": self.consensus_direction,
            "consensus_label": self.consensus_label,
            "alignment": self.alignment,
            "conflicting": list(self.conflicting),
            "reasons": list(self.reasons),
        }


def predict_multi_timeframe(
    frames: Dict[str, pd.DataFrame],
    symbol: Optional[str] = None,
    model_path: Optional[str] = None,
    higher_timeframe: Optional[str] = None,
    min_oos_edge: float = 0.0,
) -> MultiTimeframePrediction:
    """Combine per-timeframe predictions into one confluence read.

    ``frames`` maps a timeframe label (e.g. "H1", "H4", "D1") to its OHLCV
    frame. The higher timeframe is the dominant vote; when it is neutral the
    majority of the lower timeframes decides. Location vetoes are already
    applied inside ``predict_direction``.
    """
    result = MultiTimeframePrediction(symbol=symbol)

    if not isinstance(frames, dict) or not frames:
        result.reasons = ["No timeframe frames supplied"]
        return result

    preds: Dict[str, DirectionPrediction] = {}
    for tf, df in frames.items():
        preds[tf] = predict_direction(
            df, symbol=symbol, timeframe=tf,
            model_path=model_path, min_oos_edge=min_oos_edge,
        )
    result.timeframes = preds

    # pick higher timeframe
    if higher_timeframe is None or higher_timeframe not in preds:
        higher_timeframe = max(preds.keys(), key=lambda t: _TF_RANK.get(t.upper(), 0))
    result.higher_timeframe = higher_timeframe

    higher_dir = preds[higher_timeframe].direction

    # majority fallback
    votes = [p.direction for p in preds.values()]
    if higher_dir != 0:
        consensus = higher_dir
    else:
        from collections import Counter
        c = Counter(v for v in votes if v != 0)
        if c:
            consensus = c.most_common(1)[0][0]
        else:
            consensus = 0

    result.consensus_direction = int(consensus)
    result.consensus_label = _LABELS.get(consensus, "NEUTRAL")
    result.alignment = sum(1 for p in preds.values() if p.direction == consensus and consensus != 0)
    result.conflicting = [tf for tf, p in preds.items()
                         if consensus != 0 and p.direction != 0 and p.direction != consensus]

    # reasons
    reasons = []
    if higher_dir != 0:
        reasons.append(f"higher timeframe {higher_timeframe} biases { _LABELS.get(higher_dir)}")
    else:
        reasons.append(f"higher timeframe {higher_timeframe} is neutral; using majority")
    if result.conflicting:
        reasons.append("conflicting timeframes: " + ", ".join(result.conflicting))
    if result.alignment:
        reasons.append(f"{result.alignment}/{len(preds)} timeframe(s) align with consensus")
    result.reasons = reasons
    return result

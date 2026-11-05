"""
core/ai/nn_predictor.py — the integration surface for the sequence model.

``ModelPredictor.predict_signal(df)`` is what a bot loop calls on each closed
bar. It never raises into the caller and it fails NEUTRAL (direction 0): a
missing file, missing torch, a model whose out-of-sample edge was never
measured or is below the floor, too little history, or non-finite features all
return "no trade" with a reason, never a guess.

torch is imported only when a model is first loaded, so importing this module
is safe on a machine without torch.

The model file is re-checked on every call (one ``stat``). When a continuous
trainer replaces it, the next prediction loads the new model, with no restart.

Direction follows the bot's convention: +1 long, -1 short, 0 no trade.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.ai.deep_layout import ARCH_GRU, latest_model_window
from core.ai.nn_features import LONG, NOISE, SHORT, FeatureScaler

logger = logging.getLogger(__name__)

# Rows needed so the EWM features have converged to the values seen in training.
_WARMUP_ROWS = 400
_UNSET = object()


@dataclass(frozen=True)
class NNSignal:
    direction: int                       # +1 long, -1 short, 0 no trade
    confidence: float                    # probability of the most likely trade class
    probs: tuple                         # (noise, long, short), calibrated
    reason: str

    def as_tuple(self) -> tuple:
        """``(direction, confidence)``, the shape ``AIEnsemble.predict`` returns."""
        return self.direction, self.confidence


def _last_training_note(model_path: str) -> str:
    """Why the trainer has not saved a model yet, from its history file.

    The trainer writes a model only when one passes its out-of-sample gate; a
    rejected challenger leaves no file. Without this the bot can only say
    "no model file", which hides whether the trainer is failing, skipping, or
    simply finding no edge.
    """
    try:
        with open(model_path + ".history.jsonl", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
    except OSError:
        return " (the trainer has not finished a cycle yet)"
    if not lines:
        return " (the trainer has not finished a cycle yet)"
    try:
        last = json.loads(lines[-1])
    except ValueError:
        return ""
    return (f" (trainer after {len(lines)} cycle(s): {last.get('action', '?')} - "
            f"{last.get('reason', '?')})")


def _neutral(reason: str) -> NNSignal:
    return NNSignal(0, 0.5, (1.0, 0.0, 0.0), reason)


class ModelPredictor:
    def __init__(self, artifact_path: str, min_confidence: float = 0.72,
                 min_oos_edge: float = 0.0):
        self.path = artifact_path
        self.min_confidence = float(min_confidence)
        self.min_oos_edge = float(min_oos_edge)
        self._loaded = False
        self._load_error: Optional[str] = None
        self._members = None
        self._ensemble_proba = None
        self._scaler: Optional[FeatureScaler] = None
        self._window = 0
        self._arch = ARCH_GRU
        self._oos_edge: Optional[float] = None
        self._mtime = _UNSET

    # -- loading ---------------------------------------------------------
    def _file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return None

    def _load(self) -> None:
        current = self._file_mtime()
        if current != self._mtime:        # first call, or the file was replaced
            self._mtime = current
            self._loaded = False
            self._load_error = None
            self._members = None
        if self._loaded or self._load_error:
            return
        if current is None:
            self._load_error = f"no model file at {self.path}" + _last_training_note(self.path)
            return
        try:
            from core.ai.nn_model import ensemble_proba, load_artifact, members_from_artifact
        except Exception as exc:                      # torch not installed
            self._load_error = f"torch unavailable: {exc!r}"
            return
        try:
            art = load_artifact(self.path)
            self._members = members_from_artifact(art)
            self._ensemble_proba = ensemble_proba
            self._scaler = FeatureScaler.from_dict(art["scaler"])
            self._window = int(art["config"]["window"])
            self._arch = art["config"].get("arch", ARCH_GRU)
            self._oos_edge = (art.get("oos") or {}).get("edge")
            self._loaded = True
        except Exception as exc:
            self._load_error = f"model file unusable: {exc!r}"

    @property
    def oos_edge(self) -> Optional[float]:
        """The out-of-sample edge measured for the loaded model (``None`` = unmeasured)."""
        self._load()
        return self._oos_edge if self._loaded else None

    @property
    def status(self) -> str:
        self._load()
        return "ready" if self._loaded else (self._load_error or "not loaded")

    # -- prediction ------------------------------------------------------
    def predict_signal(self, latest_dataframe: pd.DataFrame) -> NNSignal:
        try:
            return self._predict(latest_dataframe)
        except Exception as exc:          # a broken model must never stop the bot
            logger.warning("NN predictor failed: %r", exc)
            return _neutral(f"predictor error: {exc!r}")

    def _predict(self, df: pd.DataFrame) -> NNSignal:
        self._load()
        if not self._loaded:
            return _neutral(self._load_error or "model not loaded")
        if self._oos_edge is None:
            return _neutral("model out-of-sample edge was never measured")
        if self._oos_edge < self.min_oos_edge:
            return _neutral(
                f"model out-of-sample edge {self._oos_edge:.3f} below floor "
                f"{self.min_oos_edge:.3f}")

        window = latest_model_window(df.tail(self._window + _WARMUP_ROWS), self._window,
                                     self._arch)
        if window is None:
            return _neutral("not enough finite history for the model window")

        x = self._scaler.transform(window)[None, :, :]
        p = self._ensemble_proba(self._members, x)[0]

        trade_class = LONG if p[LONG] >= p[SHORT] else SHORT
        conf = float(p[trade_class])
        probs = (float(p[NOISE]), float(p[LONG]), float(p[SHORT]))
        if p.argmax() == NOISE:
            return NNSignal(0, conf, probs, "model favours no trade")
        if conf < self.min_confidence:
            return NNSignal(0, conf, probs,
                            f"confidence {conf:.2f} below {self.min_confidence:.2f}")
        return NNSignal(1 if trade_class == LONG else -1, conf, probs, "signal")

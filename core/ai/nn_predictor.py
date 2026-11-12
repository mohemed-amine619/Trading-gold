"""
core/ai/nn_predictor.py — the integration surface for the sequence model.

``ModelPredictor.predict_signal(df)`` is what a bot loop calls on each closed
bar. It never raises into the caller and it fails NEUTRAL (direction 0): a
missing file, missing torch, a model whose out-of-sample edge was never
measured, is below the floor, is not confidently positive, or is no better than
the naive always-one-direction baseline, too little history, or non-finite
features all return "no trade" with a reason, never a guess.

Which gates apply depends on what the artifact records:

* An artifact written by the current trainer stores its decision ``threshold``,
  its ensemble ``calibration`` and an ``oos`` block with ``edge_lcb``,
  ``baseline_edge`` and ``excess_lcb``. It votes only when ALL of: ``edge >=
  min_oos_edge``; ``edge_lcb > max(min_oos_edge, 0)`` (the same rule the
  trainer's ``--floor`` applies, so ``NN_MIN_OOS_EDGE`` means one thing on both
  sides); ``edge > baseline_edge``; ``excess_lcb > 0``. A missing value on such
  an artifact is unmeasured and refuses. It votes at ``max(stored threshold,
  min_confidence)``: its edge was measured at the stored threshold, and the
  operator's ``NN_MIN_CONFIDENCE`` stays a minimum.
* An artifact saved before those fields existed (no ``threshold`` key) loads
  and predicts exactly as before: per-member temperatures, no calibration map,
  the configured ``min_confidence``, and only the ``edge >= min_oos_edge`` gate.

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

    Once there is a REAL verdict (promoted / kept / error) the note is that
    verdict, not the latest "skipped": a skip after a verdict only means
    "waiting for new bars", and its bar count changes every cycle, which made
    the bot log the line every hour while hiding why the model was rejected.
    Before any verdict the latest skip IS the reason, so it is shown verbatim
    (with its bar counts): "waiting for enough bars" used to hide whether the
    trainer lacked history, new bars, or bars before the holdout.
    """
    try:
        with open(model_path + ".history.jsonl", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
    except OSError:
        return " (the trainer has not finished a cycle yet)"
    entries = []
    for ln in lines:
        try:
            entries.append(json.loads(ln))
        except ValueError:
            continue
    if not entries:
        return " (the trainer has not finished a cycle yet)"
    verdicts = [e for e in entries if e.get("action") in ("promoted", "kept", "error")]
    waiting = entries[-1].get("action") == "skipped"
    if not verdicts:
        reason = str(entries[-1].get("reason") or "waiting for enough bars")
        return f" (the trainer has not trained yet: {reason})"
    v = verdicts[-1]
    return (f" (trainer's last verdict: {v.get('action', '?')} - {v.get('reason', '?')}"
            + ("; now waiting for new bars to retrain" if waiting else "") + ")")


def _neutral(reason: str) -> NNSignal:
    return NNSignal(0, 0.5, (1.0, 0.0, 0.0), reason)


def oos_refusal(oos: Optional[dict], min_oos_edge: float, strict: bool) -> Optional[str]:
    """Why a model with this out-of-sample record must not vote, or ``None``.

    ``strict`` is for artifacts written by the current trainer (they record a
    threshold): their lower bound, naive baseline and timing skill are
    required, and a missing value refuses. Old artifacts get only the original
    point-edge floor, so they behave exactly as before.
    """
    oos = oos or {}
    edge = oos.get("edge")
    if edge is None:
        return "model out-of-sample edge was never measured"
    if edge < min_oos_edge:
        return f"model out-of-sample edge {edge:.3f} below floor {min_oos_edge:.3f}"
    if not strict:
        return None
    lcb = oos.get("edge_lcb")
    floor = max(min_oos_edge, 0.0)
    if lcb is None:
        return f"model out-of-sample edge {edge:+.3f} has no confidence bound (unmeasured)"
    if lcb <= floor:
        return (f"model out-of-sample edge {edge:+.3f} is not confidently above the floor "
                f"{floor:.3f} (lower bound {lcb:+.3f})")
    base = oos.get("baseline_edge")
    if base is None:
        return f"model out-of-sample edge {edge:+.3f} has no naive baseline to beat (unmeasured)"
    if edge <= base:
        return (f"model out-of-sample edge {edge:+.3f} is no better than the naive "
                f"always-{oos.get('baseline_direction') or 'one-direction'} baseline {base:+.3f}")
    excess = oos.get("excess_lcb")
    if excess is None or excess <= 0:
        shown = "unmeasured" if excess is None else f"{excess:+.3f}"
        return (f"model out-of-sample edge {edge:+.3f} shows no timing skill beyond its "
                f"direction (lower bound of the excess over the naive baseline {shown})")
    return None


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
        self._oos: dict = {}
        self._oos_edge: Optional[float] = None
        self._calibration: Optional[dict] = None
        self._stored_threshold: Optional[float] = None
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
        if current is None:                 # re-read each time: the trainer's verdict changes
            self._load_error = f"no model file at {self.path}" + _last_training_note(self.path)
            return
        if self._loaded or self._load_error:
            return
        try:
            from core.ai.nn_model import (
                apply_calibration, ensemble_proba, load_artifact, members_from_artifact,
            )
        except Exception as exc:                      # torch not installed
            self._load_error = f"torch unavailable: {exc!r}"
            return
        try:
            art = load_artifact(self.path)
            calibration = art.get("calibration")
            apply_calibration(np.full((1, 3), 1.0 / 3.0), calibration)   # unknown kind: refuse now
            stored = art.get("threshold")
            self._members = members_from_artifact(art)
            self._ensemble_proba = ensemble_proba
            self._calibration = calibration
            self._stored_threshold = float(stored) if stored is not None else None
            self._scaler = FeatureScaler.from_dict(art["scaler"])
            self._window = int(art["config"]["window"])
            self._arch = art["config"].get("arch", ARCH_GRU)
            self._oos = dict(art.get("oos") or {})
            self._oos_edge = self._oos.get("edge")
            self._loaded = True
        except Exception as exc:
            self._load_error = f"model file unusable: {exc!r}"

    @property
    def oos_edge(self) -> Optional[float]:
        """The out-of-sample edge measured for the loaded model (``None`` = unmeasured)."""
        self._load()
        return self._oos_edge if self._loaded else None

    @property
    def threshold(self) -> float:
        """The confidence a call needs to vote: the artifact's stored threshold
        (where its edge was measured), never below ``min_confidence``; just
        ``min_confidence`` for an artifact that stores none."""
        if self._stored_threshold is None:
            return self.min_confidence
        return max(self._stored_threshold, self.min_confidence)

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
        oos = dict(self._oos)
        oos["edge"] = self._oos_edge
        refusal = oos_refusal(oos, self.min_oos_edge, strict=self._stored_threshold is not None)
        if refusal:
            return _neutral(refusal)

        window = latest_model_window(df.tail(self._window + _WARMUP_ROWS), self._window,
                                     self._arch)
        if window is None:
            return _neutral("not enough finite history for the model window")

        x = self._scaler.transform(window)[None, :, :]
        p = self._ensemble_proba(self._members, x, self._calibration)[0]

        trade_class = LONG if p[LONG] >= p[SHORT] else SHORT
        conf = float(p[trade_class])
        probs = (float(p[NOISE]), float(p[LONG]), float(p[SHORT]))
        if p.argmax() == NOISE:
            return NNSignal(0, conf, probs, "model favours no trade")
        threshold = self.threshold
        if conf < threshold:
            return NNSignal(0, conf, probs, f"confidence {conf:.2f} below {threshold:.2f}")
        return NNSignal(1 if trade_class == LONG else -1, conf, probs, "signal")

"""
core/ai/deep_predictor.py — the integration surface for the deep two-branch model.

``DeepPredictor.predict_signal(df)`` is what a bot loop calls on each closed bar.
It never raises into the caller and it fails NEUTRAL (direction 0), exactly like
``core/ai/nn_predictor.ModelPredictor``: a missing file, missing torch, an
unusable file, feature columns that differ from the current code, a model whose
out-of-sample edge was never measured or is below the floor, too little history
or non-finite features all return "no trade" with a reason, never a guess.

torch is imported only when a model is first loaded, so importing this module is
safe on a machine without torch.

The model file is re-checked on every call (one ``stat``). When a continuous
trainer replaces it, the next prediction loads the new model, with no restart.
The loaded model is swapped in as one object, so a call that is running while the
file changes keeps a consistent model.

Direction follows the bot's convention: +1 long, -1 short, 0 no trade.
``DeepSignal`` is an ``NNSignal`` (same fields and ``as_tuple``) plus
``excursion``, the expected favourable move in ATR multiples; it is meaningful
only when ``direction != 0``.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from core.ai.deep_features import build_sequence_features
from core.ai.loc_features import location_vector_from_frame
from core.ai.nn_features import LONG, NOISE, SHORT, FeatureScaler
from core.ai.nn_predictor import NNSignal

logger = logging.getLogger(__name__)

# Rows needed so the EWM features have converged to the values seen in training.
_WARMUP_ROWS = 400
_UNSET = object()


@dataclass(frozen=True)
class DeepSignal(NNSignal):
    """A drop-in ``NNSignal`` that also carries the expected excursion.

    ``excursion`` is the expected favourable move in ATR multiples of the
    barrier-hit move, conditional on a trade outcome. It is 0.0 whenever
    ``direction`` is 0: there is nothing to expect from a trade that is not taken,
    so a consumer must not read it without checking ``direction``.
    """

    excursion: float = 0.0


def _neutral(reason: str) -> DeepSignal:
    return DeepSignal(0, 0.5, (1.0, 0.0, 0.0), reason, 0.0)


def _clean_edge(value: Any) -> Optional[float]:
    """A measured edge as a float, or ``None``. NaN / inf / non-numbers are
    unmeasured: ``nan < floor`` is False, so a NaN would otherwise pass the floor."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class _Loaded:
    """Everything one prediction needs, swapped in as a single object."""
    members: list
    seq_scaler: FeatureScaler
    loc_scaler: FeatureScaler
    window: int
    loc_lookback: int
    oos_edge: Optional[float]
    predict: Callable


class DeepPredictor:
    def __init__(self, artifact_path: str, min_confidence: float = 0.72,
                 min_oos_edge: float = 0.0):
        self.path = artifact_path
        self.min_confidence = float(min_confidence)
        self.min_oos_edge = float(min_oos_edge)
        self._lock = threading.Lock()
        self._stamp: Any = _UNSET
        self._bundle: Optional[_Loaded] = None
        self._load_error: Optional[str] = None

    # -- loading ---------------------------------------------------------
    def _file_stamp(self) -> Optional[tuple]:
        """``(mtime, size)`` of the model file, or ``None`` if it is not there."""
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _load(self, stamp: Optional[tuple]) -> tuple:
        """``(bundle, error)``: exactly one of them is ``None``."""
        if stamp is None:
            return None, f"no model file at {self.path}"
        try:
            from core.ai.deep_model import (
                FeatureMismatchError, config_from_dict, deep_members_from_artifact,
                deep_predict, load_deep_artifact, scalers_from_artifact,
            )
        except Exception as exc:                      # torch not installed
            return None, f"torch unavailable: {exc!r}"
        try:
            art = load_deep_artifact(self.path)
            members = deep_members_from_artifact(art)          # also checks feature names
            seq_scaler, loc_scaler = scalers_from_artifact(art)
            cfg = config_from_dict(art["config"], require_all=True)
            edge = _clean_edge((art.get("oos") or {}).get("edge"))
            return _Loaded(members, seq_scaler, loc_scaler, int(cfg.window),
                           int(cfg.loc_lookback), edge, deep_predict), None
        except FeatureMismatchError as exc:
            return None, f"model feature mismatch: {exc}"
        except Exception as exc:
            return None, f"model file unusable: {exc!r}"

    def _refresh(self) -> tuple:
        """Reload when the file is new or changed; return ``(bundle, error)``."""
        with self._lock:
            stamp = self._file_stamp()
            if stamp != self._stamp:                  # first call, or the file was replaced
                self._stamp = stamp
                self._bundle, self._load_error = self._load(stamp)
            return self._bundle, self._load_error

    @property
    def oos_edge(self) -> Optional[float]:
        """The out-of-sample edge measured for the loaded model (``None`` = unmeasured)."""
        bundle, _ = self._refresh()
        return bundle.oos_edge if bundle is not None else None

    @property
    def status(self) -> str:
        bundle, error = self._refresh()
        return "ready" if bundle is not None else (error or "not loaded")

    # -- prediction ------------------------------------------------------
    def predict_signal(self, latest_dataframe: pd.DataFrame) -> DeepSignal:
        try:
            return self._predict(latest_dataframe)
        except Exception as exc:          # a broken model must never stop the bot
            logger.warning("deep predictor failed: %r", exc)
            return _neutral(f"predictor error: {exc!r}")

    def _predict(self, df: pd.DataFrame) -> DeepSignal:
        bundle, error = self._refresh()
        if bundle is None:
            return _neutral(error or "model not loaded")
        if bundle.oos_edge is None:
            return _neutral("model out-of-sample edge was never measured")
        if bundle.oos_edge < self.min_oos_edge:
            return _neutral(f"model out-of-sample edge {bundle.oos_edge:.3f} below floor "
                            f"{self.min_oos_edge:.3f}")

        feats = build_sequence_features(df.tail(bundle.window + _WARMUP_ROWS)).to_numpy()
        if len(feats) < bundle.window:
            return _neutral("not enough history for the model window")
        window = feats[-bundle.window:]
        if not np.isfinite(window).all():
            return _neutral("not enough finite history for the model window")
        loc = location_vector_from_frame(df, bundle.loc_lookback)
        if not np.isfinite(loc).all():
            return _neutral("location features are not finite")

        x_seq = bundle.seq_scaler.transform(window)[None, :, :].astype(np.float32)
        x_loc = bundle.loc_scaler.transform(loc[None, :]).astype(np.float32)
        proba, excursion = bundle.predict(bundle.members, x_seq, x_loc)
        p, exc = proba[0], float(excursion[0])
        if not (np.isfinite(p).all() and math.isfinite(exc)):
            return _neutral("model output is not finite")

        trade_class = LONG if p[LONG] >= p[SHORT] else SHORT
        conf = float(p[trade_class])
        probs = (float(p[NOISE]), float(p[LONG]), float(p[SHORT]))
        if p.argmax() == NOISE:
            return DeepSignal(0, conf, probs, "model favours no trade", 0.0)
        if conf < self.min_confidence:
            return DeepSignal(0, conf, probs,
                              f"confidence {conf:.2f} below {self.min_confidence:.2f}", 0.0)
        return DeepSignal(1 if trade_class == LONG else -1, conf, probs, "signal", exc)

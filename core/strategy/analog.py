"""
core/strategy/analog.py — "history repeats": trade what followed the most similar
past situations.

For the current bar, build a shape from the last ``window`` closes in ATR units
(close[t-i] - close[t]) / ATR[t]. Compare it with the same shape at every earlier
bar, take the ``k`` nearest, and look at what each of those bars did next: which
barrier price touched first, +1 ATR (up) or -1 ATR (down), within ``horizon``
bars. If most neighbours went the same way, that is the vote.

Two rules keep it honest:

* No look-ahead. A past bar is only used if its outcome was fully observable at
  the query bar (``j + horizon <= t``). ``predict_at`` takes the query index so
  the tests can prove nothing after it is read.
* It must prove itself first. The same procedure is replayed over the most recent
  ``check_bars`` bars, each time using only the history before that bar. The
  vote is only given when those replayed calls had a clearly positive edge (mean
  R-multiple of the directional calls: +1 win, -1 loss, 0 for a timeout) over at
  least ``min_calls`` calls. "Clearly" means above ``min_edge`` AND above
  ``min_z`` standard errors, where the effective sample is calls / horizon,
  because neighbouring bars share neighbours and outcomes: 180 overlapping calls
  are worth about 15 independent ones. Without that, a random walk "proves"
  itself about one time in six. Otherwise the engine ABSTAINS and the caller must
  treat the strategy as absent. An unmeasured edge is never treated as zero.

"History repeats" is a hypothesis about a market, not a law. On a series with no
repeating structure the engine abstains; the tests measure how often it fails to.

Pure numpy/pandas. No broker, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.ai.labels import first_barrier_labels

ATR_PERIOD = 14
_CLIP = 10.0


@dataclass(frozen=True)
class AnalogConfig:
    window: int = 24              # closes in the shape
    k: int = 15                   # nearest neighbours that vote
    horizon: int = 12             # bars an outcome is allowed to take
    barrier_mult: float = 1.0     # barrier distance in ATR
    vote: float = 0.60            # share of neighbours that must agree
    min_library: int = 300        # past bars needed before the engine speaks
    check_bars: int = 400         # recent bars replayed to measure its edge
    min_calls: int = 30           # directional calls needed to trust the measurement
    min_edge: float = 0.10        # floor on the mean R-multiple of those calls
    min_z: float = 2.0            # ...and it must also beat chance by this many sigmas
    max_distance_ratio: float = 0.85   # neighbours must be clearly closer than typical


@dataclass(frozen=True)
class AnalogSignal:
    direction: int                # +1 long, -1 short, 0 no call
    strength: float               # share of neighbours that agree (0.5 when no call)
    p_long: float
    p_short: float
    n_neighbours: int
    abstained: bool               # True: the engine is not voting at all
    reason: str
    edge: Optional[float] = None  # measured edge of its recent calls (None = unmeasured)
    calls: int = 0                # directional calls behind that measurement


def _abstain(reason: str, edge: Optional[float] = None, calls: int = 0) -> AnalogSignal:
    return AnalogSignal(0, 0.5, 0.0, 0.0, 0, True, reason, edge, calls)


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


class AnalogEngine:
    def __init__(self, cfg: Optional[AnalogConfig] = None):
        self.cfg = cfg or AnalogConfig()

    # -- building blocks ---------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shape vectors (one row per bar), outcomes and ATR for the whole frame."""
        c = self.cfg
        close = df["close"].to_numpy(dtype=float)
        atr = _atr(df)
        n = len(close)
        w = c.window
        vectors = np.full((n, w - 1), np.nan)
        for i in range(w - 1, n):
            a = atr[i]
            if not (np.isfinite(a) and a > 0):
                continue
            vectors[i] = np.clip((close[i - w + 1:i] - close[i]) / a, -_CLIP, _CLIP)
        outcomes = first_barrier_labels(df["high"].to_numpy(dtype=float),
                                        df["low"].to_numpy(dtype=float), close, atr,
                                        c.barrier_mult, c.horizon)
        return vectors, outcomes, atr

    def library_indices(self, vectors: np.ndarray, t: int) -> Optional[np.ndarray]:
        """Bars the vote at ``t`` may learn from: those whose outcome was fully
        observable by ``t`` (``j + horizon <= t``) and whose shape is defined.
        ``None`` when there are fewer than ``min_library`` of them."""
        c = self.cfg
        last = t - c.horizon                          # newest usable library bar
        first = c.window - 1
        if last - first + 1 < c.min_library:
            return None
        finite = np.isfinite(vectors[first:last + 1]).all(axis=1)
        if finite.sum() < c.min_library:
            return None
        return np.flatnonzero(finite) + first

    def _vote(self, vectors: np.ndarray, outcomes: np.ndarray, t: int) -> AnalogSignal:
        """The vote at bar ``t`` using only bars whose outcome was known by ``t``."""
        c = self.cfg
        query = vectors[t]
        if not np.isfinite(query).all():
            return _abstain("shape not defined yet")
        idx = self.library_indices(vectors, t)
        if idx is None:
            return _abstain("not enough history")
        dist = np.sqrt(((vectors[idx] - query) ** 2).sum(axis=1))
        k = min(c.k, len(idx))
        nearest = np.argpartition(dist, k - 1)[:k]
        # Analogues must be clearly closer than a typical past bar, otherwise the
        # "similar" situations are not similar at all.
        if np.median(dist[nearest]) > c.max_distance_ratio * np.median(dist):
            return AnalogSignal(0, 0.5, 0.0, 0.0, k, False, "no close analogues")
        out = outcomes[idx[nearest]]
        p_long = float(np.mean(out == 1.0))
        p_short = float(np.mean(out == 0.0))
        if p_long >= c.vote and p_long > p_short:
            return AnalogSignal(1, p_long, p_long, p_short, k, False, "analogues went up")
        if p_short >= c.vote and p_short > p_long:
            return AnalogSignal(-1, p_short, p_long, p_short, k, False, "analogues went down")
        return AnalogSignal(0, 0.5, p_long, p_short, k, False, "analogues disagree")

    def predict_at(self, df: pd.DataFrame, t: int) -> AnalogSignal:
        """The raw vote at bar ``t`` (no self-check). Reads nothing after ``t``."""
        vectors, outcomes, _ = self._prepare(df.iloc[:t + 1])
        return self._vote(vectors, outcomes, t)

    # -- self-check --------------------------------------------------------
    def measure_edge(self, df: pd.DataFrame, vectors=None, outcomes=None) -> Tuple[Optional[float], int]:
        """Replay the vote over the recent bars. Returns ``(edge, calls)``.

        Each replayed call at bar ``t`` uses only bars up to ``t - horizon``; its
        result is scored against the barrier outcome of bar ``t`` itself, which is
        known for every bar earlier than ``n - horizon``. ``edge`` is ``None``
        below ``min_calls`` calls.
        """
        c = self.cfg
        if vectors is None:
            vectors, outcomes, _ = self._prepare(df)
        n = len(df)
        newest_scored = n - 1 - c.horizon
        wins = losses = calls = 0
        for t in range(max(c.window, newest_scored - c.check_bars + 1), newest_scored + 1):
            sig = self._vote(vectors, outcomes, t)
            if sig.abstained or sig.direction == 0:
                continue
            calls += 1
            result = outcomes[t]
            if np.isnan(result):
                continue                                  # timeout: flat
            if (sig.direction == 1) == (result == 1.0):
                wins += 1
            else:
                losses += 1
        if calls < c.min_calls:
            return None, calls
        return (wins - losses) / calls, calls

    # -- public ------------------------------------------------------------
    def evaluate(self, df: pd.DataFrame) -> AnalogSignal:
        """The vote for the newest bar, given only if the engine has proven itself."""
        c = self.cfg
        if len(df) < c.min_library + c.window + c.horizon + 1:
            return _abstain("not enough history")
        vectors, outcomes, _ = self._prepare(df)
        edge, calls = self.measure_edge(df, vectors, outcomes)
        if edge is None:
            return _abstain(f"unproven: only {calls} recent calls (need {c.min_calls})",
                            None, calls)
        required = max(c.min_edge, c.min_z * float(np.sqrt(c.horizon / calls)))
        if edge < required:
            return _abstain(f"no proven edge on recent history ({edge:+.2f} over {calls} calls, "
                            f"needs {required:+.2f})", edge, calls)
        sig = self._vote(vectors, outcomes, len(df) - 1)
        return AnalogSignal(sig.direction, sig.strength, sig.p_long, sig.p_short,
                            sig.n_neighbours, False, sig.reason, edge, calls)

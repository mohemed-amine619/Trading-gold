"""
core/strategy/scalp.py — a set of classic scalping strategies that must each
PROVE themselves on recent history before they are allowed to vote.

Strategies (each a causal signal generator: +1 long, -1 short, 0 nothing, using
only bars up to and including the bar the signal is on):

  orb          Opening-range breakout: first close beyond the first ``range_bars``
               after the session open hour, once per day.
  vwap_fade    Fade a stretch of price ``k`` ATR away from the day's VWAP.
  squeeze      Bollinger-band squeeze (narrow bands for a while), then a close
               outside the band.
  rsi2         Short-term RSI(2) exhaustion against the EMA(50) trend.
  burst        Momentum burst: a wide, high-volume bar closing near its extreme.
  ema_pullback EMA 9/21/50 stack, price dips to the EMA21 and closes back above
               (below) the EMA9.
  sweep        Liquidity sweep: a new 20-bar extreme that closes back inside.

The proof rule (the same idea as core/strategy/analog.py): every signal is scored
against the barrier outcome of its own bar (+1 ATR touched first = win in the
called direction, -1 ATR first = loss, timeout = 0), over the most recent
``check_bars`` bars. A strategy may vote only if

  * it has at least ``min_calls`` INDEPENDENT calls (calls at least ``horizon``
    bars apart, because outcomes of nearby bars overlap),
  * its mean R-multiple is at least ``min_edge``, and
  * that mean beats chance by ``min_z`` standard errors, ``1 / sqrt(n_independent)``.

With seven strategies tested at once, ``min_z`` is raised from the usual 2.0 to
2.7 so that the chance that some strategy proves itself by luck stays small
(measured in the tests on random walks). A strategy with too few events in the
available history simply abstains: an unmeasured edge is never treated as zero.

When no proven strategy has a live signal on the newest bars the component is
ABSENT from the score (see ``scalp_support``), so the bot scores exactly as
without it.

Pure numpy/pandas. No broker, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.ai.labels import first_barrier_labels
from core.strategy.analog import _atr

ALL_STRATEGIES = ("orb", "vwap_fade", "squeeze", "rsi2", "burst", "ema_pullback", "sweep",
                  "breakout", "pinbar", "macd_cross", "stoch_cross", "inside_break")
_EPS = 1e-12


@dataclass(frozen=True)
class ScalpConfig:
    strategies: Tuple[str, ...] = ALL_STRATEGIES
    horizon: int = 12             # bars an outcome may take
    barrier_mult: float = 1.0     # barrier distance, ATR
    check_bars: int = 900         # recent bars replayed to prove each strategy
    min_calls: int = 15           # independent calls needed to trust the proof
    min_edge: float = 0.10        # floor on the mean R-multiple
    min_z: float = 2.7            # standard errors above chance (7 strategies tested at once)
    valid_bars: int = 2           # a signal stays live this many bars
    open_hour: int = 7            # opening-range hour, in the data's own clock
    range_bars: int = 6           # bars that make the opening range
    walkforward_folds: int = 1    # 1 = single window (default); >1 = require edge in every fold


@dataclass(frozen=True)
class ScalpVote:
    name: str
    direction: int                # +1 / -1: a live signal on the newest bars
    strength: float               # 0.6 .. 1.0, grows with the proven edge
    edge: float
    n_independent: int
    reason: str = "live signal"


@dataclass
class ScalpResult:
    votes: List[ScalpVote] = field(default_factory=list)
    report: Dict[str, dict] = field(default_factory=dict)   # per strategy: proof details


# ── signal generators ────────────────────────────────────────────────────────

def _volume(df: pd.DataFrame) -> pd.Series:
    for col in ("tick_volume", "volume"):
        if col in df.columns:
            return df[col].astype(float).clip(lower=0)
    return pd.Series(1.0, index=df.index)


def _day_id(df: pd.DataFrame) -> np.ndarray:
    days = df.index.normalize().asi8
    return np.cumsum(np.r_[0, (np.diff(days) != 0).astype(int)])


def _to_signal(long_cond: pd.Series, short_cond: pd.Series) -> np.ndarray:
    sig = np.zeros(len(long_cond), dtype=np.int8)
    sig[np.asarray(long_cond.fillna(False), dtype=bool)] = 1
    sig[np.asarray(short_cond.fillna(False), dtype=bool)] = -1
    return sig


def sig_orb(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, window_bars: int = 36) -> np.ndarray:
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hours = df.index.hour.to_numpy()
    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    day = _day_id(df)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        a, b = idx[0], idx[-1] + 1
        opens = np.flatnonzero(hours[a:b] >= cfg.open_hour)
        if len(opens) == 0:
            continue
        start = a + opens[0]
        end = start + cfg.range_bars
        if end > b:
            continue
        hi, lo = high[start:end].max(), low[start:end].min()
        for j in range(end, min(b, end + window_bars)):
            if close[j] > hi:
                sig[j] = 1
                break
            if close[j] < lo:
                sig[j] = -1
                break
    return sig


def sig_vwap_fade(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, k: float = 1.5) -> np.ndarray:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = _volume(df) + 1.0
    day = pd.Series(_day_id(df), index=df.index)
    cum_pv = (tp * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    vwap = cum_pv / cum_v
    in_day = day.groupby(day).cumcount()
    dev = (df["close"] - vwap) / pd.Series(atr, index=df.index).replace(0, np.nan)
    ready = in_day >= 12
    short = (dev >= k) & (dev.shift(1) < k) & ready
    long_ = (dev <= -k) & (dev.shift(1) > -k) & ready
    return _to_signal(long_, short)


def sig_squeeze(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, n: int = 20,
                pct: float = 0.25, lookback: int = 100, min_run: int = 6) -> np.ndarray:
    close = df["close"]
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    bw = (upper - lower) / mid
    thr = bw.rolling(lookback, min_periods=n * 2).quantile(pct)
    sq = (bw <= thr).fillna(False)
    run = sq.groupby((~sq).cumsum()).cumsum()
    armed = run.shift(1) >= min_run
    return _to_signal(armed & (close > upper), armed & (close < lower))


def sig_rsi2(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, lo: float = 10.0,
             hi: float = 90.0) -> np.ndarray:
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / (loss + _EPS))
    ema = close.ewm(span=50, adjust=False, min_periods=50).mean()
    up = (close > ema) & (ema > ema.shift(5))
    down = (close < ema) & (ema < ema.shift(5))
    return _to_signal(up & (rsi < lo) & (rsi.shift(1) >= lo), down & (rsi > hi) & (rsi.shift(1) <= hi))


def sig_burst(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, rng_mult: float = 1.5,
              vol_mult: float = 1.5, frac: float = 0.25) -> np.ndarray:
    rng = df["high"] - df["low"]
    prev_atr = pd.Series(atr, index=df.index).shift(1)
    vol = _volume(df)
    vol_avg = vol.rolling(20).mean().shift(1)
    big = (rng >= rng_mult * prev_atr) & (vol >= vol_mult * vol_avg)
    pos = (df["close"] - df["low"]) / (rng + _EPS)
    return _to_signal(big & (pos >= 1 - frac), big & (pos <= frac))


def sig_ema_pullback(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig) -> np.ndarray:
    close = df["close"]
    e9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    e21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    e50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    up = (e9 > e21) & (e21 > e50)
    down = (e9 < e21) & (e21 < e50)
    long_ = up & (df["low"] <= e21) & (close > e9)
    short = down & (df["high"] >= e21) & (close < e9)
    return _to_signal(long_, short)


def sig_sweep(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, lookback: int = 20) -> np.ndarray:
    prior_low = df["low"].shift(1).rolling(lookback).min()
    prior_high = df["high"].shift(1).rolling(lookback).max()
    long_ = (df["low"] < prior_low) & (df["close"] > prior_low) & (df["close"] > df["open"])
    short = (df["high"] > prior_high) & (df["close"] < prior_high) & (df["close"] < df["open"])
    return _to_signal(long_, short)




def sig_breakout(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, lookback: int = 20) -> np.ndarray:
    high = df["high"]; low = df["low"]; close = df["close"]
    prior_high = high.shift(1).rolling(lookback).max()
    prior_low = low.shift(1).rolling(lookback).min()
    long_ = close > prior_high
    short = close < prior_low
    return _to_signal(long_, short)


def sig_pinbar(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig, wick_ratio: float = 2.0) -> np.ndarray:
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    body = (c - o).abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    bull = (lower_wick >= wick_ratio * (body + 1e-12)) & (c > o)
    bear = (upper_wick >= wick_ratio * (body + 1e-12)) & (c < o)
    return _to_signal(bull, bear)


def sig_macd_cross(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig) -> np.ndarray:
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    long_ = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    short = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    return _to_signal(long_, short)


def sig_stoch_cross(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig) -> np.ndarray:
    hi = df["high"]; lo = df["low"]; cl = df["close"]
    lowest = lo.rolling(14).min()
    highest = hi.rolling(14).max()
    k = 100.0 * (cl - lowest) / (highest - lowest + 1e-12)
    d = k.rolling(3).mean()
    long_ = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 25)
    short = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 75)
    return _to_signal(long_, short)


def sig_inside_break(df: pd.DataFrame, atr: np.ndarray, cfg: ScalpConfig) -> np.ndarray:
    hi = df["high"]; lo = df["low"]; cl = df["close"]
    prev_hi = hi.shift(1); prev_lo = lo.shift(1)
    inside = (hi <= prev_hi) & (lo >= prev_lo)
    # breakout bar after an inside bar
    long_ = inside.shift(1).fillna(False) & (cl > hi.shift(1))
    short = inside.shift(1).fillna(False) & (cl < lo.shift(1))
    return _to_signal(long_, short)

GENERATORS: Dict[str, Callable] = {
    "orb": sig_orb, "vwap_fade": sig_vwap_fade, "squeeze": sig_squeeze, "rsi2": sig_rsi2,
    "burst": sig_burst, "ema_pullback": sig_ema_pullback, "sweep": sig_sweep,
    "breakout": sig_breakout, "pinbar": sig_pinbar, "macd_cross": sig_macd_cross,
    "stoch_cross": sig_stoch_cross, "inside_break": sig_inside_break,
}


# ── proof ────────────────────────────────────────────────────────────────────

def thin(indices: Sequence[int], spacing: int) -> List[int]:
    """Greedy subset of ``indices`` at least ``spacing`` apart: the independent calls."""
    kept: List[int] = []
    for i in indices:
        if not kept or i - kept[-1] >= spacing:
            kept.append(i)
    return kept


def score_signals(sig: np.ndarray, outcomes: np.ndarray, lo: int, hi: int) -> Tuple[int, int, int]:
    """(wins, losses, calls) of the calls on bars ``lo..hi`` (inclusive).

    ``outcomes[t]`` is 1.0 when the upper barrier was touched first, 0.0 when the
    lower was, NaN on a timeout or an unresolvable bar (scored as flat).
    """
    wins = losses = calls = 0
    for t in range(max(lo, 0), min(hi, len(sig) - 1) + 1):
        if sig[t] == 0:
            continue
        calls += 1
        result = outcomes[t]
        if np.isnan(result):
            continue
        if (sig[t] == 1) == (result == 1.0):
            wins += 1
        else:
            losses += 1
    return wins, losses, calls


def _prove_window(sig: np.ndarray, outcomes: np.ndarray, cfg: ScalpConfig,
                  lo: int, hi: int) -> dict:
    """Core proof logic on an explicit replay window ``[lo, hi]``."""
    calls_at = [t for t in range(max(lo, 0), min(hi, len(sig) - 1) + 1) if sig[t] != 0]
    independent = thin(calls_at, cfg.horizon)
    n_eff = len(independent)
    if n_eff < cfg.min_calls:
        return {"proven": False, "edge": None, "calls": len(calls_at), "n_independent": n_eff,
                "required": None, "reason": f"unproven: {n_eff} independent calls "
                                            f"(need {cfg.min_calls})",
                "lo": lo, "hi": hi}
    wins, losses, calls = score_signals(sig, outcomes, lo, hi)
    if calls <= 0:
        return {"proven": False, "edge": None, "calls": 0, "n_independent": n_eff,
                "required": None, "reason": "no scored calls in window",
                "lo": lo, "hi": hi}
    edge = (wins - losses) / calls
    required = max(cfg.min_edge, cfg.min_z / float(np.sqrt(n_eff)))
    if edge < required:
        return {"proven": False, "edge": edge, "calls": calls, "n_independent": n_eff,
                "required": required, "reason": f"no proven edge ({edge:+.2f} over {n_eff} "
                                                f"independent calls, needs {required:+.2f})",
                "lo": lo, "hi": hi}
    return {"proven": True, "edge": edge, "calls": calls, "n_independent": n_eff,
            "required": required, "reason": "proven", "lo": lo, "hi": hi}


def prove(sig: np.ndarray, outcomes: np.ndarray, cfg: ScalpConfig) -> dict:
    """Replay one strategy over the recent bars and decide if it has proven itself."""
    n = len(sig)
    hi = n - 1 - cfg.horizon                       # newest bar whose outcome is known
    lo = max(0, hi - cfg.check_bars + 1)
    if getattr(cfg, "walkforward_folds", 1) and cfg.walkforward_folds > 1:
        return prove_walkforward(sig, outcomes, cfg)
    return _prove_window(sig, outcomes, cfg, lo, hi)


def prove_walkforward(sig: np.ndarray, outcomes: np.ndarray, cfg: ScalpConfig) -> dict:
    """Require the edge to hold in every contiguous fold of the replay window."""
    n = len(sig)
    hi = n - 1 - cfg.horizon
    lo = max(0, hi - cfg.check_bars + 1)
    folds = max(2, int(cfg.walkforward_folds))
    window = hi - lo + 1
    fold_size = window // folds
    if fold_size < cfg.horizon + cfg.min_calls:
        return {"proven": False, "edge": None, "calls": 0, "n_independent": 0,
                "required": None, "reason": "not enough history for walk-forward",
                "lo": lo, "hi": hi}
    fold_results = []
    for f in range(folds):
        f_lo = lo + f * fold_size
        f_hi = (lo + (f + 1) * fold_size - 1) if f < folds - 1 else hi
        fold_results.append(_prove_window(sig, outcomes, cfg, f_lo, f_hi))
    if not all(fr.get("proven") for fr in fold_results):
        failed = [fr for fr in fold_results if not fr.get("proven")]
        return {"proven": False, "edge": float(np.mean([fr.get("edge") or 0.0
                                                         for fr in fold_results])),
                "calls": sum(fr.get("calls", 0) for fr in fold_results),
                "n_independent": sum(fr.get("n_independent", 0) for fr in fold_results),
                "required": fold_results[0].get("required"),
                "reason": "walk-forward: " + (failed[0].get("reason") if failed else "failed"),
                "lo": lo, "hi": hi, "folds": fold_results}
    edges = [fr.get("edge") for fr in fold_results if fr.get("edge") is not None]
    return {"proven": True, "edge": float(np.mean(edges)) if edges else None,
            "calls": sum(fr.get("calls", 0) for fr in fold_results),
            "n_independent": sum(fr.get("n_independent", 0) for fr in fold_results),
            "required": fold_results[0].get("required"),
            "reason": "proven (walk-forward)", "lo": lo, "hi": hi, "folds": fold_results}

class ScalpEngine:
    def __init__(self, cfg: Optional[ScalpConfig] = None):
        self.cfg = cfg or ScalpConfig()

    def signals(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        atr = _atr(df)
        return {name: GENERATORS[name](df, atr, self.cfg)
                for name in self.cfg.strategies if name in GENERATORS}

    def outcomes(self, df: pd.DataFrame) -> np.ndarray:
        return first_barrier_labels(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
                                    df["close"].to_numpy(dtype=float), _atr(df),
                                    self.cfg.barrier_mult, self.cfg.horizon)

    def evaluate(self, df: pd.DataFrame) -> ScalpResult:
        """Proof for every strategy, plus a vote from each proven one that is live now."""
        c = self.cfg
        result = ScalpResult()
        if len(df) < c.horizon + 60 or not isinstance(df.index, pd.DatetimeIndex):
            result.report = {name: {"proven": False, "reason": "not enough history"}
                             for name in c.strategies}
            return result
        sigs = self.signals(df)
        outcomes = self.outcomes(df)
        n = len(df)
        for name, sig in sigs.items():
            proof = prove(sig, outcomes, c)
            live = [t for t in range(max(0, n - c.valid_bars), n) if sig[t] != 0]
            proof["live"] = int(sig[live[-1]]) if live else 0
            result.report[name] = proof
            if proof["proven"] and live:
                strength = 0.6 + 0.4 * min(1.0, max(0.0, proof["edge"]))
                result.votes.append(ScalpVote(name, int(sig[live[-1]]), strength, proof["edge"],
                                              proof["n_independent"]))
        return result


def scalp_support(votes: Sequence[ScalpVote], direction: int) -> Optional[float]:
    """How strongly the live, proven strategies back ``direction`` (0..1).

    ``None`` when no proven strategy has a live signal: the component is then
    absent from the score. Otherwise the agreeing strengths minus the opposing
    ones, averaged over the voters and clipped at 0, so a proven live signal
    AGAINST a direction removes that direction's share of the budget.
    """
    if not votes:
        return None
    net = sum(v.strength if v.direction == direction else -v.strength for v in votes)
    return max(0.0, min(1.0, net / len(votes)))

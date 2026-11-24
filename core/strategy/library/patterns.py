"""
core/strategy/library/patterns.py — chart patterns built on a causal zig-zag of
confirmed swing pivots (head & shoulders, double tops, triangles, flags, cup and
handle, harmonic, Elliott wave 3).

A pattern is only known once its last pivot is confirmed; the trade is the first
close through its trigger level after that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import EPS, ohlc, to_signal, zigzag


def _first_cross(trigger_long: np.ndarray, trigger_short: np.ndarray, close: np.ndarray,
                 key: np.ndarray) -> np.ndarray:
    """Signal the first close beyond the trigger level for each pattern instance
    (identified by ``key`` = index of its last pivot)."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    used = set()
    for t in range(1, n):
        k = key[t]
        if k < 0 or k in used:
            continue
        tl, ts = trigger_long[t], trigger_short[t]
        if np.isfinite(tl) and close[t] > tl:
            sig[t] = 1
            used.add(k)
        elif np.isfinite(ts) and close[t] < ts:
            sig[t] = -1
            used.add(k)
    return sig


def _zz(df, k=3, depth=6):
    o, h, l, c = ohlc(df)
    P, K, I = zigzag(h, l, k, depth)
    return o, h, l, c, P, K, I


def sig_head_shoulders(df, atr, cfg):
    """Head and shoulders (and inverse): H-L-H(head)-L-H with the head >= 0.5 ATR above
    both shoulders and the shoulders within 1 ATR; trade the neckline break."""
    o, h, l, c, P, K, I = _zz(df)
    a = atr
    p1, p2, p3, p4, p5 = (P[:, j] for j in range(1, 6))
    top = (K[:, 5] == 1) & (p3 > p1 + 0.5 * a) & (p3 > p5 + 0.5 * a) & (np.abs(p1 - p5) <= a)
    bot = (K[:, 5] == -1) & (p3 < p1 - 0.5 * a) & (p3 < p5 - 0.5 * a) & (np.abs(p1 - p5) <= a)
    neck_short = np.where(top, np.fmin(p2, p4), np.nan)
    neck_long = np.where(bot, np.fmax(p2, p4), np.nan)
    key = np.where(top | bot, I[:, 5], -1)
    return _first_cross(neck_long, neck_short, c, key)


def sig_double_top_bottom(df, atr, cfg):
    """Double top / bottom: two swing extremes within 0.3 ATR, 1.5+ ATR above (below)
    the trough between them; trade the break of that trough (peak)."""
    o, h, l, c, P, K, I = _zz(df)
    p3, p4, p5 = P[:, 3], P[:, 4], P[:, 5]
    top = (K[:, 5] == 1) & (np.abs(p3 - p5) <= 0.3 * atr) & (np.fmin(p3, p5) - p4 >= 1.5 * atr)
    bot = (K[:, 5] == -1) & (np.abs(p3 - p5) <= 0.3 * atr) & (p4 - np.fmax(p3, p5) >= 1.5 * atr)
    key = np.where(top | bot, I[:, 5], -1)
    return _first_cross(np.where(bot, p4, np.nan), np.where(top, p4, np.nan), c, key)


def sig_triangle(df, atr, cfg):
    """Triangle: lower highs and higher lows over the last four pivots; trade the first
    close outside."""
    o, h, l, c, P, K, I = _zz(df)
    p2, p3, p4, p5 = P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    last_high = K[:, 5] == 1
    hi1, hi2 = np.where(last_high, p3, p2), np.where(last_high, p5, p4)
    lo1, lo2 = np.where(last_high, p2, p3), np.where(last_high, p4, p5)
    tri = (hi2 < hi1) & (lo2 > lo1) & ((hi2 - lo2) >= 0.5 * atr)
    key = np.where(tri, I[:, 5], -1)
    return _first_cross(np.where(tri, hi2, np.nan), np.where(tri, lo2, np.nan), c, key)


def sig_cup_handle(df, atr, cfg):
    """Cup and handle: rim - bottom - rim (rims within 0.5 ATR, cup >= 3 ATR deep and
    >= 20 bars wide), then a handle low in the upper half of the cup; trade the break
    of the rims. Inverted for shorts."""
    o, h, l, c, P, K, I = _zz(df)
    r1, b, r2, hd = P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    wide = (I[:, 4] - I[:, 2]) >= 20
    up = ((K[:, 5] == -1) & (np.abs(r1 - r2) <= 0.5 * atr) & (np.fmin(r1, r2) - b >= 3 * atr)
          & wide & (hd > b + 0.5 * (np.fmin(r1, r2) - b)))
    dn = ((K[:, 5] == 1) & (np.abs(r1 - r2) <= 0.5 * atr) & (b - np.fmax(r1, r2) >= 3 * atr)
          & wide & (hd < b - 0.5 * (b - np.fmax(r1, r2))))
    key = np.where(up | dn, I[:, 5], -1)
    return _first_cross(np.where(up, np.fmax(r1, r2), np.nan),
                        np.where(dn, np.fmin(r1, r2), np.nan), c, key)


def sig_flag(df, atr, cfg):
    """Flag / pennant: a pole of 3+ ATR within 8 bars, then 6 bars of consolidation
    narrower than half the pole that retrace less than half of it; trade the break."""
    o, h, l, c = ohlc(df)
    cs = pd.Series(c)
    pole = (cs.shift(6) - cs.shift(14)).to_numpy()
    cons_hi = pd.Series(h).shift(1).rolling(6).max().to_numpy()
    cons_lo = pd.Series(l).shift(1).rolling(6).min().to_numpy()
    width = cons_hi - cons_lo
    top = cs.shift(6).to_numpy()
    bull = (pole >= 3 * atr) & (width <= 0.5 * pole) & (cons_lo >= top - 0.5 * pole)
    bear = (pole <= -3 * atr) & (width <= -0.5 * pole) & (cons_hi <= top - 0.5 * pole)
    return to_signal(bull & (c > cons_hi), bear & (c < cons_lo))


def sig_harmonic(df, atr, cfg):
    """Harmonic (Gartley / Bat family): X-A-B-C with AB = 0.382-0.886 XA and
    BC = 0.382-0.886 AB, then a close back out of the 0.786-0.886 XA zone (D)."""
    o, h, l, c, P, K, I = _zz(df, depth=6)
    X, A, B, C = P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    xa = np.abs(A - X)
    ab = np.abs(A - B)
    bc = np.abs(C - B)
    shape = ((ab >= 0.382 * xa) & (ab <= 0.886 * xa) & (bc >= 0.382 * ab) & (bc <= 0.886 * ab)
             & (xa >= 3 * atr))
    bull = shape & (K[:, 2] == -1) & (K[:, 5] == 1)        # X low, A high, B low, C high
    bear = shape & (K[:, 2] == 1) & (K[:, 5] == -1)
    d_hi_bull, d_lo_bull = A - 0.786 * xa, A - 0.886 * xa
    d_lo_bear, d_hi_bear = A + 0.786 * xa, A + 0.886 * xa
    long_ = bull & (l <= d_hi_bull) & (l >= d_lo_bull) & (c > o)
    short = bear & (h >= d_lo_bear) & (h <= d_hi_bear) & (c < o)
    sig = to_signal(long_, short)
    key = np.where(bull | bear, I[:, 5], -1)
    out = np.zeros_like(sig)
    used = set()
    for t in np.flatnonzero(sig):
        if key[t] >= 0 and key[t] not in used:
            used.add(key[t]); out[t] = sig[t]
    return out


def sig_elliott_w3(df, atr, cfg):
    """Elliott wave 3 entry: wave 1 up (>= 2 ATR), wave 2 retraces 38-79% without
    breaking the start, then the close above the wave-1 top starts wave 3."""
    o, h, l, c, P, K, I = _zz(df)
    w0, w1, w2 = P[:, 3], P[:, 4], P[:, 5]
    leg = np.abs(w1 - w0)
    ret = np.abs(w1 - w2) / (leg + EPS)
    ok = (leg >= 2 * atr) & (ret >= 0.382) & (ret <= 0.786)
    up = ok & (K[:, 5] == -1) & (w1 > w0) & (w2 > w0)
    dn = ok & (K[:, 5] == 1) & (w1 < w0) & (w2 < w0)
    key = np.where(up | dn, I[:, 5], -1)
    return _first_cross(np.where(up, w1, np.nan), np.where(dn, w1, np.nan), c, key)


GENERATORS = {
    "head_shoulders": sig_head_shoulders, "double_top_bottom": sig_double_top_bottom,
    "triangle": sig_triangle, "cup_handle": sig_cup_handle, "flag": sig_flag,
    "harmonic": sig_harmonic, "elliott_w3": sig_elliott_w3,
}

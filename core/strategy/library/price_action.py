"""
core/strategy/library/price_action.py — price-action strategies (no indicator required).

Each function: ``(df, atr, cfg) -> int8 array`` of +1 long / -1 short / 0, causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import (
    EPS, Zone, adx, ema, ohlc, retest_signals, swings, to_signal,
)


def _support_resistance(df, atr, k: int, tol: float) -> np.ndarray:
    o, h, l, c = ohlc(df)
    sw = swings(h, l, k)
    a = atr
    prev_c = np.r_[np.nan, c[:-1]]
    sup, res = sw.sl1, sw.sh1
    long_ = ((l <= sup + tol * a) & (l >= sup - 2 * tol * a) & (c > o) & (c > sup + 0.1 * a)
             & (prev_c > sup + tol * a))
    short = ((h >= res - tol * a) & (h <= res + 2 * tol * a) & (c < o) & (c < res - 0.1 * a)
             & (prev_c < res - tol * a))
    return to_signal(np.nan_to_num(long_, nan=0), np.nan_to_num(short, nan=0))


def sig_sr_bounce(df, atr, cfg):
    """Support & Resistance: first rejection of the latest confirmed swing level."""
    return _support_resistance(df, atr, 3, 0.25)


def sig_sr_swing(df, atr, cfg):
    """Support/Resistance swing: the same rejection on major (5-bar) swing levels."""
    return _support_resistance(df, atr, 5, 0.35)


def sig_supply_demand(df, atr, cfg):
    """Supply & Demand: a tight base (2 small bars) left by a strong impulse; trade the
    first return into the base."""
    o, h, l, c = ohlc(df)
    zones = []
    rng = h - l
    for t in range(3, len(c)):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        base_small = rng[t - 1] < 0.6 * a and rng[t - 2] < 0.6 * a
        impulse = rng[t] >= 1.5 * a and abs(c[t] - o[t]) >= 0.6 * rng[t]
        if base_small and impulse:
            lo, hi = min(l[t - 1], l[t - 2]), max(h[t - 1], h[t - 2])
            zones.append(Zone(t, lo, hi, 1 if c[t] > o[t] else -1))
    return retest_signals(zones, h, l, c, max_age=150)


def sig_fakeout(df, atr, cfg):
    """False breakout: the previous bar CLOSED beyond a swing level, this bar closes
    back inside -> trade the failure."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    pc = np.r_[np.nan, c[:-1]]
    lvl_h = np.r_[np.nan, sw.sh1[:-1]]
    lvl_l = np.r_[np.nan, sw.sl1[:-1]]
    short = (pc > lvl_h) & (c < lvl_h) & (c < o)
    long_ = (pc < lvl_l) & (c > lvl_l) & (c > o)
    return to_signal(long_, short)


def sig_retest(df, atr, cfg):
    """Break and retest: a close through a swing level, then the first pull-back to it
    that holds."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    zones = []
    pc = np.r_[np.nan, c[:-1]]
    for t in range(1, len(c)):
        a = atr[t]
        if not np.isfinite(a):
            continue
        lh, ll = sw.sh1[t], sw.sl1[t]
        if np.isfinite(lh) and c[t] > lh >= pc[t]:
            zones.append(Zone(t, lh - 0.2 * a, lh + 0.2 * a, 1))
        if np.isfinite(ll) and c[t] < ll <= pc[t]:
            zones.append(Zone(t, ll - 0.2 * a, ll + 0.2 * a, -1))
    return retest_signals(zones, h, l, c, max_age=30)


def sig_trendline(df, atr, cfg):
    """Trendline: a line through the last two rising swing lows (falling swing highs);
    buy (sell) the first touch that holds."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    t = np.arange(len(c), dtype=float)
    up_slope = (sw.sl1 - sw.sl2) / (sw.sl1_i - sw.sl2_i + EPS)
    up_line = sw.sl1 + up_slope * (t - sw.sl1_i)
    dn_slope = (sw.sh1 - sw.sh2) / (sw.sh1_i - sw.sh2_i + EPS)
    dn_line = sw.sh1 + dn_slope * (t - sw.sh1_i)
    prev_c = np.r_[np.nan, c[:-1]]
    prev_up = np.r_[np.nan, up_line[:-1]]
    prev_dn = np.r_[np.nan, dn_line[:-1]]
    long_ = ((sw.sl1 > sw.sl2) & (l <= up_line + 0.15 * atr) & (c > up_line) & (c > o)
             & (prev_c > prev_up + 0.15 * atr))
    short = ((sw.sh1 < sw.sh2) & (h >= dn_line - 0.15 * atr) & (c < dn_line) & (c < o)
             & (prev_c < prev_dn - 0.15 * atr))
    return to_signal(long_, short)


def regression_channel(close: np.ndarray, n: int = 50):
    y = pd.Series(close)
    x = pd.Series(np.arange(len(close), dtype=float))
    mx, my = x.rolling(n).mean(), y.rolling(n).mean()
    cov = (x * y).rolling(n).mean() - mx * my
    vx = (x * x).rolling(n).mean() - mx * mx
    vy = (y * y).rolling(n).mean() - my * my
    slope = cov / (vx + EPS)
    fit = my + slope * (x - mx)
    resid = np.sqrt((vy - slope * slope * vx).clip(lower=0))
    return slope.to_numpy(), fit.to_numpy(), resid.to_numpy()


def sig_channel(df, atr, cfg):
    """Channel: a rising (falling) 50-bar regression channel; buy the lower band
    (sell the upper band) with the slope."""
    o, h, l, c = ohlc(df)
    slope, fit, sd = regression_channel(c, 50)
    lower, upper = fit - 2 * sd, fit + 2 * sd
    trend = slope * 50 / (atr + EPS)
    long_ = (trend > 1.0) & (l <= lower) & (c > lower) & (c > o)
    short = (trend < -1.0) & (h >= upper) & (c < upper) & (c < o)
    return to_signal(long_, short)


def sig_range(df, atr, cfg):
    """Range trading: a flat 60-bar box 3-10 ATR tall; fade its edges."""
    o, h, l, c = ohlc(df)
    hi = pd.Series(h).shift(1).rolling(60).max().to_numpy()
    lo = pd.Series(l).shift(1).rolling(60).min().to_numpy()
    w = hi - lo
    slope, _, _ = regression_channel(c, 60)
    flat = np.abs(slope * 60) < 0.3 * w
    box = (w > 3 * atr) & (w < 10 * atr) & flat
    long_ = box & (l <= lo + 0.15 * w) & (c > o) & (c > lo)
    short = box & (h >= hi - 0.15 * w) & (c < o) & (c < hi)
    return to_signal(long_, short)


def sig_swing_hl(df, atr, cfg):
    """Swing high / swing low: a higher low confirmed inside a higher-high structure
    (or a lower high in a lower-low structure)."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    long_ = sw.new_low & (sw.sl1 > sw.sl2) & (sw.sh1 > sw.sh2) & (c > sw.sl1)
    short = sw.new_high & (sw.sh1 < sw.sh2) & (sw.sl1 < sw.sl2) & (c < sw.sh1)
    return to_signal(long_, short)


def sig_candles_3bar(df, atr, cfg):
    """Candlestick patterns: morning/evening star and three soldiers/crows."""
    o, h, l, c = ohlc(df)
    body = c - o
    a = atr
    b1, b2, b3 = np.r_[np.nan, np.nan, body[:-2]], np.r_[np.nan, body[:-1]], body
    o1, c1 = np.r_[np.nan, np.nan, o[:-2]], np.r_[np.nan, np.nan, c[:-2]]
    mid1 = (o1 + c1) / 2
    morning = (b1 < -0.6 * a) & (np.abs(b2) < 0.3 * a) & (b3 > 0.4 * a) & (c > mid1)
    evening = (b1 > 0.6 * a) & (np.abs(b2) < 0.3 * a) & (b3 < -0.4 * a) & (c < mid1)
    cs = pd.Series(c)
    up3 = (b1 > 0.5 * a) & (b2 > 0.5 * a) & (b3 > 0.5 * a) & (cs > cs.shift(1)).to_numpy() \
        & (cs.shift(1) > cs.shift(2)).to_numpy() & ((h - c) < 0.25 * (h - l + EPS))
    dn3 = (b1 < -0.5 * a) & (b2 < -0.5 * a) & (b3 < -0.5 * a) & (cs < cs.shift(1)).to_numpy() \
        & (cs.shift(1) < cs.shift(2)).to_numpy() & ((c - l) < 0.25 * (h - l + EPS))
    return to_signal(morning | up3, evening | dn3)


def sig_engulfing(df, atr, cfg):
    """Engulfing after a 3-bar move the other way."""
    o, h, l, c = ohlc(df)
    po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    c4 = pd.Series(c).shift(4).to_numpy()
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & ((c - o) > (po - pc)) & (pc < c4)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & ((o - c) > (pc - po)) & (pc > c4)
    return to_signal(bull, bear)


def sig_outside_bar(df, atr, cfg):
    """Outside bar: engulfs the prior range, is at least 1 ATR tall and closes at an extreme."""
    o, h, l, c = ohlc(df)
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    rng = h - l
    outside = (h > ph) & (l < pl) & (rng >= atr)
    pos = (c - l) / (rng + EPS)
    return to_signal(outside & (pos >= 0.75), outside & (pos <= 0.25))


def sig_reversal(df, atr, cfg):
    """Reversal: price stretched 2.5 ATR from EMA20, then a bar that closes through
    the previous bar's extreme the other way."""
    o, h, l, c = ohlc(df)
    e = ema(c, 20).to_numpy()
    stretch = (c - e) / (atr + EPS)
    ps = np.r_[np.nan, stretch[:-1]]
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    return to_signal((ps < -2.5) & (c > ph), (ps > 2.5) & (c < pl))


def sig_momentum(df, atr, cfg):
    """Momentum: 10-bar rate of change crosses 2 ATR with ADX above 25."""
    o, h, l, c = ohlc(df)
    roc = (c - pd.Series(c).shift(10).to_numpy()) / (atr + EPS)
    proc = np.r_[np.nan, roc[:-1]]
    a, _, _ = adx(h, l, c)
    strong = a.to_numpy() > 25
    return to_signal(strong & (roc > 2) & (proc <= 2), strong & (roc < -2) & (proc >= -2))


GENERATORS = {
    "sr_bounce": sig_sr_bounce, "sr_swing": sig_sr_swing, "supply_demand": sig_supply_demand,
    "fakeout": sig_fakeout, "retest": sig_retest, "trendline": sig_trendline,
    "channel": sig_channel, "range_trading": sig_range, "swing_hl": sig_swing_hl,
    "candles_3bar": sig_candles_3bar, "engulfing": sig_engulfing, "outside_bar": sig_outside_bar,
    "reversal": sig_reversal, "momentum": sig_momentum,
}

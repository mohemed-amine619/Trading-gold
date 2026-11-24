"""
core/strategy/library/volume.py — volume, order-flow and auction strategies.

IMPORTANT: MT5 gold CFDs have no exchange volume, no bid/ask trade prints and no
order book. These strategies use TICK VOLUME (the number of price updates) as a
proxy for activity, and the close's position in the bar's range as a proxy for
who was aggressive. They are labelled "proxy" in the catalogue. Market-profile
(TPO) strategies need no volume at all and are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import (
    EPS, Zone, ny_clock, ohlc, previous_day_profile, retest_signals, to_signal, volume,
    volume_profile, first_per_group,
)


def _clock(df, cfg):
    return ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))


def _delta(df) -> np.ndarray:
    """Per-bar delta proxy: tick volume x close location value (-1 .. +1)."""
    o, h, l, c = ohlc(df)
    clv = (2 * c - h - l) / (h - l + EPS)
    return volume(df) * clv


def sig_volume_profile(df, atr, cfg, window: int = 200, step: int = 1):
    """Volume profile (rolling 200 bars, tick-volume weighted): rejection of the value
    area low (buy) or high (sell)."""
    o, h, l, c = ohlc(df)
    v = volume(df)
    n = len(c)
    val = np.full(n, np.nan); vah = np.full(n, np.nan)
    for t in range(window, n):
        _, val[t], vah[t] = volume_profile(l[t - window:t], h[t - window:t], v[t - window:t])
    return to_signal((l <= val) & (c > val) & (c > o), (h >= vah) & (c < vah) & (c < o))


def sig_poc(df, atr, cfg, window: int = 200):
    """Point of control: a close crossing the rolling 200-bar POC on 1.5x average tick volume."""
    o, h, l, c = ohlc(df)
    v = volume(df)
    n = len(c)
    poc = np.full(n, np.nan)
    for t in range(window, n):
        poc[t], _, _ = volume_profile(l[t - window:t], h[t - window:t], v[t - window:t])
    avg = pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    loud = v >= 1.5 * avg
    return to_signal(loud & (pc <= poc) & (c > poc), loud & (pc >= poc) & (c < poc))


def sig_frvp(df, atr, cfg):
    """Fixed-range volume profile (previous day): the first test of yesterday's POC from
    the side the day opened on, rejected."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    poc, _, _ = previous_day_profile(h, l, volume(df), clk.fx_day)
    dopen = pd.Series(o).groupby(pd.Series(clk.fx_day)).transform("first").to_numpy()
    sig = to_signal((dopen > poc) & (l <= poc) & (c > poc), (dopen < poc) & (h >= poc) & (c < poc))
    return first_per_group(sig, clk.fx_day)


def sig_tpo_80(df, atr, cfg):
    """Market profile / TPO 80% rule: the day opens outside yesterday's TPO value area,
    then two closes back inside -> trade toward the other side."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    _, val, vah = previous_day_profile(h, l, np.ones(len(c)), clk.fx_day)
    dopen = pd.Series(o).groupby(pd.Series(clk.fx_day)).transform("first").to_numpy()
    inside = (c > val) & (c < vah)
    two = inside & np.r_[False, inside[:-1]]
    same_day = np.r_[False, clk.fx_day[1:] == clk.fx_day[:-1]]
    sig = to_signal(two & same_day & (dopen < val), two & same_day & (dopen > vah))
    return first_per_group(sig, clk.fx_day)


def sig_vah_val(df, atr, cfg):
    """Value-area acceptance: two consecutive closes above yesterday's VAH (below VAL)."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    _, val, vah = previous_day_profile(h, l, volume(df), clk.fx_day)
    above = c > vah
    below = c < val
    sig = to_signal(above & np.r_[False, above[:-1]], below & np.r_[False, below[:-1]])
    return first_per_group(sig, clk.fx_day)


def sig_auction(df, atr, cfg):
    """Auction market theory, failed auction: a look above yesterday's VAH that closes
    back inside value (look below VAL mirrored)."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    _, val, vah = previous_day_profile(h, l, volume(df), clk.fx_day)
    pc = np.r_[np.nan, c[:-1]]
    short = (h > vah) & (c < vah) & (pc <= vah + 0.5 * atr) & (c < o)
    long_ = (l < val) & (c > val) & (pc >= val - 0.5 * atr) & (c > o)
    return first_per_group(to_signal(long_, short), clk.fx_day)


def sig_delta(df, atr, cfg):
    """Delta (proxy): 5-bar delta more than 2 standard deviations from its 50-bar mean,
    with price closing at a 5-bar extreme in the same direction."""
    o, h, l, c = ohlc(df)
    d5 = pd.Series(_delta(df)).rolling(5).sum()
    z = ((d5 - d5.shift(1).rolling(50).mean()) / (d5.shift(1).rolling(50).std() + EPS)).to_numpy()
    hi5 = pd.Series(h).shift(1).rolling(5).max().to_numpy()
    lo5 = pd.Series(l).shift(1).rolling(5).min().to_numpy()
    return to_signal((z > 2) & (c > hi5), (z < -2) & (c < lo5))


def sig_cvd_divergence(df, atr, cfg):
    """Cumulative volume delta divergence (proxy): a new 20-bar low while the session
    CVD stays above its own 20-bar low (mirrored)."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    cvd = pd.Series(_delta(df)).groupby(pd.Series(clk.fx_day)).cumsum()
    lo20 = pd.Series(l).shift(1).rolling(20).min().to_numpy()
    hi20 = pd.Series(h).shift(1).rolling(20).max().to_numpy()
    cvd_lo = cvd.shift(1).rolling(20).min().to_numpy()
    cvd_hi = cvd.shift(1).rolling(20).max().to_numpy()
    cv = cvd.to_numpy()
    return to_signal((l < lo20) & (cv > cvd_lo) & (c > o), (h > hi20) & (cv < cvd_hi) & (c < o))


def sig_absorption(df, atr, cfg):
    """Absorption (proxy): twice the average tick volume on a bar narrower than 0.6 ATR
    at a 20-bar extreme -> the extreme holds."""
    o, h, l, c = ohlc(df)
    v = volume(df)
    avg = pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    lo20 = pd.Series(l).shift(1).rolling(20).min().to_numpy()
    hi20 = pd.Series(h).shift(1).rolling(20).max().to_numpy()
    heavy = (v >= 2 * avg) & ((h - l) < 0.6 * atr)
    return to_signal(heavy & (l <= lo20 + 0.2 * atr), heavy & (h >= hi20 - 0.2 * atr))


def sig_volume_imbalance(df, atr, cfg):
    """Volume imbalance (ICT): a gap between the BODIES of two consecutive candles
    whose wicks overlap; trade its first retest."""
    o, h, l, c = ohlc(df)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    zones = []
    for t in range(1, len(c)):
        a = atr[t]
        if not np.isfinite(a):
            continue
        if bot[t] - top[t - 1] >= 0.05 * a and l[t] <= h[t - 1] and c[t] > o[t]:
            zones.append(Zone(t, top[t - 1], bot[t], 1))
        elif bot[t - 1] - top[t] >= 0.05 * a and h[t] >= l[t - 1] and c[t] < o[t]:
            zones.append(Zone(t, top[t], bot[t - 1], -1))
    return retest_signals(zones, h, l, c, max_age=60)


def sig_orderflow_scalp(df, atr, cfg):
    """Order-flow scalping (proxy): three bars in a row whose delta and close agree,
    with rising tick volume, closing beyond the 3-bar range."""
    o, h, l, c = ohlc(df)
    d = _delta(df)
    v = volume(df)
    ds, vs = pd.Series(d), pd.Series(v)
    pos3 = ((ds > 0) & (ds.shift(1) > 0) & (ds.shift(2) > 0)).to_numpy()
    neg3 = ((ds < 0) & (ds.shift(1) < 0) & (ds.shift(2) < 0)).to_numpy()
    rising = ((vs > vs.shift(1)) & (vs.shift(1) > vs.shift(2))).to_numpy()
    hi3 = pd.Series(h).shift(1).rolling(3).max().to_numpy()
    lo3 = pd.Series(l).shift(1).rolling(3).min().to_numpy()
    return to_signal(pos3 & rising & (c > hi3), neg3 & rising & (c < lo3))


GENERATORS = {
    "volume_profile": sig_volume_profile, "poc": sig_poc, "frvp": sig_frvp,
    "tpo_80": sig_tpo_80, "vah_val": sig_vah_val, "auction": sig_auction,
    "delta": sig_delta, "cvd_divergence": sig_cvd_divergence, "absorption": sig_absorption,
    "volume_imbalance": sig_volume_imbalance, "orderflow_scalp": sig_orderflow_scalp,
}

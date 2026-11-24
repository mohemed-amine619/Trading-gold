"""
core/strategy/library/smc.py — Smart Money Concepts / ICT structure strategies.

Swings are published only once confirmed (k bars after the pivot), structure
breaks are measured on closes, and every zone (order block, FVG, breaker ...)
is traded on its first retest after the bar that created it. Causal throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import (
    EPS, Structure, Zone, closed_session_range, fvg_zones, in_window, invert, ny_clock, ohlc,
    order_block_zones, previous_group_hlc, retest_signals, structure, swings, to_signal,
    first_per_group, group_open,
)


def _clock(df, cfg):
    return ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))


def sig_bos(df, atr, cfg):
    """Break of Structure: a close through the last swing in the direction of the trend."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    return to_signal(s.event == 1, s.event == -1)


def sig_choch(df, atr, cfg):
    """Change of Character: the first close through a swing AGAINST the prior trend."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    return to_signal(s.event == 2, s.event == -2)


def sig_mss(df, atr, cfg):
    """Market Structure Shift: a CHOCH made by a displacement candle (>= 1.5 ATR)
    after the last swing on the other side swept the one before it."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    sw = swings(h, l, 3)
    disp = (h - l) >= 1.5 * atr
    long_ = (s.event == 2) & disp & (sw.sl1 < sw.sl2)
    short = (s.event == -2) & disp & (sw.sh1 > sw.sh2)
    return to_signal(long_, short)


def sig_smc(df, atr, cfg):
    """SMC composite: bullish structure + price in the discount half of the last
    swing range + a fresh bullish FVG on this bar (mirrored for shorts)."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    sw = swings(h, l, 3)
    eq = (sw.sh1 + sw.sl1) / 2
    h2, l2 = np.r_[np.nan, np.nan, h[:-2]], np.r_[np.nan, np.nan, l[:-2]]
    bull_fvg = (l - h2) >= 0.15 * atr
    bear_fvg = (l2 - h) >= 0.15 * atr
    st = s.state
    long_ = (st == 1) & bull_fvg & (l < eq)
    short = (st == -1) & bear_fvg & (h > eq)
    return to_signal(long_, short)


def sig_order_block(df, atr, cfg):
    """Order block: the last opposite candle before a break of structure; trade its
    first retest."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    return retest_signals(order_block_zones(o, h, l, c, s), h, l, c, max_age=150)


def sig_breaker(df, atr, cfg):
    """Breaker block: an order block that price closed through flips sides."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    obs = order_block_zones(o, h, l, c, s)
    return retest_signals(invert(obs, h, l, c), h, l, c, max_age=150)


def sig_mitigation(df, atr, cfg):
    """Mitigation block: the order block of a CHOCH that came from a FAILURE swing
    (a lower high before a bearish break, a higher low before a bullish one)."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    sw = swings(h, l, 3)
    ev = s.event.copy()
    keep = ((ev == 2) & (sw.sl1 > sw.sl2)) | ((ev == -2) & (sw.sh1 < sw.sh2))
    ev[~keep] = 0
    filtered = Structure(ev, s.level, s.state)
    return retest_signals(order_block_zones(o, h, l, c, filtered, events=(2, -2)), h, l, c,
                          max_age=150)


def sig_fvg(df, atr, cfg):
    """Fair Value Gap: trade the first retrace into a fresh 3-bar imbalance."""
    o, h, l, c = ohlc(df)
    return retest_signals(fvg_zones(h, l, c, o, atr), h, l, c, max_age=60)


def sig_ifvg(df, atr, cfg):
    """Inverse FVG: a gap that price closed through, traded from the other side."""
    o, h, l, c = ohlc(df)
    return retest_signals(invert(fvg_zones(h, l, c, o, atr), h, l, c, 120), h, l, c, max_age=60)


def _liquidity_levels(df, atr, cfg, side: int):
    """Equal highs/lows (two swings within 0.15 ATR) and the previous day's extreme."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    clk = _clock(df, cfg)
    pdh, pdl, _, _ = previous_group_hlc(h, l, c, clk.fx_day)
    if side == 1:
        eq = np.where(np.abs(sw.sh1 - sw.sh2) <= 0.15 * atr, np.fmax(sw.sh1, sw.sh2), np.nan)
        return eq, pdh
    eq = np.where(np.abs(sw.sl1 - sw.sl2) <= 0.15 * atr, np.fmin(sw.sl1, sw.sl2), np.nan)
    return eq, pdl


def sig_bsl_sweep(df, atr, cfg):
    """Buy-side liquidity: a wick above equal highs or the previous day's high that
    closes back below -> short."""
    o, h, l, c = ohlc(df)
    eq, pdh = _liquidity_levels(df, atr, cfg, 1)
    short = ((h > eq) & (c < eq)) | ((h > pdh) & (c < pdh))
    return to_signal(np.zeros(len(c), bool), short & (c < o))


def sig_ssl_sweep(df, atr, cfg):
    """Sell-side liquidity: a wick below equal lows or the previous day's low that
    closes back above -> long."""
    o, h, l, c = ohlc(df)
    eq, pdl = _liquidity_levels(df, atr, cfg, -1)
    long_ = ((l < eq) & (c > eq)) | ((l < pdl) & (c > pdl))
    return to_signal(long_ & (c > o), np.zeros(len(c), bool))


def sig_inducement(df, atr, cfg):
    """Inducement: inside a bullish structure, price sweeps the latest MINOR swing low
    (1-bar pivot) but holds above the major one, then closes back up."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    minor = swings(h, l, 1)
    major = swings(h, l, 3)
    prev_state = np.r_[0, s.state[:-1]]
    long_ = ((prev_state == 1) & (l < minor.sl1) & (c > minor.sl1) & (l > major.sl1) & (c > o))
    short = ((prev_state == -1) & (h > minor.sh1) & (c < minor.sh1) & (h < major.sh1) & (c < o))
    return to_signal(long_, short)


def sig_premium_discount(df, atr, cfg):
    """Premium & discount: in a bullish structure, buy a bar that closes above the
    prior high while in the discount half of the 50-bar dealing range (mirrored)."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    hi = pd.Series(h).rolling(50).max().to_numpy()
    lo = pd.Series(l).rolling(50).min().to_numpy()
    eq = (hi + lo) / 2
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    long_ = (s.state == 1) & (c < eq) & (c > ph)
    short = (s.state == -1) & (c > eq) & (c < pl)
    return to_signal(long_, short)


def sig_ote(df, atr, cfg):
    """Optimal Trade Entry: a retrace into the 62-79% zone of the last impulse leg
    (>= 2 ATR) that closes back in the leg's direction."""
    o, h, l, c = ohlc(df)
    sw = swings(h, l, 3)
    leg = sw.sh1 - sw.sl1
    up = (sw.sl1_i < sw.sh1_i) & (leg >= 2 * atr)
    dn = (sw.sh1_i < sw.sl1_i) & (leg >= 2 * atr)
    up_top, up_bot = sw.sh1 - 0.62 * leg, sw.sh1 - 0.79 * leg
    dn_bot, dn_top = sw.sl1 + 0.62 * leg, sw.sl1 + 0.79 * leg
    long_ = up & (l <= up_top) & (l >= up_bot) & (c > o) & (l > sw.sl1)
    short = dn & (h >= dn_bot) & (h <= dn_top) & (c < o) & (h < sw.sh1)
    return to_signal(long_, short)


def sig_po3(df, atr, cfg):
    """Power of Three: accumulation (Asian range), manipulation (London sweeps one side),
    distribution (close back inside -> trade toward the other side). Once a day."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    asia = in_window(clk.ny_hour, 19.0, 24.0)
    ahi, alo = closed_session_range(h, l, clk.fx_day, asia)
    active = in_window(clk.ny_hour, 1.0, 11.0, clk.bar_minutes)
    long_ = active & (l < alo) & (c > alo) & (c > o)
    short = active & (h > ahi) & (c < ahi) & (c < o)
    return first_per_group(to_signal(long_, short), clk.fx_day)


def sig_judas(df, atr, cfg):
    """Judas swing: between 00:00 and 05:00 New York the market runs above the
    midnight open and the Asian high, then closes back below the midnight open."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    is_mid = (clk.ny_hour >= 0) & (clk.ny_hour < 6)
    day = clk.fx_day
    mid_open = np.full(len(c), np.nan)
    cur, val = None, np.nan
    for t in range(len(c)):
        if day[t] != cur:
            cur, val = day[t], np.nan
        if np.isnan(val) and clk.ny_hour[t] >= 0 and clk.ny_hour[t] < 12 and is_mid[t]:
            val = o[t]
        mid_open[t] = val
    asia = in_window(clk.ny_hour, 19.0, 24.0)
    ahi, alo = closed_session_range(h, l, day, asia)
    run_hi = pd.Series(h).groupby(pd.Series(day)).cummax().to_numpy()
    run_lo = pd.Series(l).groupby(pd.Series(day)).cummin().to_numpy()
    window = in_window(clk.ny_hour, 0.0, 5.0, clk.bar_minutes)
    short = window & (run_hi > ahi) & (c < mid_open) & (np.r_[np.nan, c[:-1]] >= mid_open)
    long_ = window & (run_lo < alo) & (c > mid_open) & (np.r_[np.nan, c[:-1]] <= mid_open)
    return first_per_group(to_signal(long_, short), day)


def sig_turtle_soup(df, atr, cfg):
    """Turtle soup: a new 20-bar low whose previous low is at least 4 bars old,
    closed back above that old low (mirrored)."""
    o, h, l, c = ohlc(df)
    lo_s, hi_s = pd.Series(l), pd.Series(h)
    prior_low = lo_s.shift(1).rolling(20).min()
    age_low = lo_s.shift(1).rolling(20).apply(lambda w: len(w) - 1 - int(np.argmin(w)), raw=True)
    prior_high = hi_s.shift(1).rolling(20).max()
    age_high = hi_s.shift(1).rolling(20).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True)
    long_ = (lo_s < prior_low) & (age_low >= 3) & (pd.Series(c) > prior_low)
    short = (hi_s > prior_high) & (age_high >= 3) & (pd.Series(c) < prior_high)
    return to_signal(long_.to_numpy(), short.to_numpy())


def sig_silver_bullet(df, atr, cfg):
    """ICT Silver Bullet: an FVG formed inside 03-04, 10-11 or 14-15 New York,
    traded on its retest within the hour."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    win = (in_window(clk.ny_hour, 3.0, 4.0, clk.bar_minutes)
           | in_window(clk.ny_hour, 10.0, 11.0, clk.bar_minutes)
           | in_window(clk.ny_hour, 14.0, 15.0, clk.bar_minutes))
    zones = [z for z in fvg_zones(h, l, c, o, atr) if win[z.created]]
    bars_per_hour = max(1, int(round(60.0 / max(clk.bar_minutes, 1.0))))
    return retest_signals(zones, h, l, c, max_age=bars_per_hour + 1)


def sig_unicorn(df, atr, cfg):
    """Unicorn model: a breaker block overlapping a same-direction FVG formed within
    10 bars of it; trade the retest of the overlap."""
    o, h, l, c = ohlc(df)
    s = structure(h, l, c, 3)
    breakers = invert(order_block_zones(o, h, l, c, s), h, l, c)
    fvgs = fvg_zones(h, l, c, o, atr)
    zones = []
    for b in breakers:
        for f in fvgs:
            if f.direction == b.direction and abs(f.created - b.created) <= 10:
                lo, hi = max(b.lo, f.lo), min(b.hi, f.hi)
                if hi > lo:
                    zones.append(Zone(max(b.created, f.created), lo, hi, b.direction))
    return retest_signals(zones, h, l, c, max_age=100)


GENERATORS = {
    "smc": sig_smc, "bos": sig_bos, "choch": sig_choch, "mss": sig_mss,
    "order_block": sig_order_block, "breaker": sig_breaker, "mitigation": sig_mitigation,
    "fvg": sig_fvg, "ifvg": sig_ifvg, "bsl_sweep": sig_bsl_sweep, "ssl_sweep": sig_ssl_sweep,
    "inducement": sig_inducement, "premium_discount": sig_premium_discount, "ote": sig_ote,
    "po3": sig_po3, "judas": sig_judas, "turtle_soup": sig_turtle_soup,
    "silver_bullet": sig_silver_bullet, "unicorn": sig_unicorn,
}

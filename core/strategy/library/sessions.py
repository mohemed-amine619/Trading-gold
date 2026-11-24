"""
core/strategy/library/sessions.py — time-based ICT models and intraday session
strategies.

All session times are New York times, converted from the data's own clock with
``cfg.server_utc_offset`` (hours the broker clock is ahead of UTC; Exness = 0)
and US daylight-saving rules. The trading day starts at 18:00 New York, as for
FX and gold. A session's range is used only after the session has finished.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import (
    EPS, bars_in_group, closed_session_range, ema, first_per_group, fvg_zones, in_window,
    ny_clock, ohlc, previous_group_hlc, retest_signals, session_range, swings, to_signal, vwap,
    Zone,
)


def _clock(df, cfg):
    return ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))


def _midnight_open(o, clk) -> np.ndarray:
    """Open of the first bar at or after 00:00 New York, per trading day."""
    out = np.full(len(o), np.nan)
    cur, val = None, np.nan
    for t in range(len(o)):
        if clk.fx_day[t] != cur:
            cur, val = clk.fx_day[t], np.nan
        if np.isnan(val) and clk.ny_hour[t] < 17.0:
            val = o[t]
        out[t] = val
    return out


def _range_break(df, cfg, start, end, trade_start, trade_end):
    """First close beyond a finished session range inside a later trading window."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    mask = in_window(clk.ny_hour, start, end)
    hi, lo = closed_session_range(h, l, clk.fx_day, mask)
    trade = in_window(clk.ny_hour, trade_start, trade_end, clk.bar_minutes) & ~mask
    sig = to_signal(trade & (c > hi), trade & (c < lo))
    return first_per_group(sig, clk.fx_day)


def _range_sweep(df, cfg, start, end, trade_start, trade_end):
    """A wick beyond a finished session range that closes back inside -> fade."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    mask = in_window(clk.ny_hour, start, end)
    hi, lo = closed_session_range(h, l, clk.fx_day, mask)
    trade = in_window(clk.ny_hour, trade_start, trade_end, clk.bar_minutes) & ~mask
    sig = to_signal(trade & (l < lo) & (c > lo) & (c > o), trade & (h > hi) & (c < hi) & (c < o))
    return first_per_group(sig, clk.fx_day)


# ── ICT time-based ───────────────────────────────────────────────────────────

def sig_london_killzone(df, atr, cfg):
    """London kill zone (02-05 NY): fade a sweep of the Asian range."""
    return _range_sweep(df, cfg, 19.0, 24.0, 2.0, 5.0)


def sig_ny_killzone(df, atr, cfg):
    """New York kill zone (07-10 NY): fade a sweep of the London session range."""
    return _range_sweep(df, cfg, 2.0, 5.0, 7.0, 10.0)


def sig_asian_range(df, atr, cfg):
    """Asian range: before London (00-02 NY) fade a test of the finished Asian range edges."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    mask = in_window(clk.ny_hour, 19.0, 24.0)
    hi, lo = closed_session_range(h, l, clk.fx_day, mask)
    w = hi - lo
    trade = in_window(clk.ny_hour, 0.0, 2.0, clk.bar_minutes) & ~mask
    long_ = trade & (l <= lo + 0.1 * w) & (c > o) & (c > lo)
    short = trade & (h >= hi - 0.1 * w) & (c < o) & (c < hi)
    return first_per_group(to_signal(long_, short), clk.fx_day)


def sig_london_open(df, atr, cfg):
    """London open / Asian range breakout: first close beyond the Asian range, 02-05 NY."""
    return _range_break(df, cfg, 19.0, 24.0, 2.0, 5.0)


def sig_ny_open(df, atr, cfg):
    """New York open: first close beyond the 09:30-10:00 opening range, 10:00-12:00 NY."""
    return _range_break(df, cfg, 9.5, 10.0, 10.0, 12.0)


def sig_london_close_reversal(df, atr, cfg):
    """London close (10-12 NY): after a day move of 1.5 ATR from the midnight open,
    fade it on a close through the previous bar's extreme."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    mo = _midnight_open(o, clk)
    move = (c - mo) / (atr + EPS)
    pm = np.r_[np.nan, move[:-1]]
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    trade = in_window(clk.ny_hour, 10.0, 12.0, clk.bar_minutes)
    sig = to_signal(trade & (pm < -1.5) & (c > ph), trade & (pm > 1.5) & (c < pl))
    return first_per_group(sig, clk.fx_day)


def sig_ict_2022(df, atr, cfg):
    """ICT 2022 model: a sweep of the last swing low (high), then within 12 bars a
    displacement that leaves an FVG the other way; trade the FVG retest. 02-12 NY."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    sw = swings(h, l, 3)
    swept_low = (l < sw.sl1) & (c > sw.sl1 - 0.5 * atr)
    swept_high = (h > sw.sh1) & (c < sw.sh1 + 0.5 * atr)
    last_low_sweep = pd.Series(np.where(swept_low, np.arange(len(c)), np.nan)).ffill().to_numpy()
    last_high_sweep = pd.Series(np.where(swept_high, np.arange(len(c)), np.nan)).ffill().to_numpy()
    win = in_window(clk.ny_hour, 2.0, 12.0, clk.bar_minutes)
    zones = []
    for z in fvg_zones(h, l, c, o, atr, 0.2):
        t = z.created
        if not win[t] or (h[t - 1] - l[t - 1]) < 1.2 * atr[t - 1]:
            continue
        ref = last_low_sweep[t] if z.direction == 1 else last_high_sweep[t]
        if np.isfinite(ref) and 0 < t - ref <= 12:
            zones.append(z)
    return retest_signals(zones, h, l, c, max_age=16)


_MACROS = ((2 + 33 / 60, 3.0), (4 + 3 / 60, 4.5), (8 + 50 / 60, 9 + 10 / 60),
           (9 + 50 / 60, 10 + 10 / 60), (10 + 50 / 60, 11 + 10 / 60), (11 + 50 / 60, 12 + 10 / 60),
           (13 + 10 / 60, 13 + 40 / 60), (15 + 15 / 60, 15.75))


def sig_ict_macros(df, atr, cfg):
    """ICT macros: a displacement close beyond the prior 4-bar range during one of the
    algorithmic 20-30 minute windows."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    win = np.zeros(len(c), dtype=bool)
    for a, b in _MACROS:
        win |= in_window(clk.ny_hour, a, b, clk.bar_minutes)
    hi4 = pd.Series(h).shift(1).rolling(4).max().to_numpy()
    lo4 = pd.Series(l).shift(1).rolling(4).min().to_numpy()
    body = np.abs(c - o)
    return to_signal(win & (c > hi4) & (body > 0.5 * atr), win & (c < lo4) & (body > 0.5 * atr))


def sig_weekly_profile(df, atr, cfg):
    """ICT weekly profile: early in the week (Mon-Wed) price sweeps the previous week's
    low (high) and closes back inside -> the week's low (high) is likely in."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    pwh, pwl, _, _ = previous_group_hlc(h, l, c, clk.week)
    early = clk.fx_weekday <= 2
    sig = to_signal(early & (l < pwl) & (c > pwl), early & (h > pwh) & (c < pwh))
    return first_per_group(sig, clk.week)


def _daily_bias(h, l, c, clk) -> np.ndarray:
    """+1 / -1 from the previous day's candle: a close above the day before's high is
    bullish, below its low bearish, otherwise its close against its own midpoint."""
    agg = pd.DataFrame({"h": h, "l": l, "c": c, "g": clk.fx_day}).groupby("g").agg(
        h=("h", "max"), l=("l", "min"), c=("c", "last"))
    bias = np.where(agg["c"] > agg["h"].shift(1), 1,
                    np.where(agg["c"] < agg["l"].shift(1), -1,
                             np.where(agg["c"] > (agg["h"] + agg["l"]) / 2, 1, -1)))
    prev = pd.Series(bias, index=agg.index).shift(1)
    return prev.reindex(clk.fx_day).fillna(0).to_numpy()


def sig_daily_bias(df, atr, cfg):
    """ICT daily bias: with the previous day's bias, buy (sell) the first rejection of
    the previous day's midpoint inside the London or New York kill zone."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    bias = _daily_bias(h, l, c, clk)
    pdh, pdl, _, _ = previous_group_hlc(h, l, c, clk.fx_day)
    mid = (pdh + pdl) / 2
    kz = in_window(clk.ny_hour, 2.0, 5.0, clk.bar_minutes) | in_window(clk.ny_hour, 7.0, 10.0, clk.bar_minutes)
    sig = to_signal(kz & (bias == 1) & (l <= mid) & (c > mid) & (c > o),
                    kz & (bias == -1) & (h >= mid) & (c < mid) & (c < o))
    return first_per_group(sig, clk.fx_day)


def sig_midnight_open(df, atr, cfg):
    """ICT midnight open: with a rising one-day EMA, buy a 07-11 NY close back above the
    midnight open after trading below it (mirrored)."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    mo = _midnight_open(o, clk)
    per_day = max(4, int(round(24 * 60 / max(clk.bar_minutes, 1.0))))
    e = ema(c, per_day).to_numpy()
    up = e > np.r_[np.full(4, np.nan), e[:-4]]
    win = in_window(clk.ny_hour, 7.0, 11.0, clk.bar_minutes)
    pc = np.r_[np.nan, c[:-1]]
    sig = to_signal(win & up & (pc < mo) & (c > mo), win & ~up & (pc > mo) & (c < mo) & np.isfinite(e))
    return first_per_group(sig, clk.fx_day)


# ── day trading ──────────────────────────────────────────────────────────────

def sig_initial_balance(df, atr, cfg):
    """Initial balance: first close beyond the 09:30-10:30 NY range, until 14:00 NY."""
    return _range_break(df, cfg, 9.5, 10.5, 10.5, 14.0)


def sig_london_breakout(df, atr, cfg):
    """London breakout: first close beyond London's first hour (03-04 NY), until 08:00."""
    return _range_break(df, cfg, 3.0, 4.0, 4.0, 8.0)


def sig_ny_breakout(df, atr, cfg):
    """New York breakout: first close beyond the 08:00-09:00 NY range, until 12:00."""
    return _range_break(df, cfg, 8.0, 9.0, 9.0, 12.0)


def sig_pdh_pdl(df, atr, cfg):
    """Previous day high/low breakout: first close beyond yesterday's high or low."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    pdh, pdl, _, _ = previous_group_hlc(h, l, c, clk.fx_day)
    return first_per_group(to_signal(c > pdh, c < pdl), clk.fx_day)


def sig_pwh_pwl(df, atr, cfg):
    """Previous week high/low breakout: first close beyond last week's high or low."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    pwh, pwl, _, _ = previous_group_hlc(h, l, c, clk.week)
    return first_per_group(to_signal(c > pwh, c < pwl), clk.week)


def sig_pd_sweep(df, atr, cfg):
    """Liquidity grab: a wick through yesterday's high/low that closes back inside -> fade."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    pdh, pdl, _, _ = previous_group_hlc(h, l, c, clk.fx_day)
    sig = to_signal((l < pdl) & (c > pdl) & (c > o), (h > pdh) & (c < pdh) & (c < o))
    return first_per_group(sig, clk.fx_day)


def sig_gap(df, atr, cfg):
    """Gap trading: the week's opening gap (>= 0.5 ATR from Friday's close) is faded on
    the first bar that closes toward the fill within the first 8 bars."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    n = len(c)
    sig = np.zeros(n, dtype=np.int8)
    starts = np.flatnonzero(np.r_[False, np.diff(clk.week) != 0])
    for s in starts:
        a = atr[s - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        gap = o[s] - c[s - 1]
        if abs(gap) < 0.5 * a:
            continue
        for t in range(s, min(n, s + 8)):
            if gap > 0 and c[t] < o[t] and c[t] > c[s - 1]:
                sig[t] = -1
                break
            if gap < 0 and c[t] > o[t] and c[t] < c[s - 1]:
                sig[t] = 1
                break
    return sig


def sig_trend_intraday(df, atr, cfg):
    """Intraday trend: above a rising session VWAP, buy a pull-back to EMA20 that
    closes back above it (mirrored)."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    vw, _ = vwap(df, clk.fx_day)
    vw = vw.to_numpy()
    e = ema(c, 20).to_numpy()
    e_up = e > np.r_[np.full(3, np.nan), e[:-3]]
    e_dn = e < np.r_[np.full(3, np.nan), e[:-3]]
    started = bars_in_group(clk.fx_day) >= 8
    long_ = started & (c > vw) & e_up & (l <= e) & (c > e) & (c > o)
    short = started & (c < vw) & e_dn & (h >= e) & (c < e) & (c < o)
    return to_signal(long_, short)


def sig_vol_breakout(df, atr, cfg):
    """Volatility breakout (Larry Williams): first close beyond today's open +/- half of
    yesterday's range."""
    o, h, l, c = ohlc(df)
    clk = _clock(df, cfg)
    pdh, pdl, _, _ = previous_group_hlc(h, l, c, clk.fx_day)
    dopen = pd.Series(o).groupby(pd.Series(clk.fx_day)).transform("first").to_numpy()
    k = 0.5 * (pdh - pdl)
    return first_per_group(to_signal(c > dopen + k, c < dopen - k), clk.fx_day)


GENERATORS = {
    "london_killzone": sig_london_killzone, "ny_killzone": sig_ny_killzone,
    "asian_range": sig_asian_range, "london_open": sig_london_open, "ny_open": sig_ny_open,
    "london_close_reversal": sig_london_close_reversal, "ict_2022": sig_ict_2022,
    "ict_macros": sig_ict_macros, "weekly_profile": sig_weekly_profile,
    "daily_bias": sig_daily_bias, "midnight_open": sig_midnight_open,
    "initial_balance": sig_initial_balance, "london_breakout": sig_london_breakout,
    "ny_breakout": sig_ny_breakout, "pdh_pdl": sig_pdh_pdl, "pwh_pwl": sig_pwh_pwl,
    "pd_sweep": sig_pd_sweep, "gap": sig_gap, "trend_intraday": sig_trend_intraday,
    "vol_breakout": sig_vol_breakout,
}

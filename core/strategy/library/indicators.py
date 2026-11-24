"""
core/strategy/library/indicators.py — classic technical-indicator strategies,
plus the indicator-based scalping, swing and quantitative rules.

Each function: ``(df, atr, cfg) -> int8 array``, causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.library._common import (
    EPS, adx, bollinger, cross_down, cross_up, ema, ny_clock, ohlc, previous_group_hlc, rsi,
    sma, swings, to_signal, volume, vwap,
)


# ── oscillators / crosses ────────────────────────────────────────────────────

def sig_rsi(df, atr, cfg):
    """RSI(14) crosses back up through 30 (down through 70)."""
    r = rsi(df["close"], 14).to_numpy()
    return to_signal(cross_up(r, 30.0), cross_down(r, 70.0))


def sig_ma_cross(df, atr, cfg):
    """Moving-average crossover: SMA 10 / SMA 30."""
    c = df["close"].to_numpy(dtype=float)
    f, s = sma(c, 10).to_numpy(), sma(c, 30).to_numpy()
    return to_signal(cross_up(f, s), cross_down(f, s))


def sig_ema_9_21(df, atr, cfg):
    """EMA 9 / EMA 21 crossover."""
    c = df["close"].to_numpy(dtype=float)
    f, s = ema(c, 9).to_numpy(), ema(c, 21).to_numpy()
    return to_signal(cross_up(f, s), cross_down(f, s))


def sig_ema_50_200(df, atr, cfg):
    """EMA 50 / EMA 200 golden / death cross."""
    c = df["close"].to_numpy(dtype=float)
    f, s = ema(c, 50).to_numpy(), ema(c, 200).to_numpy()
    return to_signal(cross_up(f, s), cross_down(f, s))


def sig_sma_stack(df, atr, cfg):
    """SMA 20/50/200 stacked; buy the pull-back to the SMA20 that closes back above it."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    s20, s50, s200 = (sma(c, n).to_numpy() for n in (20, 50, 200))
    up = (s20 > s50) & (s50 > s200)
    dn = (s20 < s50) & (s50 < s200)
    return to_signal(up & (l <= s20) & (c > s20) & (c > o), dn & (h >= s20) & (c < s20) & (c < o))


def sig_bollinger(df, atr, cfg):
    """Bollinger(20, 2) reversion: a close back inside after a close outside."""
    c = df["close"].to_numpy(dtype=float)
    lo, mid, hi = (x.to_numpy() for x in bollinger(c, 20, 2.0))
    pc = np.r_[np.nan, c[:-1]]
    plo, phi = np.r_[np.nan, lo[:-1]], np.r_[np.nan, hi[:-1]]
    return to_signal((pc < plo) & (c > lo), (pc > phi) & (c < hi))


def sig_adx_trend(df, atr, cfg):
    """ADX trend: +DI crosses -DI while ADX(14) is above 25."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    a, p, m = adx(h, l, c, 14)
    strong = a.to_numpy() > 25
    return to_signal(strong & cross_up(p.to_numpy(), m.to_numpy()),
                     strong & cross_down(p.to_numpy(), m.to_numpy()))


def sig_ichimoku(df, atr, cfg):
    """Ichimoku: Tenkan/Kijun cross on the right side of the cloud (cloud shifted 26 bars
    forward, i.e. built only from data 26 bars old)."""
    h, l, c = (pd.Series(df[x].to_numpy(dtype=float)) for x in ("high", "low", "close"))
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top = np.fmax(span_a, span_b).to_numpy()
    bot = np.fmin(span_a, span_b).to_numpy()
    cc = c.to_numpy()
    return to_signal(cross_up(tenkan, kijun) & (cc > top), cross_down(tenkan, kijun) & (cc < bot))


def _psar(h, l, step=0.02, max_step=0.2) -> np.ndarray:
    n = len(h)
    trend = np.zeros(n, dtype=np.int8)
    if n < 3:
        return trend
    up = True
    sar, ep, af = l[0], h[0], step
    for t in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[t - 1], l[t - 2] if t > 1 else l[t - 1])
            if l[t] < sar:
                up, sar, ep, af = False, ep, l[t], step
            elif h[t] > ep:
                ep, af = h[t], min(af + step, max_step)
        else:
            sar = max(sar, h[t - 1], h[t - 2] if t > 1 else h[t - 1])
            if h[t] > sar:
                up, sar, ep, af = True, ep, h[t], step
            elif l[t] < ep:
                ep, af = l[t], min(af + step, max_step)
        trend[t] = 1 if up else -1
    return trend


def sig_psar(df, atr, cfg):
    """Parabolic SAR flip."""
    tr = _psar(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float))
    prev = np.r_[0, tr[:-1]]
    sig = np.where((tr == 1) & (prev == -1), 1, np.where((tr == -1) & (prev == 1), -1, 0))
    sig[:20] = 0
    return sig.astype(np.int8)


def sig_atr_breakout(df, atr, cfg):
    """ATR breakout: a close-to-close move larger than 1.2 x the previous bar's ATR."""
    c = df["close"].to_numpy(dtype=float)
    pc = np.r_[np.nan, c[:-1]]
    pa = np.r_[np.nan, atr[:-1]]
    return to_signal(c - pc > 1.2 * pa, pc - c > 1.2 * pa)


def sig_vwap_trend(df, atr, cfg):
    """VWAP: a close crossing the session VWAP in the direction of its 4-bar slope."""
    c = df["close"].to_numpy(dtype=float)
    clk = ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))
    vw, _ = vwap(df, clk.fx_day)
    v = vw.to_numpy()
    slope = v - np.r_[np.full(4, np.nan), v[:-4]]
    return to_signal(cross_up(c, v) & (slope > 0), cross_down(c, v) & (slope < 0))


def sig_vwap_bands(df, atr, cfg):
    """VWAP bands: a close back inside the 2-sigma volume-weighted band."""
    c = df["close"].to_numpy(dtype=float)
    clk = ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))
    vw, sd = vwap(df, clk.fx_day)
    lo, hi = (vw - 2 * sd).to_numpy(), (vw + 2 * sd).to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    plo, phi = np.r_[np.nan, lo[:-1]], np.r_[np.nan, hi[:-1]]
    ready = pd.Series(clk.fx_day).groupby(clk.fx_day).cumcount().to_numpy() >= 12
    return to_signal(ready & (pc < plo) & (c > lo), ready & (pc > phi) & (c < hi))


# ── Fibonacci / pivots ───────────────────────────────────────────────────────

def sig_fib_retracement(df, atr, cfg):
    """Fibonacci retracement: a pull-back into 50-61.8% of the last major leg (>= 3 ATR)
    that closes back in the leg's direction."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    sw = swings(h, l, 5)
    leg = sw.sh1 - sw.sl1
    up = (sw.sl1_i < sw.sh1_i) & (leg >= 3 * atr)
    dn = (sw.sh1_i < sw.sl1_i) & (leg >= 3 * atr)
    long_ = up & (l <= sw.sh1 - 0.5 * leg) & (l >= sw.sh1 - 0.618 * leg) & (c > o)
    short = dn & (h >= sw.sl1 + 0.5 * leg) & (h <= sw.sl1 + 0.618 * leg) & (c < o)
    return to_signal(long_, short)


def sig_fib_extension(df, atr, cfg):
    """Fibonacci extension: exhaustion at the 1.618 extension of the last leg (A->B,
    projected from C) on a rejection candle -> fade."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    sw = swings(h, l, 3)
    # bullish ABC: A = sl2, B = sh1, C = sl1 (A < B, C after B) -> target C + 1.618 (B - A)
    up_abc = (sw.sl2_i < sw.sh1_i) & (sw.sh1_i < sw.sl1_i) & (sw.sl1 > sw.sl2)
    up_t = sw.sl1 + 1.618 * (sw.sh1 - sw.sl2)
    dn_abc = (sw.sh2_i < sw.sl1_i) & (sw.sl1_i < sw.sh1_i) & (sw.sh1 < sw.sh2)
    dn_t = sw.sh1 - 1.618 * (sw.sh2 - sw.sl1)
    short = up_abc & (h >= up_t) & (c < up_t) & (c < o)
    long_ = dn_abc & (l <= dn_t) & (c > dn_t) & (c > o)
    return to_signal(long_, short)


def sig_pivot_points(df, atr, cfg):
    """Classic daily pivots: rejection of S1 (buy) or R1 (sell)."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    clk = ny_clock(df.index, float(getattr(cfg, "server_utc_offset", 0.0)))
    ph, pl, pc, _ = previous_group_hlc(h, l, c, clk.fx_day)
    p = (ph + pl + pc) / 3
    s1, r1 = 2 * p - ph, 2 * p - pl
    return to_signal((l <= s1) & (c > s1) & (c > o), (h >= r1) & (c < r1) & (c < o))


# ── channels / trend tools ───────────────────────────────────────────────────

def _supertrend(h, l, c, atr, mult=3.0) -> np.ndarray:
    n = len(c)
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    for t in range(n):
        if not np.isfinite(atr[t]):
            continue
        if t == 0 or not np.isfinite(fu[t - 1]):
            fu[t], fl[t], trend[t] = ub[t], lb[t], 1
            continue
        fu[t] = ub[t] if (ub[t] < fu[t - 1] or c[t - 1] > fu[t - 1]) else fu[t - 1]
        fl[t] = lb[t] if (lb[t] > fl[t - 1] or c[t - 1] < fl[t - 1]) else fl[t - 1]
        if trend[t - 1] == 1:
            trend[t] = -1 if c[t] < fl[t] else 1
        else:
            trend[t] = 1 if c[t] > fu[t] else -1
    return trend


def sig_supertrend(df, atr, cfg):
    """Supertrend (ATR 14 x 3) flip."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    tr = _supertrend(h, l, c, atr)
    prev = np.r_[0, tr[:-1]]
    sig = np.where((tr == 1) & (prev == -1), 1, np.where((tr == -1) & (prev == 1), -1, 0))
    return sig.astype(np.int8)


def sig_donchian(df, atr, cfg):
    """Donchian(55) breakout (the Turtle rule)."""
    h, l, c = (pd.Series(df[x].to_numpy(dtype=float)) for x in ("high", "low", "close"))
    up = h.shift(1).rolling(55).max()
    dn = l.shift(1).rolling(55).min()
    return to_signal(cross_up(c, up), cross_down(c, dn))


def sig_keltner(df, atr, cfg):
    """Keltner (EMA20 +/- 2 ATR): a close breaking out of the channel."""
    c = df["close"].to_numpy(dtype=float)
    e = ema(c, 20).to_numpy()
    return to_signal(cross_up(c, e + 2 * atr), cross_down(c, e - 2 * atr))


def sig_heikin_ashi(df, atr, cfg):
    """Heikin Ashi: after 3+ opposite candles, a flat-bottomed (flat-topped) HA candle."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(c)
    hc = (o + h + l + c) / 4
    ho = np.empty(n)
    ho[0] = (o[0] + c[0]) / 2
    for t in range(1, n):
        ho[t] = (ho[t - 1] + hc[t - 1]) / 2
    hh, hl = np.maximum(h, np.maximum(ho, hc)), np.minimum(l, np.minimum(ho, hc))
    bull = hc > ho
    bear = hc < ho
    bear3 = pd.Series(bear).shift(1).rolling(3).sum().to_numpy() == 3
    bull3 = pd.Series(bull).shift(1).rolling(3).sum().to_numpy() == 3
    tol = 0.02 * atr
    long_ = bull & bear3 & (np.minimum(ho, hc) - hl <= tol)
    short = bear & bull3 & (hh - np.maximum(ho, hc) <= tol)
    return to_signal(long_, short)


# ── scalping (indicator-based) ───────────────────────────────────────────────

def sig_ribbon_scalp(df, atr, cfg):
    """1-minute style ribbon scalp: EMA 5/8/13 fanned; buy a tag of EMA8 that closes
    above EMA5."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    e5, e8, e13 = (ema(c, n).to_numpy() for n in (5, 8, 13))
    up = (e5 > e8) & (e8 > e13)
    dn = (e5 < e8) & (e8 < e13)
    return to_signal(up & (l <= e8) & (c > e5), dn & (h >= e8) & (c < e5))


def sig_bb_rsi_scalp(df, atr, cfg):
    """5-minute style scalp: a wick outside Bollinger(20,2) with RSI(7) < 25 (> 75)
    closing in the other direction."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    lo, _, hi = (x.to_numpy() for x in bollinger(c, 20, 2.0))
    r = rsi(c, 7).to_numpy()
    return to_signal((l < lo) & (r < 25) & (c > o), (h > hi) & (r > 75) & (c < o))


# ── swing (indicator-based) ──────────────────────────────────────────────────

def sig_trend_following(df, atr, cfg):
    """Trend following: EMA20 > EMA50 > EMA100 with ADX > 20; buy the close back
    above EMA20 after a close below it."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    e20, e50, e100 = (ema(c, n).to_numpy() for n in (20, 50, 100))
    a = adx(h, l, c)[0].to_numpy() > 20
    up = (e20 > e50) & (e50 > e100) & a
    dn = (e20 < e50) & (e50 < e100) & a
    return to_signal(up & cross_up(c, e20), dn & cross_down(c, e20))


def sig_pullback(df, atr, cfg):
    """Pull-back: above a rising EMA50, after three lower highs, a close above the
    previous bar's high (mirrored)."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    e = ema(c, 50).to_numpy()
    rising = e > np.r_[np.full(5, np.nan), e[:-5]]
    falling = e < np.r_[np.full(5, np.nan), e[:-5]]
    hs, ls = pd.Series(h), pd.Series(l)
    lower_highs = ((hs.shift(1) < hs.shift(2)) & (hs.shift(2) < hs.shift(3))
                   & (hs.shift(3) < hs.shift(4))).to_numpy()
    higher_lows = ((ls.shift(1) > ls.shift(2)) & (ls.shift(2) > ls.shift(3))
                   & (ls.shift(3) > ls.shift(4))).to_numpy()
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    return to_signal(rising & (c > e) & lower_highs & (c > ph),
                     falling & (c < e) & higher_lows & (c < pl))


def sig_ma_swing(df, atr, cfg):
    """Moving-average swing: a close crossing SMA50 while SMA50 slopes the same way."""
    c = df["close"].to_numpy(dtype=float)
    s = sma(c, 50).to_numpy()
    slope = s - np.r_[np.full(10, np.nan), s[:-10]]
    return to_signal(cross_up(c, s) & (slope > 0), cross_down(c, s) & (slope < 0))


def sig_divergence(df, atr, cfg):
    """RSI divergence: a lower swing low with a higher RSI (bullish), a higher swing high
    with a lower RSI (bearish), traded when the second swing is confirmed."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    r = rsi(c, 14).to_numpy()
    sw = swings(h, l, 3)
    n = len(c)

    def at(idx):
        out = np.full(n, np.nan)
        ok = np.isfinite(idx)
        out[ok] = r[idx[ok].astype(int)]
        return out

    long_ = sw.new_low & (sw.sl1 < sw.sl2) & (at(sw.sl1_i) > at(sw.sl2_i) + 3)
    short = sw.new_high & (sw.sh1 > sw.sh2) & (at(sw.sh1_i) < at(sw.sh2_i) - 3)
    return to_signal(long_, short)


# ── quantitative ─────────────────────────────────────────────────────────────

def sig_mean_reversion(df, atr, cfg):
    """Mean reversion (IBS): a bar closing in its bottom 15% above the SMA200 (top 15%
    below it)."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    ibs = (c - l) / (h - l + EPS)
    s = sma(c, 200).to_numpy()
    return to_signal((ibs < 0.15) & (c > s), (ibs > 0.85) & (c < s))


def sig_momentum_factor(df, atr, cfg):
    """Time-series momentum: the one-day return crosses +/- 2 ATR with the last
    quarter-day agreeing."""
    c = df["close"].to_numpy(dtype=float)
    cs = pd.Series(c)
    day = (c - cs.shift(96).to_numpy()) / (atr + EPS)
    q = c - cs.shift(24).to_numpy()
    return to_signal(cross_up(day, 2.0) & (q > 0), cross_down(day, -2.0) & (q < 0))


def sig_trend_algo(df, atr, cfg):
    """Trend-following algorithm: EMA 8/21/55 become aligned on this bar."""
    c = df["close"].to_numpy(dtype=float)
    e8, e21, e55 = (ema(c, n).to_numpy() for n in (8, 21, 55))
    up = (e8 > e21) & (e21 > e55)
    dn = (e8 < e21) & (e21 < e55)
    return to_signal(up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]])


def sig_regime_switch(df, atr, cfg):
    """Regime detection: Donchian(20) breakouts when ADX > 25, Bollinger reversion
    when ADX < 20, nothing in between."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    a = adx(h, l, c)[0].to_numpy()
    hs, ls = pd.Series(h), pd.Series(l)
    brk = to_signal(c > hs.shift(1).rolling(20).max().to_numpy(),
                    c < ls.shift(1).rolling(20).min().to_numpy())
    rev = sig_bollinger(df, atr, cfg)
    return np.where(a > 25, brk, np.where(a < 20, rev, 0)).astype(np.int8)


GENERATORS = {
    "rsi": sig_rsi, "ma_cross": sig_ma_cross, "ema_9_21": sig_ema_9_21,
    "ema_50_200": sig_ema_50_200, "sma_stack": sig_sma_stack, "bollinger": sig_bollinger,
    "adx_trend": sig_adx_trend, "ichimoku": sig_ichimoku, "psar": sig_psar,
    "atr_breakout": sig_atr_breakout, "vwap_trend": sig_vwap_trend, "vwap_bands": sig_vwap_bands,
    "fib_retracement": sig_fib_retracement, "fib_extension": sig_fib_extension,
    "pivot_points": sig_pivot_points, "supertrend": sig_supertrend, "donchian": sig_donchian,
    "keltner": sig_keltner, "heikin_ashi": sig_heikin_ashi,
    "ribbon_scalp": sig_ribbon_scalp, "bb_rsi_scalp": sig_bb_rsi_scalp,
    "trend_following": sig_trend_following, "pullback": sig_pullback, "ma_swing": sig_ma_swing,
    "divergence": sig_divergence,
    "mean_reversion": sig_mean_reversion, "momentum_factor": sig_momentum_factor,
    "trend_algo": sig_trend_algo, "regime_switch": sig_regime_switch,
}

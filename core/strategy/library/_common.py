"""
core/strategy/library/_common.py — shared, CAUSAL building blocks for the
strategy library.

Every helper here returns, for bar ``t``, a value computed from bars ``0..t``
only. The one place that looks "to the right" is the swing-pivot detector, and
it pays for that by publishing a pivot only ``k`` bars after it happened (the
moment a trader could actually see it). Tests replay every strategy on a
truncated frame and assert identical signals, so a look-ahead here would be
caught.

Pure numpy / pandas. No broker, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# ── conversions ──────────────────────────────────────────────────────────────

def to_signal(long_cond, short_cond) -> np.ndarray:
    """Boolean long/short conditions -> int8 array (+1 / -1 / 0). Long wins ties."""
    lc = np.asarray(pd.Series(long_cond).fillna(False), dtype=bool) \
        if not isinstance(long_cond, np.ndarray) else long_cond.astype(bool)
    sc = np.asarray(pd.Series(short_cond).fillna(False), dtype=bool) \
        if not isinstance(short_cond, np.ndarray) else short_cond.astype(bool)
    sig = np.zeros(len(lc), dtype=np.int8)
    sig[sc] = -1
    sig[lc] = 1
    return sig


def ohlc(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))  # type: ignore


def volume(df: pd.DataFrame) -> np.ndarray:
    for col in ("tick_volume", "volume", "real_volume"):
        if col in df.columns:
            return np.clip(df[col].to_numpy(dtype=float), 0, None)
    return np.ones(len(df))


def first_per_group(sig: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Keep only the first non-zero signal of each group (e.g. once per day)."""
    out = np.zeros_like(sig)
    seen = set()
    for t in np.flatnonzero(sig):
        g = group[t]
        if g in seen:
            continue
        seen.add(g)
        out[t] = sig[t]
    return out


def cross_up(a, b) -> np.ndarray:
    a = pd.Series(a); b = pd.Series(b) if not np.isscalar(b) else b
    prev_a = a.shift(1)
    prev_b = b.shift(1) if isinstance(b, pd.Series) else b
    return np.asarray(((a > b) & (prev_a <= prev_b)).fillna(False), dtype=bool)


def cross_down(a, b) -> np.ndarray:
    a = pd.Series(a); b = pd.Series(b) if not np.isscalar(b) else b
    prev_a = a.shift(1)
    prev_b = b.shift(1) if isinstance(b, pd.Series) else b
    return np.asarray(((a < b) & (prev_a >= prev_b)).fillna(False), dtype=bool)


# ── indicators ───────────────────────────────────────────────────────────────

def ema(x, n: int) -> pd.Series:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean()


def sma(x, n: int) -> pd.Series:
    return pd.Series(x).rolling(n).mean()


def rsi(close, n: int = 14) -> pd.Series:
    c = pd.Series(close)
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + EPS))


def adx(high, low, close, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX with +DI / -DI."""
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a = 1.0 / n
    atr_ = tr.ewm(alpha=a, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=n).mean() / (atr_ + EPS)
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=n).mean() / (atr_ + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return dx.ewm(alpha=a, adjust=False, min_periods=n).mean(), pdi, mdi


def bollinger(close, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.Series(close)
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def day_id(index: pd.DatetimeIndex) -> np.ndarray:
    days = index.normalize().asi8
    return np.cumsum(np.r_[0, (np.diff(days) != 0).astype(int)])


def vwap(df: pd.DataFrame, group: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    """Session VWAP and the volume-weighted standard deviation around it."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = pd.Series(volume(df) + 1.0, index=df.index)
    g = pd.Series(group, index=df.index)
    cum_v = vol.groupby(g).cumsum()
    vw = (tp * vol).groupby(g).cumsum() / cum_v
    var = (tp * tp * vol).groupby(g).cumsum() / cum_v - vw * vw
    return vw, np.sqrt(var.clip(lower=0))


# ── NY clock (ICT sessions) ──────────────────────────────────────────────────

def _us_dst_bounds(year: int) -> Tuple[np.datetime64, np.datetime64]:
    """UTC instants when US DST starts (2nd Sunday of March 07:00 UTC) and ends
    (1st Sunday of November 06:00 UTC)."""
    march1 = pd.Timestamp(year=year, month=3, day=1)
    second_sunday = march1 + pd.Timedelta(days=(6 - march1.weekday()) % 7 + 7)
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_sunday = nov1 + pd.Timedelta(days=(6 - nov1.weekday()) % 7)
    return (np.datetime64(second_sunday + pd.Timedelta(hours=7)),
            np.datetime64(first_sunday + pd.Timedelta(hours=6)))


@dataclass
class Clock:
    ny_hour: np.ndarray      # float hours in New York time (e.g. 9.5 = 09:30)
    ny_weekday: np.ndarray   # 0 = Monday in New York time
    fx_day: np.ndarray       # trading-day id; a day starts at 18:00 New York
    fx_weekday: np.ndarray   # weekday of that trading day (Sunday evening -> Monday)
    week: np.ndarray         # trading-week id
    utc_hour: np.ndarray
    bar_minutes: float


def ny_clock(index: pd.DatetimeIndex, server_utc_offset: float = 0.0) -> Clock:
    """Convert the data's own clock to New York time, DST included, without tz data."""
    naive = index.tz_convert(None) if index.tz is not None else index
    utc = naive - pd.Timedelta(hours=float(server_utc_offset))
    u = utc.to_numpy(dtype="datetime64[ns]")
    years = pd.DatetimeIndex(u).year.to_numpy()
    offset = np.full(len(u), -5.0)
    for y in np.unique(years):
        start, end = _us_dst_bounds(int(y))
        m = (years == y) & (u >= start) & (u < end)
        offset[m] = -4.0
    ny = pd.DatetimeIndex(u + (offset * 3600e9).astype("timedelta64[ns]"))
    ny_hour = ny.hour.to_numpy() + ny.minute.to_numpy() / 60.0
    shifted = ny + pd.Timedelta(hours=6)                    # 18:00 NY -> next calendar day
    fx_days = shifted.normalize().asi8
    fx_day = np.cumsum(np.r_[0, (np.diff(fx_days) != 0).astype(int)]) if len(u) else np.zeros(0, int)
    fx_weekday = shifted.weekday.to_numpy()
    iso = shifted.isocalendar()
    wk = (iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy().astype(np.int64))
    week = np.cumsum(np.r_[0, (np.diff(wk) != 0).astype(int)]) if len(u) else np.zeros(0, int)
    if len(index) > 1:
        bar_minutes = float(np.median(np.diff(index.asi8)) / 60e9)
    else:
        bar_minutes = 15.0
    return Clock(ny_hour, ny.weekday.to_numpy(), fx_day, fx_weekday, week,
                 pd.DatetimeIndex(u).hour.to_numpy() + pd.DatetimeIndex(u).minute.to_numpy() / 60.0,
                 bar_minutes)


def in_window(hour: np.ndarray, start: float, end: float, bar_minutes: float = 0.0) -> np.ndarray:
    """Bars whose interval [open, open + bar) overlaps [start, end) on a 24h clock."""
    b = bar_minutes / 60.0
    after_start = (hour >= start) | (hour + b > start)
    if start <= end:
        return after_start & (hour < end)
    return after_start | (hour < end)


def session_range(high: np.ndarray, low: np.ndarray, group: np.ndarray, mask: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """For every bar, the high/low of the ``mask`` bars of its group seen SO FAR
    (NaN before the first masked bar). Causal: a bar sees only earlier/equal bars."""
    n = len(high)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    cur_g, h, l = None, np.nan, np.nan
    for t in range(n):
        if group[t] != cur_g:
            cur_g, h, l = group[t], np.nan, np.nan
        if mask[t]:
            h = high[t] if np.isnan(h) else max(h, high[t])
            l = low[t] if np.isnan(l) else min(l, low[t])
        hi[t], lo[t] = h, l
    return hi, lo


def closed_session_range(high, low, group, mask) -> Tuple[np.ndarray, np.ndarray]:
    """Session range, but only published once the session is over (no bar of the
    session left), so a breakout cannot be measured against a range still forming."""
    hi, lo = session_range(high, low, group, mask)
    done = np.zeros(len(high), dtype=bool)
    cur_g, started = None, False
    for t in range(len(high)):
        if group[t] != cur_g:
            cur_g, started = group[t], False
        if mask[t]:
            started = True
        done[t] = started and not mask[t]
    return np.where(done, hi, np.nan), np.where(done, lo, np.nan)


def previous_group_hlc(high, low, close, group) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """High, low, close and open of the PREVIOUS group (day / week), per bar."""
    g = pd.Series(group)
    agg = pd.DataFrame({"h": high, "l": low, "c": close, "g": group}).groupby("g").agg(
        h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = agg.shift(1)
    return (prev["h"].reindex(g).to_numpy(), prev["l"].reindex(g).to_numpy(),
            prev["c"].reindex(g).to_numpy(), None)


def group_open(open_: np.ndarray, group: np.ndarray) -> np.ndarray:
    s = pd.Series(open_)
    return s.groupby(pd.Series(group)).transform("first").to_numpy()


def bars_in_group(group: np.ndarray) -> np.ndarray:
    return pd.Series(group).groupby(pd.Series(group)).cumcount().to_numpy()


# ── swing pivots, published only once confirmed ──────────────────────────────

@dataclass
class Pivots:
    """Confirmed pivots, in order of confirmation."""
    conf: np.ndarray     # bar at which the pivot became visible (pivot bar + k)
    idx: np.ndarray      # pivot bar
    price: np.ndarray
    kind: np.ndarray     # +1 high, -1 low


def pivots(high: np.ndarray, low: np.ndarray, k: int = 3) -> Pivots:
    h = pd.Series(high)
    l = pd.Series(low)
    left_h = h.shift(1).rolling(k).max()
    right_h = h[::-1].shift(1).rolling(k).max()[::-1]
    left_l = l.shift(1).rolling(k).min()
    right_l = l[::-1].shift(1).rolling(k).min()[::-1]
    ph = np.flatnonzero(((h > left_h) & (h >= right_h)).to_numpy())
    pl = np.flatnonzero(((l < left_l) & (l <= right_l)).to_numpy())
    idx = np.r_[ph, pl]
    kind = np.r_[np.ones(len(ph), dtype=int), -np.ones(len(pl), dtype=int)]
    price = np.r_[high[ph], low[pl]]
    order = np.lexsort((kind, idx))
    idx, kind, price = idx[order], kind[order], price[order]
    return Pivots(idx + k, idx, price, kind)


def last_known(n: int, conf: np.ndarray, values: np.ndarray, back: int = 0) -> np.ndarray:
    """Per bar t: the ``back``-th most recent value among events with conf <= t."""
    out = np.full(n, np.nan)
    if len(conf) == 0:
        return out
    pos = np.searchsorted(conf, np.arange(n), side="right") - 1 - back
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


@dataclass
class Swings:
    sh1: np.ndarray; sh2: np.ndarray; sh1_i: np.ndarray; sh2_i: np.ndarray
    sl1: np.ndarray; sl2: np.ndarray; sl1_i: np.ndarray; sl2_i: np.ndarray
    new_high: np.ndarray   # a swing high is confirmed on this bar
    new_low: np.ndarray


def swings(high: np.ndarray, low: np.ndarray, k: int = 3) -> Swings:
    n = len(high)
    p = pivots(high, low, k)
    hs = p.kind == 1
    ls = p.kind == -1
    ch, cl = p.conf[hs], p.conf[ls]
    new_high = np.zeros(n, dtype=bool); new_high[ch[ch < n]] = True
    new_low = np.zeros(n, dtype=bool); new_low[cl[cl < n]] = True
    return Swings(
        last_known(n, ch, p.price[hs]), last_known(n, ch, p.price[hs], 1),
        last_known(n, ch, p.idx[hs].astype(float)), last_known(n, ch, p.idx[hs].astype(float), 1),
        last_known(n, cl, p.price[ls]), last_known(n, cl, p.price[ls], 1),
        last_known(n, cl, p.idx[ls].astype(float)), last_known(n, cl, p.idx[ls].astype(float), 1),
        new_high, new_low)


def zigzag(high: np.ndarray, low: np.ndarray, k: int = 3, depth: int = 6
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating high/low pivots as known at each bar.

    Returns (price, kind, idx), each shaped (n, depth); column -1 is the most
    recent pivot. Two pivots of the same kind in a row are merged into the more
    extreme one. NaN / 0 where there is no pivot yet.
    """
    n = len(high)
    p = pivots(high, low, k)
    price = np.full((n, depth), np.nan)
    kind = np.zeros((n, depth), dtype=int)
    idx = np.full((n, depth), -1, dtype=int)
    seq: List[Tuple[float, int, int]] = []
    snaps_conf, snaps = [], []
    for c, i, pr, kd in zip(p.conf, p.idx, p.price, p.kind):
        if c >= n:
            break
        if seq and seq[-1][1] == kd:
            if (kd == 1 and pr > seq[-1][0]) or (kd == -1 and pr < seq[-1][0]):
                seq[-1] = (pr, kd, i)
        else:
            seq.append((pr, kd, i))
        tail = seq[-depth:]
        snaps_conf.append(c)
        snaps.append(tail)
    if not snaps:
        return price, kind, idx
    snaps_conf = np.asarray(snaps_conf)
    pos = np.searchsorted(snaps_conf, np.arange(n), side="right") - 1
    for t in np.flatnonzero(pos >= 0):
        tail = snaps[pos[t]]
        m = len(tail)
        for j, (pr, kd, i) in enumerate(tail):
            price[t, depth - m + j] = pr
            kind[t, depth - m + j] = kd
            idx[t, depth - m + j] = i
    return price, kind, idx


# ── structure: BOS / CHOCH ───────────────────────────────────────────────────

@dataclass
class Structure:
    event: np.ndarray     # +1 BOS up, +2 CHOCH up, -1 BOS down, -2 CHOCH down, 0 none
    level: np.ndarray     # the swing level broken by the event
    state: np.ndarray     # +1 bullish, -1 bearish, 0 unknown (after this bar)


def structure(high: np.ndarray, low: np.ndarray, close: np.ndarray, k: int = 3) -> Structure:
    n = len(close)
    sw = swings(high, low, k)
    event = np.zeros(n, dtype=np.int8)
    level = np.full(n, np.nan)
    state = np.zeros(n, dtype=np.int8)
    st = 0
    used_h, used_l = np.nan, np.nan
    for t in range(n):
        h1, l1 = sw.sh1[t], sw.sl1[t]
        if not np.isnan(h1) and h1 != used_h and close[t] > h1:
            event[t] = 1 if st == 1 else 2
            level[t] = h1
            used_h = h1
            st = 1
        elif not np.isnan(l1) and l1 != used_l and close[t] < l1:
            event[t] = -1 if st == -1 else -2
            level[t] = l1
            used_l = l1
            st = -1
        state[t] = st
    return Structure(event, level, state)


# ── zones that are traded on their first retest ──────────────────────────────

@dataclass(frozen=True)
class Zone:
    created: int     # bar on which the zone became known (signals only after it)
    lo: float
    hi: float
    direction: int   # +1 demand (buy the retest), -1 supply (sell the retest)


def retest_signals(zones: Iterable[Zone], high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   max_age: int = 120, keep: int = 4) -> np.ndarray:
    """Signal on the first bar that trades into a zone and closes back on its side.

    Demand zone: ``low <= hi`` and ``close >= lo`` -> +1, and the zone is used up.
    A close through the far side (``close < lo``) invalidates it. Supply mirrored.
    At most ``keep`` live zones per direction (the most recent).
    """
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    by_bar: dict = {}
    for z in zones:
        if 0 <= z.created < n - 1 and np.isfinite(z.lo) and np.isfinite(z.hi) and z.hi >= z.lo:
            by_bar.setdefault(z.created, []).append(z)
    if not by_bar:
        return sig
    live: List[Zone] = []
    for t in range(min(by_bar), n):
        if live:
            nxt = []
            fired = 0
            for z in live:
                if t - z.created > max_age:
                    continue
                if z.direction == 1:
                    if close[t] < z.lo:
                        continue
                    if low[t] <= z.hi:
                        fired = fired or 1
                        continue
                else:
                    if close[t] > z.hi:
                        continue
                    if high[t] >= z.lo:
                        fired = fired or -1
                        continue
                nxt.append(z)
            live = nxt
            if fired:
                sig[t] = fired
        if t in by_bar:
            live.extend(by_bar[t])
            longs = [z for z in live if z.direction == 1][-keep:]
            shorts = [z for z in live if z.direction == -1][-keep:]
            live = longs + shorts
    return sig


def fvg_zones(high, low, close, open_, atr, min_gap_atr: float = 0.15) -> List[Zone]:
    """Three-bar fair value gaps; known at the close of the third bar."""
    zones = []
    n = len(close)
    for t in range(2, n):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        if low[t] - high[t - 2] >= min_gap_atr * a and close[t - 1] > open_[t - 1]:
            zones.append(Zone(t, high[t - 2], low[t], 1))
        elif low[t - 2] - high[t] >= min_gap_atr * a and close[t - 1] < open_[t - 1]:
            zones.append(Zone(t, high[t], low[t - 2], -1))
    return zones


def invert(zones: Sequence[Zone], high, low, close, max_age: int = 200) -> List[Zone]:
    """Zones that were closed through become zones of the opposite side, known
    on the bar that closed through them (breakers, inverse FVGs)."""
    out = []
    n = len(close)
    for z in zones:
        end = min(n, z.created + 1 + max_age)
        for t in range(z.created + 1, end):
            if z.direction == 1 and close[t] < z.lo:
                out.append(Zone(t, z.lo, z.hi, -1))
                break
            if z.direction == -1 and close[t] > z.hi:
                out.append(Zone(t, z.lo, z.hi, 1))
                break
    return out


def order_block_zones(open_, high, low, close, struct: Structure, lookback: int = 10,
                      events: Tuple[int, ...] = (1, 2, -1, -2)) -> List[Zone]:
    """Last opposite-coloured candle before a structure break."""
    zones = []
    for t in np.flatnonzero(struct.event):
        e = int(struct.event[t])
        if e not in events:
            continue
        d = 1 if e > 0 else -1
        for j in range(t - 1, max(-1, t - 1 - lookback), -1):
            if (d == 1 and close[j] < open_[j]) or (d == -1 and close[j] > open_[j]):
                zones.append(Zone(t, low[j], high[j], d))
                break
    return zones


def volume_profile(price_lo: np.ndarray, price_hi: np.ndarray, weight: np.ndarray, bins: int = 30,
                   va: float = 0.70) -> Tuple[float, float, float]:
    """(POC, VAL, VAH) of a set of bars, spreading each bar's weight across its range."""
    lo, hi = float(np.nanmin(price_lo)), float(np.nanmax(price_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= EPS:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    i0 = np.clip(np.searchsorted(edges, price_lo, side="right") - 1, 0, bins - 1)
    i1 = np.clip(np.searchsorted(edges, price_hi, side="right") - 1, 0, bins - 1)
    share = np.asarray(weight, dtype=float) / (i1 - i0 + 1)
    diff = np.zeros(bins + 1)
    np.add.at(diff, i0, share)
    np.add.at(diff, i1 + 1, -share)
    hist = np.cumsum(diff[:-1])
    poc_i = int(np.argmax(hist))
    total = hist.sum()
    lo_i = hi_i = poc_i
    acc = hist[poc_i]
    while acc < va * total and (lo_i > 0 or hi_i < bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1
        up = hist[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1; acc += hist[hi_i]
        else:
            lo_i -= 1; acc += hist[lo_i]
    return centers[poc_i], edges[lo_i], edges[hi_i + 1]


def previous_day_profile(high, low, vol, group, bins: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POC/VAL/VAH of the previous group, broadcast to every bar of the current one."""
    n = len(high)
    poc = np.full(n, np.nan); val = np.full(n, np.nan); vah = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, np.diff(group) != 0])
    ends = np.r_[starts[1:], n]
    prev = None
    for s, e in zip(starts, ends):
        if prev is not None:
            poc[s:e], val[s:e], vah[s:e] = prev
        prev = volume_profile(low[s:e], high[s:e], vol[s:e], bins)
    return poc, val, vah

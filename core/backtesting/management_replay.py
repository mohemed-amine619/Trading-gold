"""
core/backtesting/management_replay.py — re-run REAL entries under different
break-even / trailing settings, on historical bars.

The question it answers: "my average win is +$1.41 and my average loss is -$4.44;
what if the stop-management settings were different?" Instead of waiting weeks on
a demo account, replay the actual entries (time, price, direction, initial stop and
target) bar by bar and apply the bot's own management rules with other numbers.

What is modelled (mirrors run.py's _manage_positions):
  * the initial stop and target;
  * the progressive break-even / profit lock: once profit passes
    ``breakeven_trigger_atr`` the stop moves to entry + ``breakeven_buffer_atr`` and
    then locks ``profit_lock_ratio`` of the extra profit, but never closer to price
    than ``lock_min_distance_atr`` (0 = the legacy 0.55 ATR for gold, so the real
    break-even trigger is that distance plus the buffer, about 0.75 ATR);
  * early protection and the trailing stop with its tightening tiers;
  * the profit-lock giveback close;
  * the rule that a stop never loosens, and the minimum step between modifications.

What is NOT modelled, and why the result is an estimate and not a forecast:
  * the structure-trailing stop and the AI stop reasoner, partial closes, AI early
    exits and trend-flip exits;
  * slippage beyond a fixed spread, and the order of price inside one bar (the
    replay is pessimistic: when a bar touches both the stop and the target the stop
    is taken, and a stop change takes effect from the NEXT bar).
Always replay the CURRENT settings first and compare with what actually happened:
how far apart they are is how much to trust the other numbers.

Pure functions over pandas frames. No broker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.position.sl_guard import is_tighter, step_ok

GOLD_LEGACY_DISTANCE_ATR = 0.55
OTHER_LEGACY_DISTANCE_ATR = 0.40


@dataclass(frozen=True)
class ManagementParams:
    breakeven_trigger_atr: float = 0.40
    breakeven_buffer_atr: float = 0.15
    early_protect_atr: float = 0.30
    trail_activate_atr: float = 1.10
    trail_atr_mult: float = 1.00
    trail_tighten1_atr: float = 2.0
    trail_tighten1_mult: float = 1.0
    trail_tighten2_atr: float = 3.0
    trail_tighten2_mult: float = 0.6
    profit_lock_ratio: float = 0.50
    profit_lock_activate_atr: float = 1.0
    profit_giveback_pct: float = 0.45
    lock_min_distance_atr: float = 0.0      # 0 = legacy (0.55 gold / 0.40 other)
    min_sl_step_atr: float = 0.05
    enable_breakeven: bool = True
    is_gold: bool = True

    def distance_atr(self) -> float:
        if self.lock_min_distance_atr > 0:
            return self.lock_min_distance_atr
        return GOLD_LEGACY_DISTANCE_ATR if self.is_gold else OTHER_LEGACY_DISTANCE_ATR


@dataclass(frozen=True)
class ReplayTrade:
    ticket: str
    direction: int                  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    volume: float
    atr: float
    actual_pnl: Optional[float] = None


@dataclass(frozen=True)
class SimResult:
    exit_price: float
    exit_time: pd.Timestamp
    reason: str                     # sl / tp / giveback / timeout
    move: float                     # price units, in the trade's direction, after the spread
    peak: float
    stop_moves: int


def simulate_trade(trade: ReplayTrade, bars: pd.DataFrame, params: ManagementParams,
                   spread: float = 0.0, max_bars: int = 720, point: float = 0.01) -> SimResult:
    """Replay one trade bar by bar. ``bars`` must have open/high/low/close and a
    sorted time index; the first bar used is the first one that opens at or after the
    entry time."""
    d, atr, entry = trade.direction, trade.atr, trade.entry_price
    sl, tp = trade.sl, trade.tp
    p = params
    dist = p.distance_atr() * atr
    be_sl = entry + d * p.breakeven_buffer_atr * atr
    min_step = max(point * 1.5, p.min_sl_step_atr * atr)
    path = bars.loc[bars.index >= trade.entry_time].iloc[:max_bars]
    peak = 0.0
    moves = 0
    last_close = entry

    def result(price, when, reason):
        return SimResult(float(price), when, reason, (price - entry) * d - spread, peak, moves)

    for when, bar in path.iterrows():
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

        # 1. stop first (pessimistic), then target; a gap through the stop fills at the open
        if d == 1:
            if l <= sl:
                return result(min(sl, o), when, "sl")
            if h >= tp:
                return result(tp, when, "tp")
            best = h - entry
        else:
            if h >= sl:
                return result(max(sl, o), when, "sl")
            if l <= tp:
                return result(tp, when, "tp")
            best = entry - l
        peak = max(peak, best)
        last_close = c
        profit = (c - entry) * d

        # 2. profit-lock giveback close
        if peak >= p.profit_lock_activate_atr * atr and profit <= peak * (1.0 - p.profit_giveback_pct):
            return result(c, when, "giveback")

        # 3. stop changes, effective from the next bar
        new_sl = sl
        if p.enable_breakeven and profit >= p.breakeven_trigger_atr * atr:
            extra = profit - p.breakeven_trigger_atr * atr
            locked = p.breakeven_buffer_atr * atr + extra * p.profit_lock_ratio
            target = entry + d * locked
            target = max(be_sl, target) if d == 1 else min(be_sl, target)
            target = min(target, c - dist) if d == 1 else max(target, c + dist)
            reaches = target >= be_sl if d == 1 else target <= be_sl
            if reaches and step_ok(d == 1, new_sl, target, min_step, be_sl):
                new_sl = target
        if 0 < profit < p.trail_activate_atr * atr and profit >= p.early_protect_atr * atr:
            protect = entry + d * profit * 0.4
            if step_ok(d == 1, new_sl, protect, min_step):
                new_sl = protect
        if profit >= p.trail_activate_atr * atr:
            trail = p.trail_atr_mult * atr
            if profit >= p.trail_tighten2_atr * atr:
                trail = p.trail_tighten2_mult * atr
            elif profit >= p.trail_tighten1_atr * atr:
                trail = p.trail_tighten1_mult * atr
            cand = c - d * trail
            cand = max(cand, be_sl) if d == 1 else min(cand, be_sl)
            if step_ok(d == 1, new_sl, cand, min_step, be_sl):
                new_sl = cand
        if new_sl != sl and is_tighter(d == 1, sl, new_sl):
            sl = new_sl
            moves += 1

    when = path.index[-1] if len(path) else trade.entry_time
    return result(last_close, when, "timeout")


def usd(move: float, volume: float, contract_size: float = 100.0) -> float:
    """Price-unit move to account currency. 0.01 lot of gold is 1 oz: $1 per $1."""
    return move * volume * contract_size


def run_replay(trades: List[ReplayTrade], bars: pd.DataFrame, params: ManagementParams,
               spread: float = 0.0, contract_size: float = 100.0, **kw) -> List[dict]:
    rows = []
    for t in trades:
        r = simulate_trade(t, bars, params, spread=spread, **kw)
        rows.append({"ticket": t.ticket, "move": r.move, "pnl": usd(r.move, t.volume, contract_size),
                     "reason": r.reason, "peak": r.peak, "stop_moves": r.stop_moves,
                     "actual_pnl": t.actual_pnl})
    return rows


def summarise(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "net": 0.0, "win_rate": None, "avg_win": None, "avg_loss": None,
                "profit_factor": None, "expectancy": None}
    pnls = [r["pnl"] for r in rows]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_loss = -sum(losses)
    return {"trades": n, "net": round(sum(pnls), 2), "win_rate": round(len(wins) / n, 3),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else None,
            "profit_factor": round(sum(wins) / gross_loss, 2) if gross_loss > 0 else None,
            "expectancy": round(sum(pnls) / n, 3)}


def paired_difference(a: List[dict], b: List[dict]) -> dict:
    """Mean per-trade difference ``b - a`` on the SAME entries, its standard error and
    the z-score. Paired, so the noise of which trades were taken cancels out. A |z| below
    about 2 means the difference is not distinguishable from chance on this many trades."""
    by_a = {r["ticket"]: r["pnl"] for r in a}
    diffs = [r["pnl"] - by_a[r["ticket"]] for r in b if r["ticket"] in by_a]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": None, "se": None, "z": None}
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    return {"n": n, "mean": round(mean, 3), "se": round(se, 3),
            "z": round(mean / se, 2) if se > 0 else None}


def calibration(rows: List[dict]) -> dict:
    """How close the replay of the CURRENT settings is to what really happened."""
    pairs = [(r["pnl"], r["actual_pnl"]) for r in rows if r["actual_pnl"] is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "mean_abs_error": None, "correlation": None,
                "replayed_net": None, "actual_net": None}
    sim, act = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    corr = float(np.corrcoef(sim, act)[0, 1]) if sim.std() > 0 and act.std() > 0 else None
    return {"n": len(pairs), "mean_abs_error": round(float(np.mean(np.abs(sim - act))), 2),
            "correlation": round(corr, 2) if corr is not None else None,
            "replayed_net": round(float(sim.sum()), 2), "actual_net": round(float(act.sum()), 2)}


def with_params(base: ManagementParams, **changes) -> ManagementParams:
    valid = {k: v for k, v in changes.items() if hasattr(base, k)}
    unknown = set(changes) - set(valid)
    if unknown:
        raise ValueError(f"unknown management parameter(s): {sorted(unknown)}")
    return replace(base, **valid)


def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple rolling-mean ATR over closed bars (the bar's own range included)."""
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(),
                    (bars["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_replay_trades(entries: List[dict], atr_bars: pd.DataFrame,
                        actual_by_ticket: Optional[Dict[str, float]] = None,
                        shift_hours: float = 0.0) -> tuple:
    """Turn parsed log entries into ReplayTrades. Returns (trades, skipped) where skipped
    counts entries that cannot be replayed: no ticket (the join key), no ATR known at the
    entry time, or a missing/zero stop. Nothing is guessed."""
    atr = atr_series(atr_bars).dropna()
    actual_by_ticket = actual_by_ticket or {}
    trades: List[ReplayTrade] = []
    skipped = {"no_ticket": 0, "no_atr": 0, "no_stop": 0}
    for e in entries:
        if not e.get("ticket"):
            skipped["no_ticket"] += 1
            continue
        when = pd.Timestamp(e["time"]) + pd.Timedelta(hours=shift_hours)
        known = atr.loc[atr.index <= when]
        if known.empty or not known.iloc[-1] > 0:
            skipped["no_atr"] += 1
            continue
        if not e.get("sl") or not e.get("tp"):
            skipped["no_stop"] += 1
            continue
        trades.append(ReplayTrade(
            ticket=str(e["ticket"]), direction=1 if e["side"] == "BUY" else -1, entry_time=when,
            entry_price=float(e["price"]), sl=float(e["sl"]), tp=float(e["tp"]),
            volume=float(e["volume"]), atr=float(known.iloc[-1]),
            actual_pnl=actual_by_ticket.get(str(e["ticket"]))))
    return trades, skipped


PRESETS: Dict[str, dict] = {
    "current": {},
    "wider": {"lock_min_distance_atr": 1.0},
    "no-breakeven": {"enable_breakeven": False},
}

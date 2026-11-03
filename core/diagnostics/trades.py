"""
core/diagnostics/trades.py — turn MT5 deal history into round-trip trades and
summary statistics.

Why the broker history and not the bot's own ``trades`` table: the bot only
records trades it closes itself. A stop-loss or take-profit hit at the broker
never goes through the bot's close path, so the table misses exactly the trades
that matter most. The deal history has every exit, with the reason the broker
gives for it.

Deals are plain dicts (the collector converts MT5's namedtuples), so this
module needs neither MetaTrader5 nor pandas.

Deal fields used: ``position_id, type, entry, time, volume, price, profit,
commission, swap, fee, symbol, magic, reason, comment``.
MT5 constants: type 0 buy / 1 sell (others are balance, credit, ...);
entry 0 in / 1 out / 2 in-out / 3 out-by; reason 0 client / 1 mobile / 2 web /
3 expert / 4 stop loss / 5 take profit / 6 stop out.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

REASONS = {0: "client", 1: "mobile", 2: "web", 3: "expert", 4: "sl", 5: "tp", 6: "stop_out"}
_ENTRY_IN = 0
_ENTRY_OUT = (1, 2, 3)           # in-out (reversal) is treated as the closing side
QUICK_STOP_SECONDS = 300


@dataclass
class Trade:
    position_id: int
    symbol: str
    direction: int               # +1 long, -1 short
    volume: float
    open_time: str
    close_time: str
    open_price: float
    close_price: float
    net_pnl: float               # profit + commission + swap + fee, every deal
    exit_reason: str
    duration_s: int
    magic: int
    comment: str

    def to_row(self) -> dict:
        return asdict(self)


def _iso(epoch) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).replace(tzinfo=None).isoformat(sep=" ")


def _money(d: dict) -> float:
    return sum(float(d.get(k) or 0.0) for k in ("profit", "commission", "swap", "fee"))


def _weighted(deals: List[dict], field: str) -> float:
    total = sum(float(d["volume"]) for d in deals)
    return sum(float(d[field]) * float(d["volume"]) for d in deals) / total if total else 0.0


def pair_deals(deals: Iterable[dict], magic: Optional[int] = None) -> List[Trade]:
    """Group deals by position into closed round trips, oldest close first.

    A position still open, or only partly closed, is not a finished trade and is
    left out. Balance/credit deals are ignored. With ``magic`` set, only
    positions opened with that magic number are kept.
    """
    by_position: Dict[int, List[dict]] = defaultdict(list)
    for d in deals:
        if int(d.get("type", -1)) in (0, 1) and d.get("position_id"):
            by_position[int(d["position_id"])].append(d)

    trades: List[Trade] = []
    for position_id, group in by_position.items():
        ins = [d for d in group if int(d["entry"]) == _ENTRY_IN]
        outs = [d for d in group if int(d["entry"]) in _ENTRY_OUT]
        if not ins or not outs:
            continue
        vol_in = sum(float(d["volume"]) for d in ins)
        vol_out = sum(float(d["volume"]) for d in outs)
        if vol_out < vol_in - 1e-9:
            continue                                  # still (partly) open
        first_in = min(ins, key=lambda d: d["time"])
        last_out = max(outs, key=lambda d: d["time"])
        if magic is not None and int(first_in.get("magic", 0)) != int(magic):
            continue
        open_t, close_t = int(first_in["time"]), int(last_out["time"])
        trades.append(Trade(
            position_id=position_id,
            symbol=str(first_in.get("symbol", "")),
            direction=1 if int(first_in["type"]) == 0 else -1,
            volume=round(vol_in, 8),
            open_time=_iso(open_t), close_time=_iso(close_t),
            open_price=_weighted(ins, "price"), close_price=_weighted(outs, "price"),
            net_pnl=round(sum(_money(d) for d in group), 8),
            exit_reason=REASONS.get(int(last_out.get("reason", 0)), "other"),
            duration_s=max(0, close_t - open_t),
            magic=int(first_in.get("magic", 0)),
            comment=str(first_in.get("comment", "")),
        ))
    return sorted(trades, key=lambda t: t.close_time)


def _bucket(trades: List[Trade], key) -> dict:
    groups: Dict[object, List[Trade]] = defaultdict(list)
    for t in trades:
        groups[key(t)].append(t)
    return {k: {"trades": len(v), "net": round(sum(t.net_pnl for t in v), 2),
                "win_rate": round(sum(t.net_pnl > 0 for t in v) / len(v), 3)}
            for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


HAND_CLOSED = ("client", "mobile", "web")      # closed by a person, not by a stop or the bot


def trade_stats(trades: List[Trade], bot_magic: Optional[int] = None) -> dict:
    """Summary of closed trades. Ratios are ``None`` when undefined, never 0.

    With ``bot_magic`` the trades are also split by ORIGIN: opened by the bot, or
    "other" (a different magic number: manual trades or another program). Money
    lost by hand is easy to mistake for the bot's own result, so the split is
    reported whenever the bot's magic number is known.
    """
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net": 0.0, "wins": 0, "losses": 0, "win_rate": None,
                "avg_win": None, "avg_loss": None, "profit_factor": None,
                "expectancy": None, "max_consecutive_losses": 0, "quick_losses": 0,
                "median_duration_s": None, "by_exit_reason": {}, "by_hour": {},
                "by_direction": {}, "by_origin": {}, "hand_closed": {"trades": 0, "net": 0.0}}
    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    streak = best = 0
    for p in pnls:
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
    gross_win, gross_loss = sum(wins), -sum(losses)
    return {
        "trades": n,
        "net": round(sum(pnls), 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n, 3),
        "avg_win": round(gross_win / len(wins), 2) if wins else None,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy": round(sum(pnls) / n, 3),
        "max_consecutive_losses": best,
        # A loss closed within minutes of entry is usually a stop that was too
        # tight for the noise, not a wrong view of the market.
        "quick_losses": sum(1 for t in trades if t.net_pnl < 0 and t.duration_s < QUICK_STOP_SECONDS),
        "median_duration_s": int(statistics.median(t.duration_s for t in trades)),
        "by_exit_reason": _bucket(trades, lambda t: t.exit_reason),
        "by_hour": _bucket(trades, lambda t: int(t.open_time[11:13])),
        "by_direction": _bucket(trades, lambda t: "long" if t.direction == 1 else "short"),
        "by_origin": (_bucket(trades, lambda t: "bot" if t.magic == bot_magic else "other")
                      if bot_magic is not None else {}),
        "hand_closed": {"trades": sum(1 for t in trades if t.exit_reason in HAND_CLOSED),
                        "net": round(sum(t.net_pnl for t in trades if t.exit_reason in HAND_CLOSED), 2)},
    }

"""
core/diagnostics/excursion.py — how far did trades run before they ended?

Each TRADE END line in the bot log carries the best and worst the trade reached
(in price units and ATR). Joined by ticket to the broker's closed trades, it
answers the questions behind break-even and trailing-stop settings:

* Of the trades that LOST, how many had first been in profit by 0.3 / 0.5 / 1.0
  ATR? Those are the trades an earlier lock-in could have changed.
* Of the trades that WON, how much of their peak did they keep?
* What happens to win rate and result as the peak grows?

None of this proves a setting is better: moving a trigger earlier also stops
some winners out. It shows where the money leaks so that the next experiment is
aimed at something.

The excursion is sampled about once a second by the bot, so it is a floor on the
true peak and worst, not an exact figure.

Pure functions over the parsed log entries and the paired trades.
"""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional

from core.diagnostics.trades import Trade

PEAK_BUCKETS = ((0.0, 0.3, "<0.3 ATR"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1.0"),
                (1.0, 99.0, ">=1.0"))
LOSER_THRESHOLDS = (0.3, 0.5, 1.0)
MIN_LOSERS_FOR_FINDING = 8
MIN_WINNERS_FOR_FINDING = 8


def _bucket(peak_atr: float) -> str:
    for low, high, name in PEAK_BUCKETS:
        if low <= peak_atr < high:
            return name
    return PEAK_BUCKETS[-1][2]


def _capture(trade: Trade, peak: float) -> Optional[float]:
    """Share of the peak move the trade actually kept (price units), or ``None``."""
    if peak <= 0:
        return None
    move = (trade.close_price - trade.open_price) * trade.direction
    return move / peak


def excursion_report(trade_ends: List[dict], trades: List[Trade]) -> dict:
    by_position: Dict[str, Trade] = {str(t.position_id): t for t in trades}
    rows = []
    unmatched = 0
    for end in trade_ends:
        trade = by_position.get(str(end.get("ticket")))
        if trade is None:
            unmatched += 1
            continue
        rows.append((trade, end))

    buckets = {name: [] for _, _, name in PEAK_BUCKETS}
    for trade, end in rows:
        buckets[_bucket(end["peak_atr"])].append(trade)

    table = {}
    for _, _, name in PEAK_BUCKETS:
        group = buckets[name]
        n = len(group)
        wins = sum(1 for t in group if t.net_pnl > 0)
        net = sum(t.net_pnl for t in group)
        table[name] = {"trades": n, "wins": wins,
                       "win_rate": round(wins / n, 3) if n else None,
                       "net": round(net, 2), "avg_net": round(net / n, 3) if n else None}

    losers = [(t, e) for t, e in rows if t.net_pnl < 0]
    winners = [(t, e) for t, e in rows if t.net_pnl > 0]
    loser_reached = {f"{x:g}": sum(1 for _, e in losers if e["peak_atr"] >= x) for x in LOSER_THRESHOLDS}
    captures = [c for t, e in winners if (c := _capture(t, e["peak"])) is not None]

    return {
        "matched": len(rows), "unmatched": unmatched,
        "peak_buckets": table,
        "losers": len(losers), "losers_once_in_profit": loser_reached,
        "winners": len(winners),
        "winner_avg_peak_atr": round(sum(e["peak_atr"] for _, e in winners) / len(winners), 2) if winners else None,
        "winner_median_capture": round(median(captures), 2) if captures else None,
        "loser_avg_worst_atr": round(sum(e["worst_atr"] for _, e in losers) / len(losers), 2) if losers else None,
    }


def excursion_findings(report: Optional[dict]) -> List[str]:
    if not report or not report["matched"]:
        return []
    out: List[str] = []
    n_losers = report["losers"]
    if n_losers >= MIN_LOSERS_FOR_FINDING:
        half = report["losers_once_in_profit"]["0.5"]
        if half / n_losers >= 0.30:
            out.append(
                f"{half} of {n_losers} losing trades ({round(100 * half / n_losers)}%) had first been "
                f"+0.5 ATR in profit, so an earlier lock-in had a chance to change them. Moving it "
                f"earlier also stops out some winners: test it, it is not proof.")
    n_winners = report["winners"]
    capture = report["winner_median_capture"]
    if n_winners >= MIN_WINNERS_FOR_FINDING and capture is not None and capture < 0.5:
        out.append(
            f"Winners kept a median {round(capture * 100)}% of their peak profit "
            f"({n_winners} winners); the rest was given back before exit.")
    return out

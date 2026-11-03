"""
core/diagnostics/confidence.py — does the AI's confidence separate winning
trades from losing ones?

The entry log line carries the ticket and the AI confidence the bot had when it
opened the trade. The broker's deal history carries the result of the same
position. Joining them by ticket answers the question directly, with no model
internals: group the trades by the confidence at entry and compare the results.

A position's ticket equals the opening order's ticket on a hedging account.
Entries without a ticket (log lines written before the entry line carried it)
cannot be joined and are counted as ``untagged``; entries whose ticket is not
among the closed trades (still open, or outside the period) are ``unmatched``.

Pure functions over the parsed entries and the paired trades.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.diagnostics.trades import Trade

BUCKETS = ((0.0, 0.55, "<0.55"), (0.55, 0.65, "0.55-0.65"),
           (0.65, 0.75, "0.65-0.75"), (0.75, 1.01, ">=0.75"))
MIN_MATCHED_FOR_FINDING = 12
MIN_PER_BUCKET = 3


def _bucket_name(conf: float) -> str:
    for low, high, name in BUCKETS:
        if low <= conf < high:
            return name
    return BUCKETS[-1][2]


def confidence_buckets(entries: List[dict], trades: List[Trade]) -> dict:
    """Closed-trade results grouped by the AI confidence at entry."""
    by_position: Dict[str, Trade] = {str(t.position_id): t for t in trades}
    grouped: Dict[str, List[Trade]] = {name: [] for _, _, name in BUCKETS}
    untagged = unmatched = matched = 0

    for entry in entries:
        ticket, conf = entry.get("ticket"), entry.get("conf")
        # ticket 0 is what the bot logs when the broker gave no order id
        if not ticket or str(ticket) == "0" or conf is None:
            untagged += 1
            continue
        trade = by_position.get(str(ticket))
        if trade is None:
            unmatched += 1
            continue
        matched += 1
        grouped[_bucket_name(float(conf))].append(trade)

    buckets = {}
    for _, _, name in BUCKETS:
        rows = grouped[name]
        n = len(rows)
        net = sum(t.net_pnl for t in rows)
        buckets[name] = {
            "trades": n,
            "wins": sum(1 for t in rows if t.net_pnl > 0),
            "win_rate": round(sum(1 for t in rows if t.net_pnl > 0) / n, 3) if n else None,
            "net": round(net, 2),
            "avg_net": round(net / n, 3) if n else None,
        }
    return {"matched": matched, "unmatched": unmatched, "untagged": untagged, "buckets": buckets}


def confidence_finding(conf: Optional[dict]) -> Optional[str]:
    """A finding only when the sample can support one.

    Compares the lowest and the highest confidence bucket that each hold at least
    ``MIN_PER_BUCKET`` trades. Higher confidence not paying better is the thing
    worth knowing; it is reported with its sample size, never as a verdict.
    """
    if not conf or conf["matched"] < MIN_MATCHED_FOR_FINDING:
        return None
    usable = [(name, b) for _, _, name in BUCKETS
              for b in [conf["buckets"][name]] if b["trades"] >= MIN_PER_BUCKET]
    if len(usable) < 2:
        return None
    (low_name, low), (high_name, high) = usable[0], usable[-1]
    if high["avg_net"] <= low["avg_net"]:
        return (f"Higher AI confidence did not pay better: avg net {high['avg_net']} per trade at "
                f"{high_name} (n={high['trades']}) vs {low['avg_net']} at {low_name} "
                f"(n={low['trades']}); {conf['matched']} trades matched. Small samples: treat as "
                f"a prompt to look, not a verdict.")
    return None

"""
core/diagnostics/logstats.py — facts the bot's own log can answer.

The log is the only record of things the broker never sees: why a signal did or
did not become an order, how often stops were modified, how large the real risk
was against the configured risk, and how the AI retraining scored. Everything
here is read from the line formats ``run.py`` writes.

Pure function over a list of lines; no files, no broker.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):\d{2},\d+ \[(\w+)\] (.*)$")
_ENTRY = re.compile(r"✅ (BUY|SELL) (\S+) vol=([\d.]+) @ ([\d.]+)\s+SL=([\d.]+)\s+TP=([\d.]+)")
_SL_MOD = re.compile(r"Modified SL ticket=(\d+)")
_MIN_LOT = re.compile(
    r"Minimum lot ([\d.]+) risks ([\d.]+) \(([\d.]+)% of equity\) against a ([\d.]+) budget "
    r"\(([\d.]+)% configured\)")
_ORDER_FAILED = re.compile(r"Order failed: (?:retcode=(\d+)|\S+ \((\d+)\))")
_BLOCKED = re.compile(r"Entry blocked: (.*)")
_SKIPPED = re.compile(r"Entry skipped: (.*)")
_BAR = re.compile(r"Bar \S+ \S+ \| direction=([+-]?\d) score=([\d.]+) ai_conf=([\d.]+) \((.*)\)$")
_CLOSE = re.compile(r"CLOSE (\S+) ticket=(\d+) pnl=([+-]?[\d.]+) reason=(.*)$")
_CHALLENGER = re.compile(r"challenger rejected \(oos accuracy ([\d.]+)")

_DIGITS = re.compile(r"[\d.]+")


def _norm(text: str) -> str:
    """Collapse numbers so 'score 44 < threshold' and 'score 39 < threshold' group."""
    return _DIGITS.sub("#", text).strip()


def parse_log(lines: Iterable[str]) -> dict:
    levels: Counter = Counter()
    entries: List[dict] = []
    entries_by_hour: Counter = Counter()
    sl_mods: Counter = Counter()
    sl_mod_minutes: Dict[str, Counter] = defaultdict(Counter)
    min_lots: List[dict] = []
    order_failed: Counter = Counter()
    blocked: Counter = Counter()
    skipped: Counter = Counter()
    signals: Counter = Counter()
    bot_closes: List[dict] = []
    challenger_acc: List[float] = []
    first = last = None
    n_lines = 0

    for raw in lines:
        m = _LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        n_lines += 1
        date, hour, minute, level, msg = m.groups()
        stamp = f"{date} {hour}:{minute}"
        first, last = first or stamp, stamp
        levels[level] += 1

        if (e := _ENTRY.search(msg)):
            side, symbol, vol, price, sl, tp = e.groups()
            entries.append({"time": stamp, "side": side, "symbol": symbol, "volume": float(vol),
                            "price": float(price), "sl": float(sl), "tp": float(tp),
                            "sl_distance": round(abs(float(price) - float(sl)), 5)})
            entries_by_hour[int(hour)] += 1
        elif (e := _SL_MOD.search(msg)):
            sl_mods[e.group(1)] += 1
            sl_mod_minutes[e.group(1)][stamp] += 1
        elif (e := _MIN_LOT.search(msg)):
            lot, risk, risk_pct, budget, budget_pct = e.groups()
            min_lots.append({"lot": float(lot), "risk": float(risk), "risk_pct": float(risk_pct),
                             "budget": float(budget), "budget_pct": float(budget_pct),
                             "overshoot": round(float(risk) / float(budget), 2) if float(budget) else None})
        elif (e := _ORDER_FAILED.search(msg)):
            order_failed[e.group(1) or e.group(2)] += 1
        elif (e := _BLOCKED.search(msg)):
            blocked[_norm(e.group(1))] += 1
        elif (e := _SKIPPED.search(msg)):
            skipped[_norm(e.group(1))] += 1
        elif (e := _BAR.search(msg)):
            direction, _score, _conf, reason = e.groups()
            signals["signal" if reason == "signal" else _norm(reason)] += 1
        elif (e := _CLOSE.search(msg)):
            bot_closes.append({"ticket": e.group(2), "pnl": float(e.group(3)),
                               "reason": e.group(4).strip()})
        elif (e := _CHALLENGER.search(msg)):
            challenger_acc.append(float(e.group(1)))

    overshoots = [x["overshoot"] for x in min_lots if x["overshoot"] is not None]
    return {
        "lines": n_lines,
        "first": first, "last": last,
        "levels": dict(levels),
        "entries": entries,
        "entries_by_hour": dict(sorted(entries_by_hour.items())),
        "sl_modifications": {
            "total": sum(sl_mods.values()),
            "per_ticket": dict(sl_mods.most_common(10)),
            "max_in_one_minute": max((c for minutes in sl_mod_minutes.values()
                                      for c in minutes.values()), default=0),
        },
        "min_lot_risk": {
            "n": len(min_lots),
            "avg_risk_pct": round(sum(x["risk_pct"] for x in min_lots) / len(min_lots), 2) if min_lots else None,
            "avg_configured_pct": round(sum(x["budget_pct"] for x in min_lots) / len(min_lots), 2) if min_lots else None,
            "avg_overshoot": round(sum(overshoots) / len(overshoots), 2) if overshoots else None,
            "max_overshoot": max(overshoots) if overshoots else None,
        },
        "order_failed": dict(order_failed),
        "entries_blocked": dict(blocked.most_common(10)),
        "entries_skipped": dict(skipped.most_common(10)),
        "signals": dict(signals.most_common(15)),
        "bot_closes": bot_closes,
        "ai_challenger_oos_accuracy": challenger_acc,
    }

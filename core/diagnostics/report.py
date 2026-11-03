"""
core/diagnostics/report.py — one readable summary from the trade statistics and
the log statistics, with rule-based findings that point at what looks wrong.

The findings are plain thresholds, not judgement: each one names the numbers it
rests on so a reader can check it. They are prompts to look, not conclusions.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from core.diagnostics.confidence import confidence_finding
from core.diagnostics.excursion import excursion_findings


def _minutes(stamp: str) -> int:
    return int(datetime.strptime(stamp, "%Y-%m-%d %H:%M").timestamp() // 60)


def findings(stats: dict, log: dict, conf: Optional[dict] = None,
             excursion: Optional[dict] = None) -> List[str]:
    out: List[str] = []

    # Whose trades are these? Money lost by hand must never read as the bot's result.
    origin = stats.get("by_origin") or {}
    other = origin.get("other")
    if other and other["trades"]:
        bot = origin.get("bot", {"trades": 0, "net": 0.0, "win_rate": None})
        hand = stats.get("hand_closed", {"trades": 0, "net": 0.0})
        out.append(
            f"{other['trades']} of {stats['trades']} trades (net {other['net']}) were NOT opened by "
            f"the bot: a different magic number, so manual trades or another program. "
            f"{hand['trades']} trades (net {hand['net']}) were closed by hand from the MT5 app or "
            f"terminal. The bot's own {bot['trades']} trades: net {bot['net']}, win rate "
            f"{bot['win_rate']}.")

    sl = log.get("sl_modifications", {})
    per_ticket = sl.get("per_ticket", {})
    if per_ticket:
        ticket, count = next(iter(per_ticket.items()))
        if count >= 20 or sl.get("max_in_one_minute", 0) >= 10:
            out.append(f"Stop-loss modified {count} times on ticket {ticket}; up to "
                       f"{sl['max_in_one_minute']} in a single minute (broker round trips, "
                       f"risk of rate limits).")

    ml = log.get("min_lot_risk", {})
    if ml.get("avg_overshoot") and ml["avg_overshoot"] >= 2:
        out.append(f"The minimum lot risks on average {ml['avg_overshoot']}x the per-trade budget "
                   f"(real {ml['avg_risk_pct']}% of equity vs {ml['avg_configured_pct']}% "
                   f"configured; worst {ml['max_overshoot']}x). Size reductions do nothing at the "
                   f"minimum lot.")

    if stats.get("trades"):
        if stats["expectancy"] is not None and stats["expectancy"] < 0:
            out.append(f"Negative expectancy: {stats['expectancy']} per trade over "
                       f"{stats['trades']} trades (net {stats['net']}).")
        if stats["profit_factor"] is not None and stats["profit_factor"] < 1:
            out.append(f"Profit factor {stats['profit_factor']} is below 1.")
        if stats["losses"] >= 3 and stats["quick_losses"] / stats["losses"] >= 0.5:
            out.append(f"{stats['quick_losses']} of {stats['losses']} losses closed within 5 "
                       f"minutes of entry (stop probably too tight for the noise).")
        sl_share = stats["by_exit_reason"].get("sl", {}).get("trades", 0) / stats["trades"]
        if stats["trades"] >= 5 and sl_share >= 0.6:
            out.append(f"{round(sl_share * 100)}% of trades ended at the stop-loss.")
        if stats["max_consecutive_losses"] >= 4:
            out.append(f"Longest losing streak: {stats['max_consecutive_losses']} trades.")
    else:
        out.append("No closed trades were found in the period.")

    if log.get("order_failed"):
        codes = ", ".join(f"{c} x{n}" for c, n in log["order_failed"].items())
        note = " (10018 = market closed)" if "10018" in log["order_failed"] else ""
        out.append(f"Orders failed: {codes}{note}.")

    acc = log.get("ai_challenger_oos_accuracy", [])
    if len(acc) >= 3 and sum(acc) / len(acc) < 0.45:
        out.append(f"AI retraining scores below a coin flip: mean out-of-sample accuracy "
                   f"{round(sum(acc) / len(acc), 3)} over {len(acc)} attempts.")

    note = confidence_finding(conf)
    if note:
        out.append(note)
    out.extend(excursion_findings(excursion))

    entries = sorted(log.get("entries", []), key=lambda e: e["time"])
    clustered = sum(1 for a, b in zip(entries, entries[1:])
                    if a["side"] == b["side"] and 0 <= _minutes(b["time"]) - _minutes(a["time"]) <= 15)
    if clustered >= 3:
        out.append(f"{clustered} entries were opened within 15 minutes of the previous one in the "
                   f"same direction.")
    return out


def _table(rows: List[tuple], header: tuple) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def build_summary(meta: dict, stats: dict, log: dict, notes: Optional[List[str]] = None,
                  conf: Optional[dict] = None, excursion: Optional[dict] = None) -> str:
    md: List[str] = ["# Trading-gold diagnostics", ""]
    md += [f"- Generated: {meta.get('generated', '')}",
           f"- Period: last {meta.get('days', '?')} days",
           f"- Account: {meta.get('account_type', 'unknown')} {meta.get('account', '')}"
           f" | balance {meta.get('balance', '?')} | equity {meta.get('equity', '?')}",
           f"- Log window: {log.get('first')} -> {log.get('last')} ({log.get('lines', 0)} lines)", ""]
    for note in notes or []:
        md.append(f"> {note}")
    md.append("")

    md += ["## Findings", ""]
    items = findings(stats, log, conf, excursion)
    md += [f"- {f}" for f in items] if items else ["- Nothing crossed a threshold."]
    md.append("")

    md += ["## Closed trades (broker history)", ""]
    if stats["trades"]:
        md += [_table([(stats["trades"], stats["net"], stats["win_rate"], stats["avg_win"],
                        stats["avg_loss"], stats["profit_factor"], stats["expectancy"],
                        stats["max_consecutive_losses"], stats["median_duration_s"])],
                      ("trades", "net", "win rate", "avg win", "avg loss", "profit factor",
                       "expectancy", "max loss streak", "median hold (s)")), ""]
        if stats.get("by_origin"):
            md += ["By origin (who opened the trade):", "",
                   _table([(k, v["trades"], v["net"], v["win_rate"])
                           for k, v in stats["by_origin"].items()],
                          ("origin", "trades", "net", "win rate")), ""]
        md += ["By exit reason:", "",
               _table([(k, v["trades"], v["net"], v["win_rate"]) for k, v in stats["by_exit_reason"].items()],
                      ("exit", "trades", "net", "win rate")), ""]
        md += ["By direction:", "",
               _table([(k, v["trades"], v["net"], v["win_rate"]) for k, v in stats["by_direction"].items()],
                      ("direction", "trades", "net", "win rate")), ""]
        md += ["By hour opened (broker time):", "",
               _table([(k, v["trades"], v["net"], v["win_rate"]) for k, v in stats["by_hour"].items()],
                      ("hour", "trades", "net", "win rate")), ""]
    else:
        md += ["No closed trades in the period.", ""]

    sl, ml = log["sl_modifications"], log["min_lot_risk"]
    md += ["## Execution (from the bot log)", "",
           f"- Entries placed: {len(log['entries'])}",
           f"- Stop-loss modifications: {sl['total']} (max {sl['max_in_one_minute']} in one minute)",
           f"- Order failures: {log['order_failed'] or 'none'}",
           f"- Minimum-lot risk: {ml['n']} trades, real {ml['avg_risk_pct']}% vs configured "
           f"{ml['avg_configured_pct']}% (avg {ml['avg_overshoot']}x budget)", ""]

    md += ["## Why signals did or did not trade", ""]
    md += [_table(list(log["signals"].items()), ("bar outcome", "count")), ""]
    if log["entries_blocked"] or log["entries_skipped"]:
        md += [_table(list({**log["entries_blocked"], **log["entries_skipped"]}.items()),
                      ("blocked / skipped", "count")), ""]

    md += ["## How far trades ran before they ended", ""]
    if excursion and excursion["matched"]:
        md += [_table([(k, v["trades"], v["win_rate"], v["net"], v["avg_net"])
                       for k, v in excursion["peak_buckets"].items()],
                      ("peak profit reached", "trades", "win rate", "net", "avg net")), ""]
        once = excursion["losers_once_in_profit"]
        md += [f"- Losing trades: {excursion['losers']}; had first reached +0.3 ATR: {once['0.3']}, "
               f"+0.5 ATR: {once['0.5']}, +1.0 ATR: {once['1']}",
               f"- Winning trades: {excursion['winners']}; average peak "
               f"{excursion['winner_avg_peak_atr']} ATR; median share of the peak kept "
               f"{excursion['winner_median_capture']}",
               f"- {excursion['matched']} trades matched to a TRADE END line; "
               f"{excursion['unmatched']} lines had no closed trade in the period. "
               f"Sampled about once a second, so peaks are a floor.", ""]
    else:
        md += ["No TRADE END lines matched a closed trade yet. The bot now writes one when each "
               "trade ends; results appear after trades close.", ""]

    md += ["## AI confidence vs results", ""]
    if conf and conf["matched"]:
        md += [_table([(k, v["trades"], v["win_rate"], v["net"], v["avg_net"])
                       for k, v in conf["buckets"].items()],
                      ("confidence at entry", "trades", "win rate", "net", "avg net")),
               "",
               f"{conf['matched']} trades matched to their entry confidence; "
               f"{conf['unmatched']} entries had no closed trade yet; {conf['untagged']} entries "
               f"were logged before the confidence was recorded.", ""]
    else:
        md += ["No trades could be matched to an entry confidence yet. Entries now log their "
               "ticket and confidence; results appear after those trades close.", ""]

    acc = log["ai_challenger_oos_accuracy"]
    md += ["## AI model quality (from the bot's own retraining)", ""]
    if log.get("ai_models"):
        md += [_table([(m["time"], f"#{m['run']}", m["train"], m["oos_acc"], m["auc"], m["base"],
                        m["lift"]) for m in log["ai_models"][-5:]],
                      ("time", "model", "train rows", "oos acc", "auc", "baseline", "lift")), ""]
    md += [f"- Challenger attempts rejected: {len(acc)}"
           + (f" (accuracy min {min(acc)}, max {max(acc)})" if acc else ""), ""]
    return "\n".join(md)

"""
scripts/replay_management.py — what would other break-even / trailing settings have done
on the SAME entries? Offline: reads files, never touches a broker.

    python scripts/replay_management.py --log trading_bot.log --trades diagnostics/closed_trades.csv \
        --bars data/history/XAUUSDm_M1.csv --atr-bars data/history/XAUUSDm_M5.csv \
        --set mine=lock_min_distance_atr:1.0,min_sl_step_atr:0.1

Presets: current (the bot's defaults), wider (lock distance 1.0 ATR), no-breakeven.
Always read the CALIBRATION line first: it replays the current settings and compares with
what actually happened. If it is far off, do not trust the other rows.
Only entries logged with a ticket can be replayed (log lines since the confidence update).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.backtesting.management_replay import (  # noqa: E402
    PRESETS, ManagementParams, build_replay_trades, calibration, paired_difference, run_replay,
    summarise, with_params)
from core.diagnostics.logstats import parse_log  # noqa: E402

MAX_PATH_BAR_MINUTES = 4


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def parse_set(text: str) -> tuple:
    name, _, body = text.partition("=")
    changes = {}
    for part in filter(None, body.split(",")):
        key, _, val = part.partition(":")
        low = val.strip().lower()
        changes[key.strip()] = (low == "true") if low in ("true", "false") else float(val)
    return name.strip(), changes


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--trades", type=Path, help="closed_trades.csv (position_id, net_pnl) for calibration")
    ap.add_argument("--bars", required=True, type=Path, help="fine bars (M1) the trade path is replayed on")
    ap.add_argument("--atr-bars", type=Path, help="M5 bars for the ATR; required if --bars is finer than 4 min")
    ap.add_argument("--set", action="append", default=[], help="name=key:value,key:value")
    ap.add_argument("--spread", type=float, default=0.0, help="price units charged per trade")
    ap.add_argument("--shift-hours", type=float, default=0.0, help="bar time minus log time, in hours")
    args = ap.parse_args(argv)

    path_bars = load_bars(args.bars)
    step = path_bars.index.to_series().diff().median()
    if pd.notna(step) and step < pd.Timedelta(minutes=MAX_PATH_BAR_MINUTES) and not args.atr_bars:
        print("error: --bars is finer than 4 minutes, so its ATR would be tiny. Pass --atr-bars (M5).")
        return 2
    atr_bars = load_bars(args.atr_bars) if args.atr_bars else path_bars

    entries = parse_log(args.log.read_text(encoding="utf-8", errors="replace").splitlines())["entries"]
    actual = {}
    if args.trades and args.trades.exists():
        with args.trades.open(newline="", encoding="utf-8") as fh:
            actual = {str(r["position_id"]): float(r["net_pnl"]) for r in csv.DictReader(fh)}
    trades, skipped = build_replay_trades(entries, atr_bars, actual, args.shift_hours)
    print(f"{len(entries)} entries in the log, {len(trades)} replayable, skipped: {skipped}")
    if not trades:
        return 1

    sets = {name: with_params(ManagementParams(), **chg) for name, chg in PRESETS.items()}
    for text in args.set:
        name, chg = parse_set(text)
        sets[name] = with_params(ManagementParams(), **chg)

    results = {name: run_replay(trades, path_bars, p, spread=args.spread) for name, p in sets.items()}
    print(f"CALIBRATION (current settings vs what happened): {calibration(results['current'])}")
    print(f"{'set':<14}{'n':>5}{'net':>9}{'win%':>7}{'avgW':>8}{'avgL':>8}{'PF':>6}{'exp':>8}   vs current")
    for name, rows in results.items():
        s = summarise(rows)
        d = paired_difference(results["current"], rows) if name != "current" else None
        diff = f"{d['mean']:+.3f}/trade z={d['z']}" if d and d["mean"] is not None else ""
        print(f"{name:<14}{s['trades']:>5}{s['net']:>9}{str(s['win_rate']):>7}{str(s['avg_win']):>8}"
              f"{str(s['avg_loss']):>8}{str(s['profit_factor']):>6}{str(s['expectancy']):>8}   {diff}")
    print("|z| < 2 means the difference is not distinguishable from chance on this many trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

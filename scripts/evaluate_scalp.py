"""
scripts/evaluate_scalp.py — which scalping strategies work on THIS market's data?

Runs the bot's own scalping engine (core/strategy/scalp.py) over exported bars and,
for each of the seven strategies, answers:

1. How often does it prove itself on the trailing window the live bot sees?
2. After it has proven itself, does the edge hold on the bars that follow?
   (the out-of-sample number that matters; a proof that vanishes is chance)
3. For reference, what would the strategy have done if it were always on?

    .venv\\Scripts\\python.exe scripts\\evaluate_scalp.py --csv data\\history\\XAUUSDm_M5.csv

Each window is the trailing ``--window`` bars ending at ``e``; the engine is run on
exactly that frame (as the bot would), and if a strategy is proven, its calls on the
next ``--forward`` bars are scored with the same barrier rule: +1 when the +1 ATR
barrier is touched first in the called direction, -1 when the opposite barrier is, 0
for a timeout. Spread and commission are NOT included, so a small positive edge is
not yet a profit.

Read the result with the multiple-testing problem in mind: seven strategies were
looked at, so one of them looking good by chance is expected now and then. Trust a
strategy only if its forward edge is clearly positive over many calls AND it holds
on other periods and in a demo account.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.strategy.scalp import ALL_STRATEGIES, ScalpConfig, ScalpEngine, score_signals   # noqa: E402

MIN_FORWARD_CALLS = 20
MIN_FORWARD_EDGE = 0.05


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    cols = [c for c in ("open", "high", "low", "close", "tick_volume") if c in df.columns]
    return df[cols].astype(float)


def evaluate(df: pd.DataFrame, cfg: ScalpConfig, window: int, step: int, forward: int) -> Dict[str, dict]:
    engine = ScalpEngine(cfg)
    sigs = engine.signals(df)
    outcomes = engine.outcomes(df)
    n, horizon = len(df), cfg.horizon

    stats = {name: {"windows": 0, "proven": 0, "fw": 0, "fl": 0, "fc": 0, "aw": 0, "al": 0, "ac": 0}
             for name in sigs}
    for end in range(window, n - forward - horizon, step):
        report = engine.evaluate(df.iloc[end - window:end]).report
        stop = min(end + forward, n - 1 - horizon)
        for name, sig in sigs.items():
            s = stats[name]
            s["windows"] += 1
            w, l, c = score_signals(sig, outcomes, end, stop)
            s["aw"], s["al"], s["ac"] = s["aw"] + w, s["al"] + l, s["ac"] + c
            if report.get(name, {}).get("proven"):
                s["proven"] += 1
                s["fw"], s["fl"], s["fc"] = s["fw"] + w, s["fl"] + l, s["fc"] + c

    out = {}
    for name, s in stats.items():
        out[name] = {
            "windows": s["windows"], "proven_windows": s["proven"],
            "proven_share": s["proven"] / s["windows"] if s["windows"] else None,
            "forward_calls_proven": s["fc"],
            "forward_edge_proven": (s["fw"] - s["fl"]) / s["fc"] if s["fc"] else None,
            "forward_calls_always_on": s["ac"],
            "forward_edge_always_on": (s["aw"] - s["al"]) / s["ac"] if s["ac"] else None,
        }
    return out


def verdict(r: dict) -> str:
    if not r["proven_windows"]:
        return "never proved itself"
    if r["forward_calls_proven"] < MIN_FORWARD_CALLS:
        return "proved itself, but too few calls afterwards to judge"
    if r["forward_edge_proven"] >= MIN_FORWARD_EDGE:
        return "edge HELD after proof"
    return "edge did NOT hold after proof (chance)"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--window", type=int, default=1200, help="bars the live bot sees")
    parser.add_argument("--step", type=int, default=150)
    parser.add_argument("--forward", type=int, default=300, help="bars scored after each proof")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    parser.add_argument("--min-z", type=float, default=2.7)
    parser.add_argument("--open-hour", type=int, default=7, help="opening-range hour in the CSV's clock")
    args = parser.parse_args(argv)

    df = load_bars(args.csv)
    need = args.window + args.forward + args.horizon + 10
    if len(df) < need:
        print(f"not enough data: need at least {need} bars, have {len(df)}")
        return 1
    cfg = ScalpConfig(strategies=tuple(s.strip() for s in args.strategies.split(",") if s.strip()),
                      horizon=args.horizon, min_z=args.min_z, open_hour=args.open_hour,
                      check_bars=min(900, args.window - 100))
    results = evaluate(df, cfg, args.window, args.step, args.forward)

    def fmt(x):
        return "   n/a" if x is None else f"{x:+.3f}"

    first = next(iter(results.values()))
    print(f"{len(df)} bars, {first['windows']} windows of {args.window}, forward {args.forward} bars")
    print(f"{'strategy':14}{'proven':>9}{'fwd calls':>11}{'fwd edge':>10}{'always-on edge':>16}   verdict")
    for name, r in results.items():
        share = f"{(r['proven_share'] or 0) * 100:.0f}%"
        print(f"{name:14}{share:>9}{r['forward_calls_proven']:>11}{fmt(r['forward_edge_proven']):>10}"
              f"{fmt(r['forward_edge_always_on']):>16}   {verdict(r)}")
    held = [n for n, r in results.items() if verdict(r) == "edge HELD after proof"]
    print()
    print(f"strategies whose edge held after proof: {', '.join(held) if held else 'none'}")
    print("Seven strategies were examined, so one looking good by chance is expected. Confirm on "
          "other periods and on a demo account before relying on any of them (spread and "
          "commission are not included).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

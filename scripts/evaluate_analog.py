"""
scripts/evaluate_analog.py — does "history repeats" work on THIS market's data?

Runs the bot's own engine (core/strategy/analog.py) over exported bars and answers
two questions:

1. How often does the strategy prove itself? (the bot only lets it vote when it
   has a statistically clear edge on its own recent history)
2. When it has proven itself, does that edge hold on the bars AFTER the proof?
   This is the out-of-sample number that matters: a proof that disappears at once
   is chance, not structure.

    .venv\\Scripts\\python.exe scripts\\evaluate_analog.py --csv data\\history\\XAUUSDm_M5.csv

Each window ends at bar ``e``; the engine is evaluated on bars up to ``e`` only; if
it is proven, its calls on the next ``--forward`` bars are scored using only the
history before each call (never the future). Edge is the mean R-multiple of the
directional calls: +1 when the +1 ATR barrier is touched first in the called
direction, -1 when the opposite barrier is, 0 for a timeout. Spread and commission
are not included, so a small positive edge here is not yet a profit.

Read the result before enabling the strategy for real orders. If almost no window
is proven, or the forward edge of proven windows is near zero, the market does not
repeat in this way at this timeframe, and the strategy will mostly abstain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.strategy.analog import AnalogConfig, AnalogEngine          # noqa: E402


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    return df[["open", "high", "low", "close"]].astype(float)


def forward_score(engine: AnalogEngine, vectors, outcomes, start: int, stop: int) -> tuple:
    """(wins, losses, calls) of the engine's raw votes on bars ``start..stop-1``."""
    wins = losses = calls = 0
    for t in range(start, stop):
        sig = engine._vote(vectors, outcomes, t)
        if sig.abstained or sig.direction == 0:
            continue
        calls += 1
        result = outcomes[t]
        if np.isnan(result):
            continue
        if (sig.direction == 1) == (result == 1.0):
            wins += 1
        else:
            losses += 1
    return wins, losses, calls


def evaluate(df: pd.DataFrame, cfg: AnalogConfig, window: int, step: int, forward: int) -> dict:
    engine = AnalogEngine(cfg)
    vectors, outcomes, _ = engine._prepare(df)
    n = len(df)
    horizon = cfg.horizon
    windows = proven = 0
    fw = fl = fc = 0                               # forward totals for proven windows
    aw = al = ac = 0                               # forward totals for ALL windows (strategy always on)
    per_window: List[dict] = []

    for end in range(window, n - forward - horizon, step):
        sub = df.iloc[:end]
        if len(sub) > window:
            sub = sub.iloc[-window:]
        # evaluate() on the trailing frame is what the live bot sees
        sig = engine.evaluate(sub)
        windows += 1
        w, l, c = forward_score(engine, vectors, outcomes, end, min(end + forward, n - horizon - 1))
        aw, al, ac = aw + w, al + l, ac + c
        if not sig.abstained:
            proven += 1
            fw, fl, fc = fw + w, fl + l, fc + c
            per_window.append({"end": str(df.index[end - 1]), "proof_edge": sig.edge,
                               "forward_calls": c,
                               "forward_edge": (w - l) / c if c else None})
    return {
        "bars": n, "windows": windows, "proven_windows": proven,
        "proven_share": proven / windows if windows else None,
        "forward_calls_proven": fc,
        "forward_edge_proven": (fw - fl) / fc if fc else None,
        "forward_calls_always_on": ac,
        "forward_edge_always_on": (aw - al) / ac if ac else None,
        "per_window": per_window,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--window", type=int, default=1200, help="bars the live bot sees")
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--forward", type=int, default=200, help="bars scored after each proof")
    parser.add_argument("--analog-window", type=int, default=24)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args(argv)

    df = load_bars(args.csv)
    if len(df) < args.window + args.forward + args.horizon + 10:
        print(f"not enough data: need at least {args.window + args.forward + args.horizon + 10} bars, "
              f"have {len(df)}")
        return 1
    cfg = AnalogConfig(window=args.analog_window, k=args.k, horizon=args.horizon)
    r = evaluate(df, cfg, args.window, args.step, args.forward)

    def fmt(x):
        return "n/a (no calls)" if x is None else f"{x:+.3f}"

    print(f"{r['bars']} bars, {r['windows']} windows of {args.window}")
    print(f"proved itself in      : {r['proven_windows']} windows ({(r['proven_share'] or 0) * 100:.0f}%)")
    print(f"forward edge, proven  : {fmt(r['forward_edge_proven'])}  over {r['forward_calls_proven']} calls")
    print(f"forward edge, always-on: {fmt(r['forward_edge_always_on'])}  over {r['forward_calls_always_on']} calls")
    if not r["proven_windows"]:
        print("verdict: the strategy never proved itself here; the bot would keep it abstaining.")
    elif (r["forward_edge_proven"] or 0) <= 0.05:
        print("verdict: it proved itself on paper but the edge did NOT hold on later bars. "
              "Treat the proofs as chance; consider disabling ANALOG_ENABLED.")
    else:
        print("verdict: the edge held after proof on these bars (before spread and commission). "
              "Confirm on other periods and on a demo account before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

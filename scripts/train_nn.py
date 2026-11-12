"""
scripts/train_nn.py — train and walk-forward-test the sequence model on exported bars.

Reads a CSV written by export_bars.py, measures out-of-sample performance with a
walk-forward test, trains the model that would go live, and saves it.

    .venv\\Scripts\\python.exe scripts\\train_nn.py --csv data\\history\\XAUUSDm_M5.csv --out data\\models\\nn_XAUUSDm_M5.pt

Read the report before acting on it:

* ``edge`` is the mean R-multiple of the trades the model would take at the
  threshold chosen on each fold's validation rows (``--threshold`` is the
  minimum), counted one position at a time, before spread and commission. It
  is "unmeasured" below 30 non-overlapping out-of-sample trades. The report
  also shows its lower confidence bound, the naive always-one-direction
  baseline and the timing skill beyond it; the predictor refuses the model
  unless all of them pass.
* A positive number on one run is a hypothesis, not proof. Run it on more
  history and different periods, and trade it on paper first.
* Nothing here changes the live bot. The saved file is only used by
  core/ai/nn_predictor.py, which refuses a model whose edge is unmeasured,
  not confidently above its floor, or no better than the naive baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    return df[["open", "high", "low", "close", "tick_volume"]].astype(float)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--barrier-mult", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-bars", type=int, default=6000)
    parser.add_argument("--test-bars", type=int, default=1500)
    args = parser.parse_args(argv)

    try:
        from core.ai.nn_model import TrainConfig, save_artifact, train_final
    except ImportError as exc:
        print(f"torch is required: {exc}")
        return 1

    df = load_bars(args.csv)
    cfg = TrainConfig(window=args.window, horizon=args.horizon,
                      barrier_mult=args.barrier_mult, threshold=args.threshold,
                      epochs=args.epochs)
    print(f"{len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    if len(df) < args.train_bars + args.test_bars + cfg.horizon:
        print(f"not enough data: need at least {args.train_bars + args.test_bars + cfg.horizon} bars")
        return 1

    artifact = train_final(df, cfg, args.train_bars, args.test_bars)
    oos = artifact["oos"]

    from core.ai.nn_predictor import oos_report_lines
    print(f"walk-forward folds : {oos['n_folds']}")
    for line in oos_report_lines(oos):
        print(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(str(args.out), artifact)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
scripts/train_nn.py — train and walk-forward-test the sequence model on exported bars.

Reads a CSV written by export_bars.py, measures out-of-sample performance with a
walk-forward test, trains the model that would go live, and saves it.

    .venv\\Scripts\\python.exe scripts\\train_nn.py --csv data\\history\\XAUUSDm_M5.csv --out data\\models\\nn_XAUUSDm_M5.pt

Read the report before acting on it:

* ``edge`` is the mean R-multiple of the trades the model would take at the
  confidence threshold, before spread and commission. It is "unmeasured" when
  fewer than 30 out-of-sample trades were taken.
* A positive number on one run is a hypothesis, not proof. Run it on more
  history and different periods, and trade it on paper first.
* Nothing here changes the live bot. The saved file is only used by
  core/ai/nn_predictor.py, which refuses a model whose edge is unmeasured or
  below its floor.
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

    print(f"walk-forward folds : {oos['n_folds']}")
    print(f"out-of-sample trades: {oos['trades']} at confidence >= {cfg.threshold}")
    if oos["edge"] is None:
        print("edge               : UNMEASURED (too few out-of-sample trades) "
              "- the predictor will refuse this model")
    else:
        print(f"precision          : {oos['precision']:.3f}")
        print(f"edge (mean R)      : {oos['edge']:+.3f}  (before spread and commission)")
        if oos["edge"] <= 0:
            print("verdict            : NO EDGE - the predictor will refuse this model")
        else:
            print("verdict            : positive on this data; confirm on other periods "
                  "and on paper before trusting it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(str(args.out), artifact)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

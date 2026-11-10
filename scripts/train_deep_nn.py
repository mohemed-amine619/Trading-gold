"""Train the dual-stream deep network (temporal + market-location) from a bar CSV.

    .venv/bin/python scripts/train_deep_nn.py --csv data/cache/XAUUSDm_M5.csv \
        --out data/models/nn_XAUUSDm_M5.pt --preset small

What this does that the plain trainer does not:

* walk-forward validation with a purge gap of one label horizon, so no training
  label is decided by prices inside a test fold (no leakage);
* the market-location features are computed causally (bar i sees bars <= i);
* AdamW weight decay, a reduce-on-plateau learning-rate schedule and early
  stopping on a validation block that sits after the training block;
* an auxiliary expected-excursion head trained alongside the three classes.

The saved model carries its measured out-of-sample edge. The predictor refuses a
model with no measured edge or one below NN_MIN_OOS_EDGE, so a large network
that merely memorised the training data never votes. The default preset is
small on purpose: a few hundred windows cannot support the large one.
CSV columns: time, open, high, low, close, tick_volume (as export_bars.py writes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    return df[["open", "high", "low", "close", "tick_volume"]].astype(float)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preset", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--barrier-mult", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", type=int, default=1, help="seeds averaged")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 (AdamW)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--struct-stride", type=int, default=8)
    parser.add_argument("--train-bars", type=int, default=6000)
    parser.add_argument("--test-bars", type=int, default=1500)
    args = parser.parse_args(argv)

    try:
        from core.ai.deep_nn import build_dual
        from core.ai.deep_layout import N_SEQ
        from core.ai.location_features import N_STRUCT
        from core.ai.nn_model import TrainConfig, save_artifact, train_final
    except ImportError as exc:
        print(f"torch is required: {exc}")
        return 1

    df = load_bars(args.csv)
    cfg = TrainConfig(arch="dual", preset=args.preset, window=args.window,
                      horizon=args.horizon, barrier_mult=args.barrier_mult,
                      threshold=args.threshold, epochs=args.epochs, n_models=args.models,
                      lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                      patience=args.patience, struct_stride=args.struct_stride)
    need = args.train_bars + args.test_bars + cfg.horizon
    print(f"{len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    n_params = sum(p.numel() for p in build_dual(N_SEQ + N_STRUCT, cfg.preset, cfg.dropout).parameters())
    print(f"preset {cfg.preset}: {n_params:,} parameters")
    if len(df) < need:
        print(f"not enough data: need at least {need} bars")
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
        print("verdict            : " + (
            "NO EDGE - the predictor will refuse this model" if oos["edge"] <= 0 else
            "positive on this data; confirm on other periods and on paper before trusting it"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(str(args.out), artifact)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

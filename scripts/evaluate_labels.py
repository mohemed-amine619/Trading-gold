"""
scripts/evaluate_labels.py — how predictable is each training label, walk-forward?

Compares AI_LABEL_MODE=close (legacy) with AI_LABEL_MODE=barrier on the same
exported bars, using the bot's own feature builder, label code, purged split
and model pipeline. Nothing is re-implemented here, so the numbers describe the
code the live bot runs.

    .venv\\Scripts\\python.exe scripts\\evaluate_labels.py --csv data\\history\\XAUUSDm_M15.csv --sl-mult 1.3

Read this before acting on the output:

* Predictability is not profit. A label the model predicts well can still
  lose money once spread, slippage and exits are applied. This tool answers
  "does the model learn the label better than chance", and nothing more.
* Windows overlap (step < window), so the windows are not independent samples.
  Use the spread of results across windows, not a single number, to judge.
* Enable barrier labels only if the out-of-sample edge is consistently
  positive and clearly better than the close label on the same bars.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches run.py's own minimums: below these, a fit or a validation split measures noise.
MIN_VALIDATION = 40
MIN_TRAIN = 150


def load_bars(path: Path) -> pd.DataFrame:
    """Read a CSV written by export_bars.py into the frame the bot's feature builder expects."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    return df[["open", "high", "low", "close", "tick_volume", "spread"]].astype(float)


def make_config(run, mode: str, horizon: int, sl_mult: float):
    """The config keys AIEnsemble reads, with run.py's CONFIG for everything else."""
    base = run.CONFIG
    return type("EvalConfig", (), {
        "AI_CONFIDENCE_THRESHOLD": base["ai_confidence_threshold"],
        "AI_TRAIN_BARS": base["ai_train_bars"],
        "AI_RETRAIN_BARS": base["ai_retrain_bars"],
        "AI_VALIDATION_FRACTION": 0.20,
        "AI_MIN_OOS_ACCURACY": 0.0,
        "AI_MIN_OOS_AUC": 0.0,
        "AI_MIN_OOS_EDGE": -1.0,       # evaluation only: never gate the measurement itself
        "AI_PROMOTE_TOLERANCE": 0.0,
        "AI_CALIBRATE": False,
        "AI_MIN_EDGE_FOR_SIZE_UP": base["ai_min_edge_for_size_up"],
        "AI_LABEL_MODE": mode,
        "AI_LABEL_HORIZON": horizon,
        "AI_SL_ATR_MULT": sl_mult,
        "AI_MODEL_PERSIST": False,
        "MODEL_TIMEFRAME": None,
    })()


def evaluate_window(run, df: pd.DataFrame, mode: str, horizon: int, sl_mult: float,
                    val_frac: float = 0.20) -> Optional[Dict]:
    """Train on the earlier part of ``df``, score the later part. None if too small."""
    ai = run.AIEnsemble("EVAL", make_config(run, mode, horizon, sl_mult))
    X, y, _ = ai._build_xy(df)
    if X is None or len(X) < MIN_TRAIN + MIN_VALIDATION:
        return None

    n_val = int(len(X) * val_frac)
    if n_val < MIN_VALIDATION:
        return None
    try:
        train_slice, val_slice = run._purged_split(len(X), n_val, ai._label_horizon())
    except ValueError:
        return None

    y_tr, y_va = y[train_slice], y[val_slice]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 1:
        return None

    pipe = run._make_pipeline()
    pipe.fit(X[train_slice], y_tr, model__sample_weight=np.linspace(0.4, 1.0, len(y_tr)))
    p_up = pipe.predict_proba(X[val_slice])[:, 1]

    accuracy = float(np.mean((p_up >= 0.5).astype(int) == y_va))
    base_rate = float(np.mean(y_va))
    baseline = max(base_rate, 1.0 - base_rate)   # best trivial predictor, as in run.py
    auc = float(roc_auc_score(y_va, p_up)) if 0.0 < base_rate < 1.0 else None
    return {
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "accuracy": accuracy,
        "auc": auc,
        "base_rate": base_rate,
        "edge": accuracy - baseline,
    }


def walk_forward(run, df: pd.DataFrame, mode: str, horizon: int, sl_mult: float,
                 window: int = 2000, step: int = 500) -> List[Dict]:
    results = []
    for end in range(window, len(df) + 1, step):
        result = evaluate_window(run, df.iloc[end - window:end], mode, horizon, sl_mult)
        if result is not None:
            results.append(result)
    return results


def summarise(results: List[Dict]) -> Dict:
    if not results:
        return {"windows": 0}
    aucs = [r["auc"] for r in results if r["auc"] is not None]
    return {
        "windows": len(results),
        "mean_edge": float(np.mean([r["edge"] for r in results])),
        "min_edge": float(np.min([r["edge"] for r in results])),
        "share_edge_positive": float(np.mean([r["edge"] > 0 for r in results])),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "mean_base_rate": float(np.mean([r["base_rate"] for r in results])),
    }


def _load_run():
    """Import run.py from the repository root. On Windows this needs MetaTrader5 installed."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("run")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--sl-mult", type=float, default=1.3,
                        help="stop distance in ATR, matching sl_atr_mult in .env")
    parser.add_argument("--horizon", type=int, default=12, help="barrier label horizon in bars")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--step", type=int, default=500)
    args = parser.parse_args(argv)

    run = _load_run()
    df = load_bars(args.csv)
    print(f"{len(df)} bars from {df.index[0]} to {df.index[-1]}")

    for mode in ("close", "barrier"):
        results = walk_forward(run, df, mode, args.horizon, args.sl_mult,
                               window=args.window, step=args.step)
        summary = summarise(results)
        print(f"\n[{mode}] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
scripts/evaluate_strategy_features.py — do the library strategies carry information?

Two questions, on exported bars, with the last ``--holdout-days`` kept out:

1. Per strategy: independent (horizon-thinned) calls, edge under the +/-1 ATR
   first-barrier rule, the same edge net of the CSV's ``spread`` column, and its
   z — on the training part and on the holdout. Plus a noise check: across all
   strategies the train z should average ~0 with sd ~1 if nothing is real, and
   the train/holdout z correlation says whether rankings persist.
2. Model inputs: a classifier trained on the base NN features, then on the base
   features plus the strategy block (compact, and optionally per strategy), on the
   SAME walk-forward folds and the same holdout. Reported: accuracy, AUC and lift
   over the majority-class baseline. If the strategy block does not raise the
   out-of-sample numbers, leave AI_STRATEGY_FEATURES / NN_STRATEGY_FEATURES off.

    .venv\\Scripts\\python.exe scripts\\evaluate_strategy_features.py --csv data\\history\\XAUUSDm_M15.csv

Spread and commission: the spread column (points) is converted with ``--point``
(XAUUSDm: 0.001) and expressed in ATR; commission is not included.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ai.nn_features import build_features  # noqa: E402
from core.strategy.analog import _atr  # noqa: E402
from core.strategy.features import COMPACT_COLUMNS, strategy_feature_matrix  # noqa: E402
from core.strategy.panel import strategy_category  # noqa: E402
from core.strategy.scalp import ScalpConfig, ScalpEngine, required_z, thin  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in ("open", "high", "low", "close", "tick_volume", "spread") if c in df.columns]
    return df[keep].astype(float)


def strategy_stats(sig, outcomes, cost, lo, hi, horizon) -> dict:
    idx = thin([t for t in range(lo, hi) if sig[t] != 0], horizon)
    if len(idx) < 5:
        return {"n": len(idx), "edge": None, "net": None, "z": None}
    r = np.array([0.0 if np.isnan(outcomes[t]) else (1.0 if (sig[t] == 1) == (outcomes[t] == 1.0) else -1.0)
                  for t in idx])
    m = float(r.mean())
    sd = float(r.std(ddof=1)) or 1.0
    c = np.array([cost[t] for t in idx], dtype=float)
    net = m - float(np.nanmean(c)) if np.isfinite(c).any() else None
    return {"n": len(idx), "edge": m, "net": net, "z": m / sd * math.sqrt(len(idx))}


def per_strategy(df, cfg, split, point) -> tuple:
    eng = ScalpEngine(cfg)
    sigs = eng.signals(df)
    outcomes = eng.outcomes(df)
    atr = _atr(df)
    atr = np.asarray(atr, dtype=float)
    cost = (df["spread"].to_numpy() * point / atr) if "spread" in df.columns else np.full(len(df), np.nan)
    n, h = len(df), cfg.horizon
    rows = []
    for name, s in sigs.items():
        tr = strategy_stats(s, outcomes, cost, 300, split, h)
        ho = strategy_stats(s, outcomes, cost, split, n - h, h)
        rows.append({"strategy": name, "category": strategy_category(name),
                     **{f"train_{k}": v for k, v in tr.items()},
                     **{f"hold_{k}": v for k, v in ho.items()}})
    return pd.DataFrame(rows), outcomes, float(np.nanmedian(cost))


def _fit_eval(X_tr, y_tr, X_te, y_te, model: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "logit":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             l2_regularization=1.0, random_state=0)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    acc = float(((p >= 0.5).astype(int) == y_te).mean())
    base = float(max(y_te.mean(), 1 - y_te.mean()))
    try:
        auc = float(roc_auc_score(y_te, p))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "lift": acc - base, "n": int(len(y_te))}


def model_comparison(df, outcomes, split, horizon, folds, model, per_strategy_cols) -> pd.DataFrame:
    base = build_features(df[["open", "high", "low", "close"] + (["tick_volume"] if "tick_volume" in df else [])])
    strat = strategy_feature_matrix(df, per_strategy=per_strategy_cols)
    sets: Dict[str, pd.DataFrame] = {
        "base": base,
        "base+strategy(compact)": pd.concat([base, strat[list(COMPACT_COLUMNS)]], axis=1),
    }
    if per_strategy_cols:
        sets["base+strategy(full)"] = pd.concat([base, strat], axis=1)
    y = pd.Series(outcomes, index=df.index)        # 1 = up barrier first, 0 = down, NaN = timeout
    ok = y.notna() & base.notna().all(axis=1)
    ok.iloc[-horizon:] = False
    pos = np.flatnonzero(ok.to_numpy())
    train_pos = pos[pos < split - horizon]
    hold_pos = pos[pos >= split]
    # expanding walk-forward folds inside the training part, gap = horizon
    edges = np.linspace(0, len(train_pos), folds + 2).astype(int)
    rows = []
    for name, X in sets.items():
        Xv = X.to_numpy(dtype=float)
        yv = y.to_numpy()
        for f in range(1, folds + 1):
            tr = train_pos[:edges[f]]
            te = train_pos[edges[f]:edges[f + 1]]
            if len(te) == 0 or len(tr) < 500:
                continue
            tr = tr[tr < te[0] - horizon]
            res = _fit_eval(Xv[tr], yv[tr].astype(int), Xv[te], yv[te].astype(int), model)
            rows.append({"features": name, "fold": f"wf{f}", **res})
        res = _fit_eval(Xv[train_pos], yv[train_pos].astype(int), Xv[hold_pos], yv[hold_pos].astype(int), model)
        rows.append({"features": name, "fold": "holdout", **res})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--holdout-days", type=float, default=91.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--point", type=float, default=0.001, help="price of one spread point (XAUUSDm 0.001)")
    ap.add_argument("--utc-offset", type=float, default=0.0)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--model", choices=("hgb", "logit"), default="hgb")
    ap.add_argument("--full", action="store_true", help="also test one column per strategy")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None, help="write the per-strategy table as CSV")
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args(argv)

    df = load(args.csv)
    split = int(df.index.searchsorted(df.index[-1] - pd.Timedelta(days=args.holdout_days)))
    if split < 2000 or len(df) - split < 500:
        print(f"not enough data: {len(df)} bars, split at {split}")
        return 1
    cfg = ScalpConfig(strategies=tuple(s.strip() for s in args.strategies.split(",") if s.strip()),
                      horizon=args.horizon, server_utc_offset=args.utc_offset)
    table, outcomes, med_cost = per_strategy(df, cfg, split, args.point)
    zreq = required_z(len(table), 2.7)
    print(f"{len(df)} bars {df.index[0]} -> {df.index[-1]}; holdout from {df.index[split]} "
          f"({len(df) - split} bars); median spread {med_cost:.3f} ATR; {len(table)} strategies, "
          f"Bonferroni bar {zreq:.2f} sigma")
    t = table.sort_values("train_z", ascending=False, na_position="last")
    f = lambda x: "   n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"
    print(f"\n{'strategy':24}{'n':>5}{'edge':>8}{'net':>8}{'z':>8}  ||{'n':>5}{'edge':>8}{'net':>8}{'z':>8}")
    for _, r in t.head(args.top).iterrows():
        print(f"{r.strategy:24}{int(r.train_n):5}{f(r.train_edge)}{f(r.train_net)}{f(r.train_z):>8}  ||"
              f"{int(r.hold_n):5}{f(r.hold_edge)}{f(r.hold_net)}{f(r.hold_z):>8}")
    z = t["train_z"].dropna()
    both = t.dropna(subset=["train_z", "hold_z"])
    corr = float(np.corrcoef(both.train_z, both.hold_z)[0, 1]) if len(both) > 3 else float("nan")
    passed = list(t.loc[t.train_z >= zreq, "strategy"])
    held = list(t.loc[(t.train_net > 0) & (t.hold_net > 0) & (t.train_z >= 1.0), "strategy"])
    print(f"\nnoise check: train z mean {z.mean():+.2f}, sd {z.std():.2f} over {len(z)} (pure noise ~ 0 / 1); "
          f"corr(train z, holdout z) {corr:+.2f} over {len(both)}")
    print(f"pass the Bonferroni bar on train: {', '.join(passed) or 'none'}")
    print(f"train z >= 1 and net > 0 in BOTH periods (watch list, not proof): {', '.join(held) or 'none'}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"per-strategy table written to {args.out}")

    if not args.no_models:
        res = model_comparison(df, outcomes, split, args.horizon, args.folds, args.model, args.full)
        print(f"\nmodel comparison ({args.model}, label = which +/-1 ATR barrier is hit first in "
              f"{args.horizon} bars, timeouts dropped):")
        print(f"{'features':26}{'fold':>9}{'n':>7}{'acc':>8}{'auc':>8}{'lift':>8}")
        for _, r in res.iterrows():
            print(f"{r.features:26}{r.fold:>9}{r.n:7}{r.acc:8.3f}{r.auc:8.3f}{r.lift:+8.3f}")
        summ = res.groupby("features")[["auc", "lift"]].mean()
        print("\nmean over walk-forward folds + holdout:")
        print(summ.round(3).to_string())
        b = summ.loc["base"]
        for name in summ.index:
            if name == "base":
                continue
            d_auc, d_lift = summ.loc[name, "auc"] - b.auc, summ.loc[name, "lift"] - b.lift
            verdict = ("helps" if d_auc > 0.01 and d_lift > 0.005 else "no measurable gain")
            print(f"{name}: AUC {d_auc:+.3f}, lift {d_lift:+.3f} vs base -> {verdict}")
    print("\nSpread is included in 'net'; commission is not. Trust nothing here without a "
          "paper-trading period on top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

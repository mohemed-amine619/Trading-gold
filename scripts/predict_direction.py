"""
scripts/predict_direction.py — standalone directional bias for a symbol.

Runs entirely on exported price history (no broker, no MT5). Safe to run
off-market. Prints a compact report, or JSON with --json.

    .venv\\Scripts\\python.exe scripts\\predict_direction.py --csv data\\history\\XAUUSDm_M5.csv
    .venv/bin/python scripts/predict_direction.py --csv data/history/XAUUSDm_M5.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from core.prediction.direction import predict_direction


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df = df.set_index("time")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def render(pred) -> str:
    lines = []
    head = f"{pred.symbol or '?'} {pred.timeframe or ''}".strip()
    lines.append(f"Direction: {pred.label}  (score {pred.score}, confidence {pred.confidence})")
    if head:
        lines.append(f"Symbol: {head}")
    if pred.as_of:
        lines.append(f"As of: {pred.as_of}")
    lines.append(f"Location: {pred.location_bias} (range pos {pred.range_position})")
    lines.append(f"Structure: {pred.structure_bias}")
    if pred.reasons:
        lines.append("Reasons:")
        for r in pred.reasons:
            lines.append(f"  - {r}")
    if pred.invalidation:
        lines.append(f"Invalidation: {pred.invalidation}")
    if pred.nn is not None:
        lines.append(f"NN: {pred.nn.get('status')} -> "
                    f"{_label(pred.nn.get('direction', 0))} "
                    f"(conf {pred.nn.get('confidence')})")
        if pred.consensus:
            lines.append(f"Consensus: {pred.consensus}")
    lines.append("Note: analysis only, not a trading signal.")
    return "\n".join(lines)


_LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}


def _label(d) -> str:
    try:
        return _LABELS.get(int(d), "NEUTRAL")
    except (TypeError, ValueError):
        return "NEUTRAL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone directional bias from price history")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV (with 'time' column)")
    parser.add_argument("--symbol", default=None, help="Symbol label (display only)")
    parser.add_argument("--timeframe", default=None, help="Timeframe label (display only)")
    parser.add_argument("--model", default=None, help="Optional NN artifact path")
    parser.add_argument("--min-oos-edge", type=float, default=0.0,
                        help="Minimum NN out-of-sample edge to trust the model")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    try:
        df = load_bars(Path(args.csv))
    except Exception as exc:
        print(f"failed to read {args.csv}: {exc!r}", file=sys.stderr)
        return 2

    pred = predict_direction(
        df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_path=args.model,
        min_oos_edge=args.min_oos_edge,
    )

    if args.json:
        print(json.dumps(asdict(pred), default=str, indent=2))
    else:
        print(render(pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

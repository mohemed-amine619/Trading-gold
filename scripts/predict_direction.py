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

from core.prediction.direction import predict_direction, predict_multi_timeframe


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df = df.set_index("time")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def render_panel(panel, top: int = 15) -> str:
    """Compact text view of a strategy panel dict (core/strategy/panel.py)."""
    if not panel:
        return ""
    lines = [f"Strategies {panel.get('timeframe')}: {panel.get('long_count')} long / "
             f"{panel.get('short_count')} short / {panel.get('flat_count')} flat, breadth "
             f"{panel.get('net_breadth', 0.0):+.2f} (z-weighted {panel.get('weighted_breadth', 0.0):+.2f}); "
             f"proof bar {panel.get('required_z', 0.0):.2f} sigma over {panel.get('n_tested')} strategies; "
             f"proven: {', '.join(panel.get('proven') or []) or 'none'}"]
    rows = [r for r in panel.get("readings", []) if r.get("available") and not r.get("error")]
    rows.sort(key=lambda r: -abs(r.get("z") or 0.0))
    if rows:
        lines.append(f"  {'strategy':24}{'cat':14}{'now':>6}{'calls':>7}{'edge':>8}{'z':>7}  proven")
        for r in rows[:top]:
            edge = "   n/a" if r.get("edge") is None else f"{r['edge']:+.2f}"
            z = "   n/a" if r.get("z") is None else f"{r['z']:+.2f}"
            now = {1: "BUY", -1: "SELL"}.get(r.get("signal"), "-")
            lines.append(f"  {r['name']:24}{r['category']:14}{now:>6}{r.get('calls', 0):>7}{edge:>8}{z:>7}"
                         f"  {'yes' if r.get('proven') else 'no'}")
    cats = panel.get("by_category") or {}
    if cats:
        lines.append("  by category: " + ", ".join(
            f"{c} {v['long']}L/{v['short']}S" for c, v in sorted(cats.items())))
    errs = [r["name"] for r in panel.get("readings", []) if r.get("error")]
    if errs:
        lines.append("  failed (skipped): " + ", ".join(errs))
    lines.append("  " + str(panel.get("note", "")))
    return "\n".join(lines)


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
    if getattr(pred, "strategy_panel", None):
        lines.append(render_panel(pred.strategy_panel))
    lines.append("Note: analysis only, not a trading signal.")
    return "\n".join(lines)


_LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}


def _label(d) -> str:
    try:
        return _LABELS.get(int(d), "NEUTRAL")
    except (TypeError, ValueError):
        return "NEUTRAL"



def render_multi(pred) -> str:
    lines = []
    lines.append(f"Multi-timeframe consensus: {pred.consensus_label} "
                f"(alignment {pred.alignment}/{len(pred.timeframes)})")
    if pred.symbol:
        lines.append(f"Symbol: {pred.symbol}")
    if pred.higher_timeframe:
        lines.append(f"Higher timeframe: {pred.higher_timeframe}")
    lines.append("")
    for tf, p in pred.timeframes.items():
        lines.append(f"  {tf}: {p.label} (score {p.score}, loc {p.location_bias})")
    for tf, p in pred.timeframes.items():
        if getattr(p, "strategy_panel", None):
            lines.append(render_panel(p.strategy_panel))
    if pred.conflicting:
        lines.append("Conflicting: " + ", ".join(pred.conflicting))
    if pred.reasons:
        lines.append("Reasons:")
        for r in pred.reasons:
            lines.append(f"  - {r}")
    lines.append("Note: analysis only, not a trading signal.")
    return "\n".join(lines)


def _strategies_arg(value):
    """``--strategies`` -> what predict_direction's ``include_strategies`` takes."""
    names = tuple(s.strip() for s in str(value or "").split(",") if s.strip())
    if not names or names == ("none",):
        return False
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone directional bias from price history")
    parser.add_argument("--csv", required=False, help="Path to OHLCV CSV (with 'time' column)")
    parser.add_argument("--tf", action="append", default=None,
                        help="Timeframe frame as NAME=PATH (repeatable), e.g. --tf H1=a.csv --tf H4=b.csv")
    parser.add_argument("--symbol", default=None, help="Symbol label (display only)")
    parser.add_argument("--timeframe", default=None, help="Timeframe label (display only)")
    parser.add_argument("--model", default=None, help="Optional NN artifact path")
    parser.add_argument("--min-oos-edge", type=float, default=0.0,
                        help="Minimum NN out-of-sample edge to trust the model")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--strategies", default="all",
                        help="strategy panel: all, classic, a group (smc, ict, ...), comma-separated "
                             "names, or none")
    args = parser.parse_args()
    strategies = _strategies_arg(args.strategies)

    # Multi-timeframe mode
    if args.tf:
        frames = {}
        for spec in args.tf:
            if "=" not in spec:
                print(f"bad --tf spec (expected NAME=PATH): {spec}", file=sys.stderr)
                return 2
            name, path = spec.split("=", 1)
            try:
                frames[name] = load_bars(Path(path))
            except Exception as exc:
                print(f"failed to read {path}: {exc!r}", file=sys.stderr)
                return 2
        pred = predict_multi_timeframe(
            frames,
            symbol=args.symbol,
            model_path=args.model,
            higher_timeframe=args.timeframe,
            min_oos_edge=args.min_oos_edge,
            include_strategies=strategies,
        )
        if args.json:
            print(json.dumps(pred.to_dict(), default=str, indent=2))
        else:
            print(render_multi(pred))
        return 0

    if not args.csv:
        print("either --csv or --tf is required", file=sys.stderr)
        return 2
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
        include_strategies=strategies,
    )

    if args.json:
        print(json.dumps(asdict(pred), default=str, indent=2))
    else:
        print(render(pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

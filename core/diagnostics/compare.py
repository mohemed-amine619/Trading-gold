"""
core/diagnostics/compare.py — A/B comparison of bot variants, by magic number.

Two instances of the bot with different settings and different MAGIC_NUMBERs trade the
same account; this answers "is B really better than A, or is that luck?". Per-trade P&L
is compared with Welch's t statistic (unequal variances). With the handful of trades a
demo account produces the answer is usually "not distinguishable yet", and the report
says so instead of crowning a winner: |z| < 2 is noise.

Pure functions over ``Trade`` objects. Ratios are ``None`` when undefined, never 0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from core.diagnostics.trades import Trade

MIN_TRADES = 10          # below this a per-trade mean says almost nothing


def _moments(pnls: List[float]) -> dict:
    """Rounded figures for display, plus the exact mean/se under ``_mean``/``_se`` so the
    z-score is not computed from already-rounded numbers."""
    n = len(pnls)
    if n == 0:
        return {"trades": 0, "net": 0.0, "expectancy": None, "se": None, "win_rate": None,
                "_mean": None, "_se": None}
    mean = sum(pnls) / n
    se = None
    if n >= 2:
        var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        se = math.sqrt(var / n)
    return {"trades": n, "net": round(sum(pnls), 2), "expectancy": round(mean, 3),
            "se": round(se, 3) if se is not None else None,
            "win_rate": round(sum(p > 0 for p in pnls) / n, 3), "_mean": mean, "_se": se}


def split_by_magic(trades: List[Trade], magics: List[int]) -> Dict[int, List[Trade]]:
    return {m: [t for t in trades if t.magic == m] for m in magics}


def compare_variants(groups: Dict[str, List[Trade]]) -> dict:
    """``groups`` maps a label to its trades; the FIRST label is the baseline.
    Returns {"variants": {label: moments + diff vs baseline}, "baseline": label, "verdict": str}."""
    labels = list(groups)
    if len(labels) < 2:
        raise ValueError("need at least two variants to compare")
    pnls = {k: [t.net_pnl for t in v] for k, v in groups.items()}
    stats = {k: _moments(v) for k, v in pnls.items()}
    base = labels[0]
    for k in labels[1:]:
        a, b = stats[base], stats[k]
        z: Optional[float] = None
        diff: Optional[float] = None
        if a["_se"] is not None and b["_se"] is not None:
            diff = round(b["_mean"] - a["_mean"], 3)
            denom = math.sqrt(a["_se"] ** 2 + b["_se"] ** 2)
            z = round((b["_mean"] - a["_mean"]) / denom, 2) if denom > 0 else None
        stats[k]["diff_vs_baseline"], stats[k]["z"] = diff, z
    for s in stats.values():
        s.pop("_mean"), s.pop("_se")
    small = [k for k in labels if stats[k]["trades"] < MIN_TRADES]
    best = max((k for k in labels[1:] if stats[k].get("z") is not None),
               key=lambda k: abs(stats[k]["z"]), default=None)
    if small:
        verdict = (f"Too few trades to tell (fewer than {MIN_TRADES}): {', '.join(small)}. "
                   "Keep both running; do not pick a winner yet.")
    elif best is None:
        verdict = "No comparable difference could be computed."
    elif abs(stats[best]["z"]) < 2.0:
        verdict = (f"Not distinguishable from chance (|z|={abs(stats[best]['z'])} < 2). "
                   "More trades are needed before choosing a variant.")
    else:
        direction = "better" if stats[best]["z"] > 0 else "worse"
        verdict = (f"{best} is {direction} than {base} per trade (z={stats[best]['z']}). Two variants "
                   "on one account share market conditions, but this is still one sample: confirm "
                   "on fresh trades.")
    return {"baseline": base, "variants": stats, "verdict": verdict}


def render(result: dict) -> str:
    rows = ["| variant | trades | net | win rate | per trade | vs baseline | z |", "|---|---|---|---|---|---|---|"]
    for k, s in result["variants"].items():
        rows.append(f"| {k} | {s['trades']} | {s['net']} | {s['win_rate']} | {s['expectancy']} | "
                    f"{s.get('diff_vs_baseline', '')} | {s.get('z', '')} |")
    return "\n".join(rows) + f"\n\n{result['verdict']}\n"

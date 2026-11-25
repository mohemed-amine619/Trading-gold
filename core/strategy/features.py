"""
core/strategy/features.py — the strategy library as model inputs.

Every generator is causal (``sig[t]`` uses bars ``<= t``), so the matrix built here
can be appended to a model's feature frame without look-ahead. Whether the
signals carry any information is NOT assumed: the AI / NN promotion gates measure
it out-of-sample, and the switches (``AI_STRATEGY_FEATURES``,
``NN_STRATEGY_FEATURES``) are off by default.

Two shapes:

* compact (default for the live AI): ``strat_long_n``, ``strat_short_n``,
  ``strat_breadth`` and one ``cat_<category>_net`` per category that has strategies — 13 columns, cheap
  to learn from a few thousand rows;
* full: the compact block plus one int8 column ``strat_<name>`` per strategy
  (118 for ``all``), and optionally ``strat_<name>_age`` (bars since the last
  call, capped at 48).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.strategy.library import CATEGORIES
from core.strategy.panel import strategy_category
from core.strategy.scalp import GENERATORS, ScalpConfig
from core.strategy.library import resolve

__all__ = ["CATEGORY_COLUMNS", "COMPACT_COLUMNS", "strategy_feature_matrix", "add_strategy_features"]

AGE_CAP = 48
_USED = {strategy_category(n) for n in resolve(("all",))}
CATEGORY_COLUMNS: Tuple[str, ...] = tuple(f"cat_{c}_net" for c in ("classic",) + CATEGORIES if c in _USED)
COMPACT_COLUMNS: Tuple[str, ...] = ("strat_long_n", "strat_short_n", "strat_breadth") + CATEGORY_COLUMNS

_CACHE: Dict[str, tuple] = {}


def _bars_since(sig: np.ndarray, cap: int = AGE_CAP) -> np.ndarray:
    n = len(sig)
    idx = np.where(sig != 0, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.where(last >= 0, np.arange(n) - last, cap).astype(float)
    return np.minimum(age, cap)


def _signals(df: pd.DataFrame, cfg: ScalpConfig, names: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    from core.strategy.analog import _atr
    atr = _atr(df)
    out: Dict[str, np.ndarray] = {}
    for name in names:
        try:
            out[name] = np.asarray(GENERATORS[name](df, atr, cfg), dtype=np.int8)
        except Exception:
            out[name] = np.zeros(len(df), dtype=np.int8)
    return out


def strategy_feature_matrix(df: pd.DataFrame, cfg: Optional[ScalpConfig] = None,
                            names: Optional[Iterable[str]] = None, per_strategy: bool = True,
                            ages: bool = False) -> pd.DataFrame:
    """Strategy features aligned to ``df.index``. No NaN: a strategy that has not
    called yet reads 0 (and age ``AGE_CAP``)."""
    cfg = cfg or ScalpConfig()
    names = resolve(tuple(names) if names is not None else ("all",))
    key = (len(df), df.index[0] if len(df) else None, df.index[-1] if len(df) else None,
           hash(df[["open", "high", "low", "close"]].to_numpy(dtype=float).tobytes()) if len(df) else 0,
           names, per_strategy, ages, cfg)
    hit = _CACHE.get("last")
    if hit is not None and hit[0] == key:
        return hit[1].copy()

    sigs = _signals(df, cfg, names) if len(df) else {n: np.zeros(0, np.int8) for n in names}
    n = len(df)
    long_n = np.zeros(n); short_n = np.zeros(n)
    cat_net = {c: np.zeros(n) for c in CATEGORY_COLUMNS}
    for name, s in sigs.items():
        long_n += s > 0
        short_n += s < 0
        col = f"cat_{strategy_category(name)}_net"
        if col in cat_net:
            cat_net[col] += s
    cols: Dict[str, np.ndarray] = {
        "strat_long_n": long_n, "strat_short_n": short_n,
        "strat_breadth": (long_n - short_n) / np.maximum(1.0, long_n + short_n),
    }
    cols.update(cat_net)
    if per_strategy:
        for name, s in sigs.items():
            cols[f"strat_{name}"] = s.astype(np.int8)
            if ages:
                cols[f"strat_{name}_age"] = _bars_since(s)
    out = pd.DataFrame(cols, index=df.index)
    _CACHE["last"] = (key, out)
    return out.copy()


def add_strategy_features(feat: pd.DataFrame, df: pd.DataFrame, cfg: Optional[ScalpConfig] = None,
                          per_strategy: bool = False) -> pd.DataFrame:
    """``feat`` with the strategy block joined on the index (compact by default)."""
    block = strategy_feature_matrix(df, cfg, per_strategy=per_strategy)
    block = block.reindex(feat.index).fillna(0.0)
    return pd.concat([feat, block], axis=1)


def feature_columns(per_strategy: bool = False, names: Optional[Iterable[str]] = None,
                    ages: bool = False) -> List[str]:
    cols = list(COMPACT_COLUMNS)
    if per_strategy:
        for name in resolve(tuple(names) if names is not None else ("all",)):
            cols.append(f"strat_{name}")
            if ages:
                cols.append(f"strat_{name}_age")
    return cols

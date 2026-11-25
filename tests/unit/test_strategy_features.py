"""
tests/unit/test_strategy_features.py — the strategy library as model inputs
(core/strategy/features.py, AI_STRATEGY_FEATURES, NN arch gru_strat).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.strategy.features import (
    COMPACT_COLUMNS, add_strategy_features, feature_columns, strategy_feature_matrix,
)
from core.strategy.library import resolve
from tests.unit.test_strategy_library import market


@pytest.fixture(scope="module")
def df():
    return market(1500, seed=11)


def test_matrix_aligned_no_nan_int8(df):
    m = strategy_feature_matrix(df, ages=True)
    assert m.index.equals(df.index)
    assert not m.isna().any().any()
    names = resolve(("all",))
    for n in names:
        assert m[f"strat_{n}"].dtype == np.int8
        assert set(np.unique(m[f"strat_{n}"])) <= {-1, 0, 1}
        assert m[f"strat_{n}_age"].between(0, 48).all()
    assert list(m.columns) == feature_columns(per_strategy=True, ages=True)


def test_compact_block(df):
    m = strategy_feature_matrix(df, per_strategy=False)
    assert list(m.columns) == list(COMPACT_COLUMNS)
    assert m["strat_breadth"].between(-1, 1).all()
    full = strategy_feature_matrix(df)
    cat_cols = [c for c in COMPACT_COLUMNS if c.startswith("cat_")]
    per = [c for c in full.columns if c.startswith("strat_") and c not in COMPACT_COLUMNS]
    assert (full[per].clip(lower=0).sum(axis=1) == full["strat_long_n"]).all()
    assert (full[cat_cols].sum(axis=1) == full["strat_long_n"] - full["strat_short_n"]).all()


def test_causal(df):
    full = strategy_feature_matrix(df, per_strategy=False)
    for cut in (900, 1300):
        part = strategy_feature_matrix(df.iloc[:cut + 1], per_strategy=False)
        assert np.allclose(part.to_numpy(), full.iloc[:cut + 1].to_numpy()), cut


def test_add_strategy_features_joins(df):
    base = df[["close"]].rename(columns={"close": "x"})
    out = add_strategy_features(base, df)
    assert list(out.columns) == ["x"] + list(COMPACT_COLUMNS)


def test_nn_arch_parity_and_extension(df):
    from core.ai.deep_layout import ARCH_GRU, ARCH_GRU_STRAT, build_model_features, feature_names
    from core.ai.nn_features import FEATURE_NAMES
    base = build_model_features(df, ARCH_GRU)
    assert feature_names(ARCH_GRU) == list(FEATURE_NAMES)          # flag off: unchanged
    ext = build_model_features(df, ARCH_GRU_STRAT)
    assert feature_names(ARCH_GRU_STRAT) == list(FEATURE_NAMES) + list(COMPACT_COLUMNS)
    assert ext.shape == (len(df), base.shape[1] + len(COMPACT_COLUMNS))
    assert np.allclose(ext[:, :base.shape[1]], base, equal_nan=True)

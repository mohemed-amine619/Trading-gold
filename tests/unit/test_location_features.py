"""Structural feature vector: shape, scale, honesty about missing data, causality."""

import numpy as np
import pandas as pd

from core.ai.location_features import (
    N_STRUCT, STRUCT_FEATURE_NAMES, build_location_features, latest_location_features,
    location_feature_vector,
)
from core.market.location import analyse_location


def _frame(n=420, seed=3, scale=1.0, base=2000.0):
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, scale, n))
    idx = pd.date_range("2025-03-03", periods=n, freq="5min")
    return pd.DataFrame({"open": close, "high": close + rng.random(n) * scale,
                         "low": close - rng.random(n) * scale, "close": close,
                         "tick_volume": 100.0}, index=idx)


def test_names_match_width():
    assert len(STRUCT_FEATURE_NAMES) == N_STRUCT == len(set(STRUCT_FEATURE_NAMES))


def test_vector_is_finite_and_bounded():
    df = _frame()
    v = location_feature_vector(analyse_location(df), len(df))
    assert v.shape == (N_STRUCT,) and np.isfinite(v).all()
    assert abs(v[STRUCT_FEATURE_NAMES.index("range_pos")]) <= 0.5
    for name in ("support_dist_atr", "resistance_dist_atr", "liq_above_dist_atr",
                 "liq_below_dist_atr"):
        assert 0.0 <= v[STRUCT_FEATURE_NAMES.index(name)] <= 5.0


def test_missing_thing_is_flagged_not_encoded_as_zero_distance():
    """Too little data: no range, no zones. Flags are 0 and distances sit at the cap."""
    v = latest_location_features(_frame(n=20))
    i = STRUCT_FEATURE_NAMES.index
    assert v[i("has_range")] == 0.0 and v[i("has_support")] == 0.0
    assert v[i("support_dist_atr")] == 5.0 and v[i("liq_above_dist_atr")] == 5.0


def test_scale_invariant_across_instruments():
    """The same price path at 1x and 50x price scale gives the same vector."""
    a = _frame(scale=1.0, base=2000.0)
    b = a.copy()
    b[["open", "high", "low", "close"]] = b[["open", "high", "low", "close"]] * 50.0
    va, vb = latest_location_features(a), latest_location_features(b)
    assert np.allclose(va, vb, atol=1e-6)


def test_build_is_causal_future_bars_do_not_change_past_rows():
    df = _frame(n=360)
    full = build_location_features(df, stride=8)
    cut = 250
    # Only the first `cut` bars exist; rows must be identical to the full run's.
    part = build_location_features(df.iloc[:cut], stride=8)
    a, b = full.iloc[:cut].to_numpy(), part.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_warmup_rows_are_nan_not_zero():
    f = build_location_features(_frame(n=120), stride=8)
    assert f.iloc[:30].isna().all().all()
    assert f.iloc[-1].notna().all()

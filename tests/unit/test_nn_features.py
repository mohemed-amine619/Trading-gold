"""
tests/unit/test_nn_features.py — leak-free features, labels, scaler and splits.

The leakage tests are the point: a feature at bar t must not change when future
bars change, the scaler must not see validation data, and a label must not be
invented for bars whose future is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ai.nn_features import (
    FEATURE_NAMES, LONG, NOISE, SHORT, UNKNOWN,
    FeatureScaler, build_features, latest_window, make_labels, make_windows,
    walk_forward_splits,
)


def random_walk(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(50, 500, n)}, index=idx)


def flat_market(n=100):
    """Constant 1.0 range around 100: ATR settles at exactly 1.0."""
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="5min")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "tick_volume": 100}, index=idx)


class TestFeatureCausality:
    def test_columns_and_index(self):
        f = build_features(random_walk())
        assert tuple(f.columns) == FEATURE_NAMES
        assert f.index.equals(random_walk().index)

    def test_a_row_does_not_change_when_later_bars_are_removed(self):
        df = random_walk()
        full = build_features(df)
        cut = build_features(df.iloc[:300])
        np.testing.assert_allclose(full.iloc[:300].to_numpy(), cut.to_numpy(),
                                   equal_nan=True, rtol=1e-9, atol=1e-12)

    def test_a_row_does_not_change_when_future_prices_are_altered(self):
        df = random_walk()
        base = build_features(df)
        changed = df.copy()
        changed.iloc[300:, changed.columns.get_loc("close")] *= 1.5
        changed.iloc[300:, changed.columns.get_loc("high")] *= 1.5
        after = build_features(changed)
        np.testing.assert_allclose(base.iloc[:300].to_numpy(), after.iloc[:300].to_numpy(),
                                   equal_nan=True, rtol=1e-9, atol=1e-12)

    def test_features_are_stationary_in_price_level(self):
        # Shifting the whole price level must not move scale-free features.
        df = random_walk()
        shifted = df.copy()
        for c in ("open", "high", "low", "close"):
            shifted[c] = shifted[c] + 1000.0
        a, b = build_features(df), build_features(shifted)
        for col in ("dist_ema21", "dist_ema50", "macd_hist", "body", "upper_wick",
                    "lower_wick", "range_pos", "rsi"):
            np.testing.assert_allclose(a[col].dropna().to_numpy(),
                                       b[col].dropna().to_numpy(), rtol=1e-6, atol=1e-6,
                                       err_msg=col)

    def test_warmup_rows_are_nan_and_later_rows_are_finite(self):
        f = build_features(random_walk())
        assert f.iloc[:5].isna().any(axis=1).all()
        assert np.isfinite(f.iloc[-100:].to_numpy()).all()

    def test_session_encoding_is_cyclic(self):
        f = build_features(random_walk(n=400))
        assert (f["hour_sin"] ** 2 + f["hour_cos"] ** 2).round(9).eq(1.0).all()

    def test_missing_volume_column_is_tolerated(self):
        f = build_features(random_walk().drop(columns="tick_volume"))
        assert (f["vol_ratio"] == 0.0).all()

    def test_requires_ohlc_and_datetime_index(self):
        with pytest.raises(ValueError):
            build_features(random_walk().drop(columns="high"))
        with pytest.raises(ValueError):
            build_features(random_walk().reset_index(drop=True))


class TestLabels:
    def _market(self):
        df = flat_market()
        # i=20: +1.5 barrier is 101.5. Bar 21 pokes to 101.6 -> LONG.
        df.iloc[21, df.columns.get_loc("high")] = 101.6
        # i=40: barrier ~1.9 by now. Bar 41 drops to 97.0 -> SHORT.
        df.iloc[41, df.columns.get_loc("low")] = 97.0
        # i=80: bar 81 touches both barriers inside one candle -> ambiguous.
        df.iloc[81, df.columns.get_loc("high")] = 102.0
        df.iloc[81, df.columns.get_loc("low")] = 97.0
        return df

    def test_upper_barrier_first_is_long(self):
        assert make_labels(self._market(), 1.5, 5)[20] == LONG

    def test_lower_barrier_first_is_short(self):
        assert make_labels(self._market(), 1.5, 5)[40] == SHORT

    def test_no_barrier_touched_is_noise(self):
        assert make_labels(self._market(), 1.5, 5)[60] == NOISE

    def test_both_barriers_in_one_candle_is_noise_not_a_direction(self):
        assert make_labels(self._market(), 1.5, 5)[80] == NOISE

    def test_the_last_horizon_bars_are_unknown_not_noise(self):
        y = make_labels(self._market(), 1.5, 5)
        assert (y[-5:] == UNKNOWN).all()
        assert y[-6] != UNKNOWN

    def test_bars_before_atr_exists_are_unknown(self):
        y = make_labels(self._market(), 1.5, 5)
        assert (y[:13] == UNKNOWN).all()


class TestScaler:
    def test_statistics_come_from_the_training_rows_only(self):
        rng = np.random.default_rng(0)
        train = rng.normal(5.0, 2.0, (500, 3))
        val = rng.normal(500.0, 90.0, (200, 3))        # a very different regime
        s = FeatureScaler().fit(train)
        np.testing.assert_allclose(s.mean, train.mean(axis=0))
        s.transform(val)                                # transforming must not refit
        np.testing.assert_allclose(s.mean, train.mean(axis=0))
        np.testing.assert_allclose(s.std, train.std(axis=0))

    def test_output_is_clipped(self):
        s = FeatureScaler(clip=5.0).fit(np.random.default_rng(0).normal(0, 1, (200, 2)))
        assert np.abs(s.transform(np.array([[1e6, -1e6]]))).max() == 5.0

    def test_constant_column_does_not_divide_by_zero(self):
        x = np.column_stack([np.ones(50), np.arange(50.0)])
        out = FeatureScaler().fit(x).transform(x)
        assert np.isfinite(out).all()

    def test_round_trip_through_dict(self):
        x = np.random.default_rng(0).normal(0, 1, (100, 3))
        s = FeatureScaler().fit(x)
        s2 = FeatureScaler.from_dict(s.to_dict())
        np.testing.assert_allclose(s.transform(x), s2.transform(x))

    def test_unfitted_scaler_refuses(self):
        with pytest.raises(RuntimeError):
            FeatureScaler().transform(np.zeros((2, 2)))


class TestWindows:
    def test_shape_and_alignment(self):
        x = np.arange(40.0).reshape(20, 2)
        y = np.arange(20) % 3
        X, Y, ends = make_windows(x, y, window=4)
        assert X.shape == (17, 4, 2)
        np.testing.assert_array_equal(X[0], x[0:4])
        assert Y[0] == y[3] and ends[0] == 3

    def test_unknown_labels_and_nan_windows_are_dropped(self):
        x = np.ones((20, 2))
        x[5, 0] = np.nan
        y = np.zeros(20, dtype=int)
        y[-3:] = UNKNOWN
        _, _, ends = make_windows(x, y, window=4)
        assert 17 not in ends and 19 not in ends             # unknown labels
        assert not any(5 <= e <= 8 for e in ends)            # window touches the NaN

    def test_latest_window(self):
        x = np.arange(20.0).reshape(10, 2)
        np.testing.assert_array_equal(latest_window(x, 3), x[-3:])
        assert latest_window(x, 11) is None
        x[-1, 0] = np.nan
        assert latest_window(x, 3) is None


class TestWalkForward:
    def test_folds_never_overlap_and_respect_the_gap(self):
        folds = list(walk_forward_splits(n=1000, train=400, test=100, gap=24))
        assert folds
        for tr, te in folds:
            assert te.start - tr.stop == 24
            assert tr.stop <= te.start
            assert te.stop <= 1000

    def test_fold_count_and_rolling_origin(self):
        folds = list(walk_forward_splits(n=1000, train=400, test=100, gap=24))
        assert len(folds) == 5                       # starts 0,100,...,400 (524+100<=1000)
        assert [f[0].start for f in folds] == [0, 100, 200, 300, 400]

    def test_not_enough_data_yields_nothing(self):
        assert list(walk_forward_splits(n=100, train=80, test=30, gap=5)) == []

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            list(walk_forward_splits(100, 0, 10, 0))

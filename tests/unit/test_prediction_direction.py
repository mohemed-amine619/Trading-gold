import numpy as np
import pandas as pd
import pytest

from core.prediction.direction import DirectionPrediction, predict_direction


def _trending_df(n: int = 260, slope: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 2000.0 + np.arange(n) * slope
    noise = rng.normal(0, 0.5, n)
    close = base + noise
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    op = close - rng.uniform(-0.5, 0.5, n)
    vol = rng.integers(100, 1000, n)
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close, "tick_volume": vol},
        index=idx,
    )


def test_empty_df_is_neutral():
    pred = predict_direction(pd.DataFrame())
    assert pred.direction == 0
    assert pred.label == "NEUTRAL"
    assert pred.reasons


def test_none_is_neutral():
    pred = predict_direction(None)  # type: ignore[arg-type]
    assert pred.direction == 0
    assert pred.label == "NEUTRAL"


def test_missing_close_is_neutral():
    df = pd.DataFrame({"open": [1.0, 2.0], "high": [2.0, 3.0]})
    pred = predict_direction(df)
    assert pred.direction == 0
    assert any("close" in r or "usable" in r for r in pred.reasons)


def test_valid_series_returns_well_formed_prediction():
    df = _trending_df()
    pred = predict_direction(df, symbol="XAUUSDm", timeframe="H1")
    assert isinstance(pred, DirectionPrediction)
    assert pred.direction in (-1, 0, 1)
    assert pred.label in ("LONG", "SHORT", "NEUTRAL")
    assert 0 <= pred.score <= 100
    assert 0.0 <= pred.confidence <= 1.0
    assert isinstance(pred.location_bias, str)
    assert pred.as_of is not None
    assert pred.symbol == "XAUUSDm"
    assert pred.timeframe == "H1"


def test_missing_model_reports_not_loaded_and_does_not_flip_direction():
    df = _trending_df()
    base = predict_direction(df)
    with_model = predict_direction(df, model_path="/tmp/does_not_exist_model.pt")
    assert with_model.direction == base.direction
    assert with_model.nn is not None
    assert with_model.nn.get("status") != "ready"


def test_to_dict_json_serializable():
    import json
    df = _trending_df()
    pred = predict_direction(df)
    payload = json.dumps(pred.to_dict(), default=str)
    assert isinstance(payload, str)
    assert "direction" in payload


def test_location_bias_never_empty():
    df = _trending_df()
    pred = predict_direction(df)
    assert pred.location_bias != ""

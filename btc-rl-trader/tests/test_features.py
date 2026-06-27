import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features import (
    compute_features,
    rolling_higuchi_fd,
    rolling_quantile,
    rolling_trend_slope,
    rolling_trend_curvature,
    FEATURE_NAMES,
)


def _make_dummy_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(0)
    close = 30000 * np.exp(np.random.randn(n).cumsum() * 0.002)
    volume = np.abs(np.random.randn(n)) * 1000 + 500
    taker_buy = volume * np.random.uniform(0.3, 0.7, n)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "Gmt time": timestamps,
        "open": close * 0.999,
        "close": close,
        "volume": volume,
        "taker_buy_volume": taker_buy,
    })


def test_compute_features_columns():
    df = _make_dummy_df()
    out = compute_features(df)
    for col in FEATURE_NAMES:
        assert col in out.columns, f"Missing column: {col}"


def test_compute_features_no_inf():
    df = _make_dummy_df()
    out = compute_features(df)
    arr = out[FEATURE_NAMES].to_numpy(dtype=float)
    assert not np.any(np.isinf(arr))


def test_rolling_higuchi_shape():
    s = pd.Series(np.random.randn(200))
    result = rolling_higuchi_fd(s, window=50, kmax=10, min_periods=20)
    assert len(result) == len(s)


def test_rolling_higuchi_range():
    s = pd.Series(np.cumsum(np.random.randn(200)))
    result = rolling_higuchi_fd(s, window=50, kmax=10, min_periods=20)
    valid = result.dropna()
    assert (valid >= 1.0).all() and (valid <= 2.0).all()


def test_rolling_quantile_range():
    s = pd.Series(np.random.randn(200))
    result = rolling_quantile(s, window=100, min_periods=10)
    valid = result.dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_rolling_trend_slope_shape():
    s = pd.Series(np.linspace(0, 10, 200))
    result = rolling_trend_slope(s, window=20)
    assert len(result) == 200
    assert not np.any(np.isnan(result[20:]))


def test_rolling_trend_curvature_flat():
    s = pd.Series(np.linspace(0, 10, 200))
    result = rolling_trend_curvature(s, window=20)
    valid = result.iloc[20:].dropna()
    assert np.allclose(valid.values, 0.0, atol=1e-6)


def test_feature_determinism():
    df = _make_dummy_df()
    a = compute_features(df.copy())[FEATURE_NAMES].to_numpy()
    b = compute_features(df.copy())[FEATURE_NAMES].to_numpy()
    np.testing.assert_array_equal(a, b)

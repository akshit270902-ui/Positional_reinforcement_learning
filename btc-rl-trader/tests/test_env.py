import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.env import TradingEnv


def _make_dummy_df(n: int = 1200) -> pd.DataFrame:
    np.random.seed(1)
    close = 30000 * np.exp(np.random.randn(n).cumsum() * 0.001)
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


def test_env_reset_obs_shape():
    df = _make_dummy_df()
    env = TradingEnv(df)
    obs, info = env.reset()
    assert "features" in obs
    assert obs["features"].shape == (env.num_features,)


def test_env_step_shapes():
    df = _make_dummy_df()
    env = TradingEnv(df)
    env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert "features" in obs
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_env_no_nan_obs():
    df = _make_dummy_df()
    env = TradingEnv(df)
    env.reset()
    for _ in range(50):
        obs, _, done, _, _ = env.step(np.random.randint(0, 3))
        assert not np.any(np.isnan(obs["features"]))
        if done:
            break


def test_env_long_trade_profit():
    df = _make_dummy_df()
    env = TradingEnv(df)
    env.reset()
    env.current_step = 1001

    env.position = 0
    env.step(1)
    entry = env.entry_price

    for _ in range(5):
        env.step(1)

    env.df.iloc[env.current_step, env.df.columns.get_loc('close')] = entry + 500.0
    obs, reward, done, _, info = env.step(0)
    if info.get("trade_executed"):
        assert info["profit_realized_raw"] > 0


def test_env_short_trade_loss():
    df = _make_dummy_df()
    env = TradingEnv(df)
    env.reset()
    env.current_step = 1001
    env.position = 0
    env.step(2)
    entry = env.entry_price

    env.df.iloc[env.current_step, env.df.columns.get_loc('close')] = entry + 200.0
    obs, reward, done, _, info = env.step(0)
    if info.get("trade_executed"):
        assert info["profit_realized_raw"] < 0


def test_env_action_space():
    df = _make_dummy_df()
    env = TradingEnv(df)
    assert env.action_space.n == 3

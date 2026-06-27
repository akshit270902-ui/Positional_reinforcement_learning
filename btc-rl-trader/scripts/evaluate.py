import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import pathlib
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

from config import (
    SEED, RAW_DATA_PATH, OUTPUT_DIR, TRAIN_SPLIT, EVAL_BUFFER_BARS,
)
from src.env import TradingEnv
from src.utils import (
    load_data, print_trade_stats, plot_pnl, plot_weekly_pnl_distribution,
    compute_max_drawdown, compute_sharpe,
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

OUTPUT_PATH = pathlib.Path(OUTPUT_DIR)
MODEL_PATH = OUTPUT_PATH / "final_model.zip"
VECNORM_PATH = OUTPUT_PATH / "vec_normalize.pkl"


def main():
    df = load_data(RAW_DATA_PATH)
    print(f"Loaded {len(df)} bars.")

    dummy_env = TradingEnv(df.copy())
    dummy_env.reset()
    full_df = dummy_env.df.copy()

    train_size = int(len(full_df) * TRAIN_SPLIT)
    df_test = full_df.iloc[train_size:].reset_index(drop=True)
    buffer_df = full_df.iloc[max(0, train_size - EVAL_BUFFER_BARS):train_size].reset_index(drop=True)
    segment_df = pd.concat([buffer_df, df_test]).reset_index(drop=True)
    buffer_len = len(buffer_df)

    print(f"Test bars: {len(df_test)}, buffer: {buffer_len}")

    trading_env = TradingEnv(segment_df.copy())
    monitored = Monitor(trading_env)
    vec_env = DummyVecEnv([lambda: monitored])

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VECNORM_PATH.exists():
        raise FileNotFoundError(f"VecNormalize not found: {VECNORM_PATH}")

    vec_env = VecNormalize.load(str(VECNORM_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = RecurrentPPO.load(str(MODEL_PATH), env=vec_env)

    try:
        internal_env = vec_env.envs[0].env
    except Exception:
        internal_env = vec_env.envs[0]

    internal_env.current_step = 0
    internal_env.position = 0
    internal_env.balance = internal_env.initial_balance
    internal_env.equity = internal_env.initial_balance
    internal_env.entry_price = 0.0
    internal_env.entry_step = None

    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    started = False
    baseline_price = None

    cumulative_prices = []
    cumulative_equities = []
    cumulative_buyhold = []
    trade_logs = []
    weekly_pnl_records = []

    current_week_key = None
    week_start_price = None
    week_start_equity = None

    ep_total_trades = 0
    ep_winning_trades = 0
    ep_losing_trades = 0
    ep_long_trades = 0
    ep_short_trades = 0

    for global_idx in range(len(segment_df)):
        pre_idx = int(min(max(0, internal_env.current_step), len(internal_env.df) - 1))
        try:
            price_pre = float(internal_env.df['close'].iloc[pre_idx])
            time_pre = pd.to_datetime(internal_env.df['Gmt time'].iloc[pre_idx]) \
                if 'Gmt time' in internal_env.df.columns else pd.Timestamp.now()
        except Exception:
            price_pre = cumulative_prices[-1] if cumulative_prices else 0.0
            time_pre = pd.Timestamp.now()

        equity_pre = float(getattr(internal_env, 'equity', internal_env.initial_balance))

        if not started:
            if global_idx < buffer_len:
                obs, _, dones, _ = vec_env.step(np.array([0]))
                if isinstance(obs, tuple):
                    obs = obs[0]
                episode_starts = np.array(dones if isinstance(dones, (list, np.ndarray)) else [dones], dtype=bool)
                continue
            else:
                started = True
                shifted = time_pre - pd.Timedelta(days=1)
                iso = shifted.isocalendar()
                current_week_key = int(iso[0]) * 100 + int(iso[1])
                week_start_price = price_pre
                week_start_equity = equity_pre

        if started:
            shifted = time_pre - pd.Timedelta(days=1)
            iso = shifted.isocalendar()
            this_week_key = int(iso[0]) * 100 + int(iso[1])
            if this_week_key != current_week_key:
                if current_week_key is not None and week_start_price:
                    week_pnl = (equity_pre - week_start_equity) / week_start_price
                    weekly_pnl_records.append(week_pnl)
                current_week_key = this_week_key
                week_start_price = price_pre
                week_start_equity = equity_pre

        with torch.no_grad():
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )

        obs, rewards, dones, infos = vec_env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_starts = np.array(
            dones if isinstance(dones, (list, np.ndarray)) else [dones], dtype=bool
        )

        info = infos[0] if isinstance(infos, (list, np.ndarray)) else infos

        if started:
            cumulative_prices.append(price_pre)
            cumulative_equities.append(equity_pre)

            if baseline_price is None:
                baseline_price = price_pre
            cumulative_buyhold.append(price_pre - baseline_price)

            if info and info.get("trade_executed", False) and pre_idx >= buffer_len:
                ep_total_trades += 1
                profit = float(info.get("profit_realized_after_costs", 0.0))
                closed_side = info.get("closed_side")
                holding_time = info.get("holding_time")
                trade_logs.append({
                    "step": pre_idx - buffer_len,
                    "timestamp": time_pre,
                    "profit_realized": profit,
                    "closed_side": closed_side,
                    "equity_before": equity_pre,
                    "holding_time": holding_time,
                })
                if closed_side == "long":
                    ep_long_trades += 1
                elif closed_side == "short":
                    ep_short_trades += 1
                if profit > 0:
                    ep_winning_trades += 1
                else:
                    ep_losing_trades += 1

        is_done = (
            (isinstance(dones, (list, np.ndarray)) and bool(dones[0]))
            or (isinstance(dones, (bool, np.bool_)) and bool(dones))
            or (started and getattr(internal_env, 'current_step', pre_idx) >= len(internal_env.df) - 1)
        )

        if is_done and started:
            if week_start_price and week_start_equity is not None and cumulative_equities:
                final_equity = cumulative_equities[-1]
                week_pnl = (final_equity - week_start_equity) / week_start_price
                weekly_pnl_records.append(week_pnl)
            break

    print_trade_stats(trade_logs)

    final_equity = cumulative_equities[-1] if cumulative_equities else internal_env.initial_balance
    print(f"\nFinal equity     : {final_equity:,.2f}")
    print(f"Max drawdown     : {compute_max_drawdown(cumulative_equities):.4f}")
    print(f"Sharpe (equity)  : {compute_sharpe(cumulative_equities):.4f}")
    print(f"Long trades      : {ep_long_trades}  |  Short trades: {ep_short_trades}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    timestamps = None
    if 'Gmt time' in segment_df.columns:
        start = buffer_len
        end = buffer_len + len(cumulative_equities)
        timestamps = segment_df['Gmt time'].iloc[start:end].reset_index(drop=True)

    plot_pnl(cumulative_equities, cumulative_buyhold, timestamps, OUTPUT_PATH)
    plot_weekly_pnl_distribution(weekly_pnl_records, OUTPUT_PATH)

    pd.DataFrame(trade_logs).to_csv(OUTPUT_PATH / "trade_log.csv", index=False)
    print(f"\nResults saved to {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()

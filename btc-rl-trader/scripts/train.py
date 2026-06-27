import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import pathlib
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO

from config import (
    SEED, N_ENVS, TOTAL_TIMESTEPS, N_STEPS, BATCH_SIZE, GAMMA, GAE_LAMBDA,
    CLIP_RANGE, ENT_COEF, N_EPOCHS, MAX_GRAD_NORM, VF_COEF, LR_INITIAL, LR_FINAL,
    RAW_DATA_PATH, OUTPUT_DIR, WARMUP_MONTHS, TRAIN_SPLIT,
)
from src.env import TradingEnv
from src.policy import get_policy_kwargs
from src.utils import load_data, linear_schedule
from src.callbacks import TrainingMonitorCallback, TrainingRewardLogger


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def main():
    df = load_data(RAW_DATA_PATH)
    print(f"Loaded {len(df)} bars.")

    dummy_env = TradingEnv(df.copy())
    dummy_env.reset()
    full_df = dummy_env.df.copy()

    train_size = int(len(full_df) * TRAIN_SPLIT)
    warmup_bars = int(WARMUP_MONTHS * 30.4375 * 24)
    slice_start = min(warmup_bars, train_size - 1)
    df_train = full_df.iloc[slice_start:train_size].reset_index(drop=True)

    print(f"Training on {len(df_train)} bars (after {warmup_bars}-bar warmup).")

    def make_env(seed, _df=df_train):
        def _init():
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            env = TradingEnv(_df.copy())
            env = Monitor(env)
            try:
                env.reset(seed=seed)
            except TypeError:
                env.reset()
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        vec_env,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        learning_rate=linear_schedule(LR_INITIAL, LR_FINAL),
        n_epochs=N_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=get_policy_kwargs(),
        vf_coef=VF_COEF,
        tensorboard_log=str(output_dir),
        normalize_advantage=True,
    )

    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    log_every = N_STEPS * 128
    callbacks = CallbackList([
        TrainingMonitorCallback(log_every_steps=log_every, verbose=1),
        TrainingRewardLogger(log_dir=output_dir, log_every_steps=log_every),
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Training exited: {e}")

    model.save(str(output_dir / "final_model.zip"))
    vec_env.save(str(output_dir / "vec_normalize.pkl"))
    print(f"Saved model and VecNormalize to {output_dir}/")

    rewards_path = output_dir / "train_episode_rewards.npy"
    if rewards_path.exists():
        rewards = np.load(rewards_path)
        smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed.values, label='Episode reward (rolling-50)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=150)
        plt.close()
        print("Saved training curve.")


if __name__ == "__main__":
    main()

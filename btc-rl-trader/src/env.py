import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.features import compute_features, FEATURE_NAMES
from config import INITIAL_BALANCE, COMMISSION_PER_TRADE


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_per_trade = COMMISSION_PER_TRADE

        self._calculate_all_features()

        self.n_positional_feats = 2
        self.num_features = self.base_features.shape[1] + self.n_positional_feats

        self.action_space = spaces.Discrete(3)
        self._action_map = {0: 0, 1: 1, 2: -1}

        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_features,), dtype=np.float32
            )
        })

        self.balance = initial_balance
        self.net_profit = 0.0
        self.equity = initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = None
        self.winning_profits = []
        self.losing_losses = []
        self.returns_history = []
        self.current_episode_equities = []
        self.episode_start_step = 0
        self.steps_in_episode = 0
        self.episode_count = 0
        self.episode_lengths = []
        self.current_step = 0
        self.np_random = None

    def _calculate_all_features(self):
        processed = compute_features(self.df)
        self.df = processed
        self.base_features = processed[FEATURE_NAMES].to_numpy(dtype=np.float32)

    def _get_obs(self):
        idx = int(min(max(0, self.current_step), len(self.df) - 1))
        feats = self.base_features[idx].astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        pos_val = float(self.position)
        position_return = 0.0
        try:
            current_price = float(self.df['close'].iloc[idx])
            if self.position != 0 and self.entry_price and float(self.entry_price) != 0.0:
                entry = float(self.entry_price)
                if self.position == 1:
                    position_return = (current_price - entry) / entry
                elif self.position == -1:
                    position_return = (entry - current_price) / entry
        except Exception:
            position_return = 0.0

        obs_vector = np.concatenate(
            [feats, np.array([pos_val, position_return], dtype=np.float32)]
        ).astype(np.float32)
        obs_vector = np.nan_to_num(obs_vector, nan=0.0, posinf=0.0, neginf=0.0)
        return {"features": obs_vector}

    def _get_info(self):
        return {
            "balance": self.balance,
            "net_profit": self.net_profit,
            "equity": self.equity,
            "current_step": self.current_step,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random if seed is None else np.random.default_rng(seed)

        min_start = 1000
        high = max(min_start + 1, len(self.df) - 1)
        if hasattr(rng, 'integers'):
            self.current_step = int(rng.integers(min_start, high))
        else:
            self.current_step = int(rng.randint(min_start, high))

        if self.steps_in_episode > 0:
            self.episode_lengths.append(self.steps_in_episode)
        self.episode_count += 1
        self.steps_in_episode = 0
        self.episode_start_step = self.current_step

        self.balance = self.initial_balance
        self.net_profit = 0.0
        self.equity = self.initial_balance
        self.entry_price = 0.0
        self.entry_step = None
        self.winning_profits = []
        self.losing_losses = []
        self.returns_history = []
        self.current_episode_equities = [self.initial_balance]
        self.position = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        idx = int(min(max(0, self.current_step), len(self.df) - 1))
        try:
            close_price = float(self.df['close'].iloc[idx])
        except Exception:
            close_price = float(self.df['close'].iloc[-1])

        action_int = int(np.asarray(action).flatten()[0]) if np.asarray(action).size else int(action)
        mapped_action = self._action_map.get(action_int, 0)

        pre_position = int(self.position)
        new_position = pre_position
        reward = 0.0
        trade_executed = False
        profit_realized_raw = 0.0
        profit_realized_after_costs = 0.0
        closed_side = None
        holding_time = None

        if pre_position == 0:
            if mapped_action in (1, -1):
                new_position = mapped_action
                self.entry_price = close_price
                self.entry_step = self.current_step

        elif pre_position == 1:
            if mapped_action in (0, -1):
                profit_realized_raw = close_price - float(self.entry_price)
                profit_realized_after_costs = profit_realized_raw - self.commission_per_trade
                closed_side = 'long'
                self.net_profit += profit_realized_after_costs
                self.balance += profit_realized_after_costs
                (self.winning_profits if profit_realized_after_costs > 0 else self.losing_losses).append(profit_realized_after_costs)
                holding_time = self.current_step - self.entry_step if self.entry_step is not None else None
                denom = float(self.entry_price) if self.entry_price else (close_price or 1e-9)
                reward = profit_realized_raw / denom
                trade_executed = True
                self.entry_price = 0.0
                self.entry_step = None
                new_position = -1 if mapped_action == -1 else 0
                if mapped_action == -1:
                    self.entry_price = close_price
                    self.entry_step = self.current_step

        elif pre_position == -1:
            if mapped_action in (0, 1):
                profit_realized_raw = float(self.entry_price) - close_price
                profit_realized_after_costs = profit_realized_raw - self.commission_per_trade
                closed_side = 'short'
                self.net_profit += profit_realized_after_costs
                self.balance += profit_realized_after_costs
                (self.winning_profits if profit_realized_after_costs > 0 else self.losing_losses).append(profit_realized_after_costs)
                holding_time = self.current_step - self.entry_step if self.entry_step is not None else None
                denom = float(self.entry_price) if self.entry_price else (close_price or 1e-9)
                reward = profit_realized_raw / denom
                trade_executed = True
                self.entry_price = 0.0
                self.entry_step = None
                new_position = 1 if mapped_action == 1 else 0
                if mapped_action == 1:
                    self.entry_price = close_price
                    self.entry_step = self.current_step

        self.position = int(new_position)

        if self.position == 0 or not self.entry_price:
            unrealized = 0.0
        elif self.position == 1:
            unrealized = close_price - float(self.entry_price)
        else:
            unrealized = float(self.entry_price) - close_price

        self.equity = float(self.balance + unrealized)
        self.returns_history.append(float(reward))
        self.current_episode_equities.append(self.equity)

        info = self._get_info()
        info.update({
            "trade_executed": bool(trade_executed),
            "profit_realized_raw": float(profit_realized_raw),
            "profit_realized_after_costs": float(profit_realized_after_costs),
            "closed_side": closed_side,
            "position": int(self.position),
            "entry_price": float(self.entry_price) if self.entry_price else 0.0,
            "holding_time": int(holding_time) if holding_time is not None else None,
            "final_reward": float(reward),
        })

        self.current_step += 1
        self.steps_in_episode += 1

        done = self.current_step >= len(self.df) - 1

        if done:
            last_idx = min(len(self.df) - 1, self.current_step - 1)
            last_price = float(self.df['close'].iloc[last_idx])
            if self.position == 1 and self.entry_price:
                final_pnl = last_price - self.entry_price
                denom = self.entry_price or 1e-9
                reward += final_pnl / denom
                self.balance += final_pnl
                (self.winning_profits if final_pnl > 0 else self.losing_losses).append(final_pnl)
            elif self.position == -1 and self.entry_price:
                final_pnl = self.entry_price - last_price
                denom = self.entry_price or 1e-9
                reward += final_pnl / denom
                self.balance += final_pnl
                (self.winning_profits if final_pnl > 0 else self.losing_losses).append(final_pnl)
            self.equity = float(self.balance)
            self.position = 0

        return self._get_obs(), float(reward), bool(done), False, info

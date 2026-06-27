import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config import FEATURES_DIM, LSTM_HIDDEN_SIZE, N_LSTM_LAYERS, NET_ARCH


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)

        if isinstance(observation_space, spaces.Dict):
            feat_space = observation_space.spaces.get("features")
            if feat_space is None:
                raise ValueError("Expected 'features' key in observation_space Dict")
            in_dim = int(np.prod(feat_space.shape))
        else:
            in_dim = int(np.prod(observation_space.shape))

        self._features_dim = features_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        x = observations["features"] if isinstance(observations, dict) else observations
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        return self.projector(x)


def get_policy_kwargs() -> dict:
    return dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        activation_fn=torch.nn.ReLU,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        n_lstm_layers=N_LSTM_LAYERS,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=NET_ARCH,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5),
    )

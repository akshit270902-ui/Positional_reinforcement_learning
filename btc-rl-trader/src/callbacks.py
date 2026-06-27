import numpy as np
import pathlib
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, log_every_steps: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps

    def _on_step(self) -> bool:
        if self.log_every_steps and self.n_calls % self.log_every_steps == 0:
            if self.verbose:
                print(f"[train] step {self.n_calls}")
        return True


class TrainingRewardLogger(BaseCallback):
    def __init__(self, log_dir: pathlib.Path, log_every_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_every_steps = log_every_steps
        self._episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self._episode_rewards.append(float(ep_info["r"]))
        if self.log_every_steps and self.n_calls % self.log_every_steps == 0:
            arr = np.array(self._episode_rewards)
            np.save(self.log_dir / "train_episode_rewards.npy", arr)
        return True

"""OGBench cube-single-play-singletask data loading.

Returns numpy arrays. Single-task mode has rewards in the dataset (-1 per step, 0 on success).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

DATASET_NAME = "cube-single-play-singletask-v0"


def load(dataset_name: str = DATASET_NAME):
    import ogbench
    env, train_ds, val_ds = ogbench.make_env_and_datasets(dataset_name)
    return env, train_ds, val_ds


class ReplayBatcher:
    """Uniform random batches over the offline dataset."""

    def __init__(self, dataset: dict, batch_size: int, seed: int = 0):
        self.obs = dataset["observations"]
        self.act = dataset["actions"]
        self.rew = dataset["rewards"]
        self.next_obs = dataset["next_observations"]
        self.mask = dataset["masks"]
        self.n = len(self.obs)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def sample(self):
        idx = self.rng.integers(0, self.n, size=self.batch_size)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "mask": self.mask[idx],
        }

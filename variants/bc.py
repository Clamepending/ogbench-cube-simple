"""Behavior cloning baseline.

Deterministic MLP actor trained with MSE on (obs, act). No value function.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.nets import DeterministicActor


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BCTrainer:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.device = _device()
        self.actor = DeterministicActor(obs_dim, act_dim, hidden=(256, 256, 256)).to(self.device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    def config(self) -> dict:
        return {
            "algo": "bc",
            "steps": 200_000,
            "eval_every": 20_000,
            "eval_episodes": 50,
            "batch_size": 256,
            "lr": 3e-4,
            "hidden": [256, 256, 256],
            "device": str(self.device),
        }

    def update(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        pred = self.actor(obs)
        loss = F.mse_loss(pred, act)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"bc_loss": float(loss.detach().cpu())}

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0)


def build_trainer(obs_dim: int, act_dim: int, seed: int = 0) -> BCTrainer:
    return BCTrainer(obs_dim=obs_dim, act_dim=act_dim, seed=seed)

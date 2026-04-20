"""Implicit Q-Learning (Kostrikov et al., 2022).

Decouples value learning (expectile regression of V toward Q) from policy
learning (advantage-weighted regression). Avoids actor-critic bootstrapping
instabilities that hurt TD3+BC on sparse-reward tasks.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets import MLP, DeterministicActor


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden=hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden=hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class V(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden=hidden)

    def forward(self, obs: torch.Tensor):
        return self.v(obs).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLTrainer:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.device = _device()
        self.act_dim = act_dim

        _hidden = (512, 512, 512)
        self.actor = DeterministicActor(obs_dim, act_dim, hidden=_hidden).to(self.device)
        self.q = TwinQ(obs_dim, act_dim, hidden=_hidden).to(self.device)
        self.q_tgt = copy.deepcopy(self.q).eval()
        self.v = V(obs_dim, hidden=_hidden).to(self.device)

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005          # target network soft update rate
        self.expectile = 0.7      # τ in the paper; expectile for V(s)
        self.awr_beta = 3.0       # inverse-temperature for advantage weighting

    def config(self) -> dict:
        return {
            "algo": "iql",
            "steps": 200_000,
            "eval_every": 20_000,
            "eval_episodes": 50,
            "batch_size": 256,
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "expectile": 0.7,
            "awr_beta": 3.0,
            "hidden": [512, 512, 512],
            "device": str(self.device),
        }

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        with torch.no_grad():
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self, batch: dict) -> dict:
        d = self.device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=d)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=d)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=d)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=d)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=d)

        # Value: expectile regression of V(s) toward min(Q_tgt(s, a_data))
        with torch.no_grad():
            q1_t, q2_t = self.q_tgt(obs, act)
            q_t = torch.min(q1_t, q2_t)
        v_pred = self.v(obs)
        v_loss = expectile_loss(q_t - v_pred, self.expectile)
        self.opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        self.opt_v.step()

        # Q: Bellman target uses V(s') (no policy evaluation at next state)
        with torch.no_grad():
            v_next = self.v(next_obs)
            target = rew + self.gamma * mask * v_next
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        # Actor: advantage-weighted regression toward a_data
        with torch.no_grad():
            q1_f, q2_f = self.q(obs, act)
            q_f = torch.min(q1_f, q2_f)
            adv = q_f - self.v(obs)
            weight = torch.exp(self.awr_beta * adv).clamp(max=100.0)
        pi = self.actor(obs)
        actor_loss = (weight * F.mse_loss(pi, act, reduction="none").mean(dim=-1)).mean()
        self.opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_a.step()

        self._soft_update(self.q, self.q_tgt)

        return {
            "v_loss": float(v_loss.detach().cpu()),
            "q_loss": float(q_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "adv_mean": float(adv.mean().detach().cpu()),
        }

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0)


def build_trainer(obs_dim: int, act_dim: int, seed: int = 0) -> IQLTrainer:
    return IQLTrainer(obs_dim=obs_dim, act_dim=act_dim, seed=seed)

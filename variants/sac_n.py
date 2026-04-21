"""SAC-N: offline RL via Q-ensemble pessimism (An et al., 2021).

N independent Q-networks; Bellman target uses `min` over N target heads at
a policy-sampled next-action. No BC anchor. Stochastic tanh-Gaussian actor
trained to maximize `min_i Q_i(s,a) - α·log π(a|s)` with learned α.

The failure mode that killed TD3+BC and ReBRAC on this sparse-reward task
was `-Q.mean()` dominating `β · MSE(π, a_data)` in the actor loss, pushing
the actor OOD. SAC-N replaces BC regularization entirely with pessimism
(min over N independent Q heads), so there is no Q-magnitude-vs-β
balancing to fail.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets import MLP


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class QEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, N: int = 10, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.N = N
        self.heads = nn.ModuleList([MLP(obs_dim + act_dim, 1, hidden=hidden) for _ in range(N)])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([h(x).squeeze(-1) for h in self.heads], dim=0)  # (N, B)


class TanhGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pre = mu + std * noise  # reparameterized
        a = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # Tanh correction (numerically stable): log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
        logp = logp - (2 * (math.log(2.0) - pre - F.softplus(-2 * pre))).sum(-1)
        return a, logp

    def act_det(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class SACNTrainer:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.device = _device()
        self.act_dim = act_dim
        self.N = 10

        self.actor = TanhGaussianActor(obs_dim, act_dim).to(self.device)
        self.q = QEnsemble(obs_dim, act_dim, N=self.N).to(self.device)
        self.q_tgt = copy.deepcopy(self.q).eval()

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(math.log(0.2), device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(act_dim)

        self.gamma = 0.99
        self.tau = 0.005

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def config(self) -> dict:
        return {
            "algo": "sac_n",
            "steps": 200_000,
            "eval_every": 20_000,
            "eval_episodes": 50,
            "batch_size": 256,
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "N": self.N,
            "target_entropy": self.target_entropy,
            "hidden": [256, 256, 256],
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

        # Q update: min-over-N target pessimism at policy-sampled next action
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(next_obs)
            q_next = self.q_tgt(next_obs, a_next)  # (N, B)
            q_next_min = q_next.min(dim=0).values
            target = rew + self.gamma * mask * (q_next_min - self.alpha() * logp_next)
        q_all = self.q(obs, act)  # (N, B)
        q_loss = F.mse_loss(q_all, target.unsqueeze(0).expand_as(q_all))
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        # Actor: maximize min-over-N Q - α · log π
        a_pi, logp_pi = self.actor.sample(obs)
        q_pi = self.q(obs, a_pi).min(dim=0).values
        actor_loss = (self.alpha() * logp_pi - q_pi).mean()
        self.opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_a.step()

        # α update: match target entropy
        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        self._soft_update(self.q, self.q_tgt)

        return {
            "q_loss": float(q_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha": float(self.alpha().cpu()),
            "logp_pi_mean": float(logp_pi.mean().detach().cpu()),
        }

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor.act_det(o).squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0)


def build_trainer(obs_dim: int, act_dim: int, seed: int = 0) -> SACNTrainer:
    return SACNTrainer(obs_dim=obs_dim, act_dim=act_dim, seed=seed)

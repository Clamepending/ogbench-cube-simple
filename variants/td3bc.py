"""TD3+BC offline RL (Fujimoto & Gu, 2021).

Twin critics + target networks + policy-delayed updates + BC regularization
on the actor objective. One of the strongest simple offline-RL baselines.
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


class TD3BCTrainer:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.device = _device()
        self.act_dim = act_dim

        self.actor = DeterministicActor(obs_dim, act_dim).to(self.device)
        self.actor_tgt = copy.deepcopy(self.actor).eval()
        self.q = TwinQ(obs_dim, act_dim).to(self.device)
        self.q_tgt = copy.deepcopy(self.q).eval()

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.alpha = 2.5  # BC weight from paper
        self._step = 0

    def config(self) -> dict:
        return {
            "algo": "td3bc",
            "steps": 200_000,
            "eval_every": 20_000,
            "eval_episodes": 50,
            "batch_size": 256,
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "alpha": 2.5,
            "hidden": [256, 256, 256],
            "device": str(self.device),
        }

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        with torch.no_grad():
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self, batch: dict) -> dict:
        self._step += 1
        d = self.device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=d)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=d)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=d)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=d)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=d)  # 1 if not terminal

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_tgt(next_obs) + noise).clamp(-1.0, 1.0)
            q1_n, q2_n = self.q_tgt(next_obs, na)
            q_n = torch.min(q1_n, q2_n)
            target = rew + self.gamma * mask * q_n

        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        log = {"q_loss": float(q_loss.detach().cpu())}

        if self._step % self.policy_delay == 0:
            pi = self.actor(obs)
            q_pi, _ = self.q(obs, pi)
            lmbda = self.alpha / (q_pi.abs().mean().detach() + 1e-6)
            actor_loss = -lmbda * q_pi.mean() + F.mse_loss(pi, act)
            self.opt_a.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.opt_a.step()
            self._soft_update(self.actor, self.actor_tgt)
            self._soft_update(self.q, self.q_tgt)
            log["actor_loss"] = float(actor_loss.detach().cpu())

        return log

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0)


def build_trainer(obs_dim: int, act_dim: int, seed: int = 0) -> TD3BCTrainer:
    return TD3BCTrainer(obs_dim=obs_dim, act_dim=act_dim, seed=seed)

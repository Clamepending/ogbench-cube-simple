"""ReBRAC: Revisited Behavior-Regularized Actor-Critic (Tarasov et al., 2023).

Extends TD3+BC with two ideas:
1. Actor-side BC penalty with a small fixed coefficient β_a (no λ-normalization trick).
2. Critic-side BC penalty on the target action: the Q-target at s' is computed from
   Q_tgt(s', π_tgt(s')) − β_c · ||π_tgt(s') − a_next||^2. This pulls the target-policy
   action back toward the dataset at the bootstrap site, controlling OOD Q-explosion.
Also uses LayerNorm inside the critic MLPs for stability.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets import DeterministicActor


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LNQ(nn.Module):
    """LayerNorm critic MLP — ReBRAC's stability ingredient."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers = []
        last = obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class TwinLNQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.q1 = LNQ(obs_dim, act_dim, hidden=hidden)
        self.q2 = LNQ(obs_dim, act_dim, hidden=hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.q1(obs, act), self.q2(obs, act)


class ReBRACTrainer:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.device = _device()
        self.act_dim = act_dim

        self.actor = DeterministicActor(obs_dim, act_dim).to(self.device)
        self.actor_tgt = copy.deepcopy(self.actor).eval()
        self.q = TwinLNQ(obs_dim, act_dim).to(self.device)
        self.q_tgt = copy.deepcopy(self.q).eval()

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.beta_actor = 1.0    # actor-side BC penalty (cycle 2: anchor strongly on sparse reward)
        self.beta_critic = 0.1   # critic-side BC penalty on target action
        self._step = 0

    def config(self) -> dict:
        return {
            "algo": "rebrac",
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
            "beta_actor": 1.0,
            "beta_critic": 0.1,
            "hidden": [256, 256, 256],
            "critic_layernorm": True,
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
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=d)
        next_act = torch.as_tensor(batch["next_act"], dtype=torch.float32, device=d)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_tgt(next_obs) + noise).clamp(-1.0, 1.0)
            q1_n, q2_n = self.q_tgt(next_obs, na)
            q_n = torch.min(q1_n, q2_n)
            bc_penalty_next = (na - next_act).pow(2).mean(dim=-1)
            target = rew + self.gamma * mask * (q_n - self.beta_critic * bc_penalty_next)

        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        log = {"q_loss": float(q_loss.detach().cpu())}

        if self._step % self.policy_delay == 0:
            pi = self.actor(obs)
            q_pi, _ = self.q(obs, pi)
            actor_loss = -q_pi.mean() + self.beta_actor * F.mse_loss(pi, act)
            self.opt_a.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.opt_a.step()
            self._soft_update(self.actor, self.actor_tgt)
            self._soft_update(self.q, self.q_tgt)
            log["actor_loss"] = float(actor_loss.detach().cpu())

        return log

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(o).squeeze(0).cpu().numpy()
        return a


def build_trainer(obs_dim: int, act_dim: int, seed: int = 0) -> ReBRACTrainer:
    return ReBRACTrainer(obs_dim, act_dim, seed=seed)

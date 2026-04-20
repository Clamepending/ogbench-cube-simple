"""Shared networks: MLP policy head with tanh-squashed continuous actions."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicActor(nn.Module):
    """Tanh-squashed deterministic policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.trunk = MLP(obs_dim, act_dim, hidden=hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(obs))

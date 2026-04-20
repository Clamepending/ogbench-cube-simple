"""Rollout evaluation on the OGBench single-task env.

Returns mean episode return and success rate over N episodes, deterministic policy.
"""
from __future__ import annotations

import numpy as np
import torch


def evaluate(env, act_fn, n_episodes: int = 50, seed_start: int = 10_000) -> dict:
    returns = []
    successes = []
    lengths = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        done = False
        trunc = False
        ret = 0.0
        steps = 0
        while not (done or trunc):
            with torch.no_grad():
                act = act_fn(obs)
            obs, r, done, trunc, info = env.step(act)
            ret += float(r)
            steps += 1
        returns.append(ret)
        successes.append(int(info.get("success", False)))
        lengths.append(steps)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns, ddof=1)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_episode_len_mean": float(np.mean(lengths)),
        "n_episodes": int(n_episodes),
    }

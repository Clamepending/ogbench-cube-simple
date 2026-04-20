"""Train one variant for one seed. Writes a JSON metrics file.

Usage: python -m src.train --variant <name> --seed <int> [--outdir outputs] [--steps N]
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data import load, ReplayBatcher
from src.eval import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(variant: str, seed: int, outdir: str, steps_override: int | None = None) -> dict:
    set_seed(seed)
    mod = importlib.import_module(f"variants.{variant}")
    build = getattr(mod, "build_trainer")

    env, train_ds, _val_ds = load()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    trainer = build(obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    train_cfg = trainer.config()
    steps = int(steps_override) if steps_override is not None else int(train_cfg["steps"])
    eval_every = int(train_cfg.get("eval_every", 20_000))
    eval_episodes = int(train_cfg.get("eval_episodes", 50))
    batch_size = int(train_cfg.get("batch_size", 256))

    batcher = ReplayBatcher(train_ds, batch_size=batch_size, seed=seed)

    t0 = time.perf_counter()
    eval_log = []
    for step in range(1, steps + 1):
        batch = batcher.sample()
        trainer.update(batch)
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(env, trainer.act, n_episodes=eval_episodes, seed_start=10_000 + seed * 1000)
            metrics["step"] = step
            eval_log.append(metrics)
            print(f"step={step} return={metrics['eval_return_mean']:.2f} succ={metrics['eval_success_rate']:.2f}")
    train_s = time.perf_counter() - t0

    # Final reported metric = last eval
    final = eval_log[-1]
    out = {
        "variant": variant,
        "seed": int(seed),
        "steps": int(steps),
        "train_seconds": float(train_s),
        "eval_return_mean": float(final["eval_return_mean"]),
        "eval_return_std": float(final["eval_return_std"]),
        "eval_success_rate": float(final["eval_success_rate"]),
        "eval_episode_len_mean": float(final["eval_episode_len_mean"]),
        "n_eval_episodes": int(final["n_episodes"]),
        "eval_log": eval_log,
        "config": train_cfg,
    }

    outpath = Path(outdir) / variant
    outpath.mkdir(parents=True, exist_ok=True)
    (outpath / f"seed_{seed}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def summarize(variant: str, outdir: str) -> dict:
    out = Path(outdir) / variant
    seed_files = sorted(out.glob("seed_*.json"))
    data = [json.loads(p.read_text()) for p in seed_files]
    if not data:
        return {"variant": variant, "n_seeds": 0}
    rets = [d["eval_return_mean"] for d in data]
    succ = [d["eval_success_rate"] for d in data]
    summary = {
        "variant": variant,
        "n_seeds": len(data),
        "eval_return_mean": float(np.mean(rets)),
        "eval_return_std": float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
        "eval_success_rate_mean": float(np.mean(succ)),
        "eval_success_rate_std": float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0,
        "seeds": [int(d["seed"]) for d in data],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--steps", type=int, default=None, help="Override training steps")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    m = run(args.variant, args.seed, args.outdir, steps_override=args.steps)
    print(json.dumps({k: v for k, v in m.items() if k != "eval_log"}))
    if args.summarize:
        s = summarize(args.variant, args.outdir)
        print("SUMMARY", json.dumps(s))


if __name__ == "__main__":
    main()

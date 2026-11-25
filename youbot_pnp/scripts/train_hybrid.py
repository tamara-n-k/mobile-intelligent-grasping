"""Train the MAPLE-style hybrid SAC (youbot_pnp/rl/hybrid_sac.py) on YouBotPnPEnv.

    python youbot_pnp/scripts/train_hybrid.py --cubes 4 --variant fixed_goals --envs 4 --steps 100000 --name run11_hybrid

Output in runs/<name>/: tb/ (TensorBoard), evaluations.npz (same format as SB3's EvalCallback), best_model.pt,
hybrid_<N>_steps.pt checkpoints, final.pt and config.json (all settings of the run).
"""

import argparse
import collections
import json
import time

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from youbot_pnp.rl.hybrid_sac import HybridSAC
from youbot_pnp.scripts.train_sac import EVAL_SEEDS, PRIMITIVES, RUNS_DIR, make_env

CHECKPOINT_EVERY = 25_000  # primitive steps
LOG_EVERY = 1_000          # primitive steps


def evaluate(agent, env, n_episodes):
    """deterministic policy on the fixed evaluation tasks (env cycles through EVAL_SEEDS on every reset)"""
    returns, lengths, successes = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ret, n, info = False, 0.0, 0, {}
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret, n, done = ret + r, n + 1, term or trunc
        returns.append(ret)
        lengths.append(n)
        successes.append(bool(info.get("is_success", False)))
    return np.array(returns), np.array(lengths), np.array(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000, help="primitive steps, summed over all envs")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--variant", default="random")
    parser.add_argument("--curriculum", action="store_true", help="training episodes start with 1-n cubes still to move")
    parser.add_argument("--learning-starts", type=int, default=5_000, help="random steps before the first update")
    parser.add_argument("--utd", type=int, default=1, help="gradient steps per collected transition")
    parser.add_argument("--high-ent-steps", type=int, default=50_000,
                        help="steps with an almost uniform primitive choice (target entropy 0.97 log 3)")
    parser.add_argument("--one-hot-factor", type=float, default=0.5,
                        help="primitive target entropy after that, as a fraction of log 3")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5_000, help="primitive steps between evaluations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name, default: generated from the settings")
    args = parser.parse_args()

    torch.set_num_threads(2)  # leave the other cores to the env processes
    run = args.name or f"hybrid_{args.cubes}cube_{args.variant}_s{args.seed}_{time.strftime('%m%d_%H%M')}"
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    envs = SubprocVecEnv([make_env(args.cubes, args.variant, curriculum=args.curriculum) for _ in range(args.envs)],
                         start_method="forkserver")
    envs.seed(args.seed)
    eval_env = make_env(args.cubes, args.variant, seeds=EVAL_SEEDS)()
    obs = envs.reset()

    agent = HybridSAC(obs.shape[1], lr=args.lr, batch_size=args.batch_size, high_ent_steps=args.high_ent_steps,
                      one_hot_factor=args.one_hot_factor, seed=args.seed)
    writer = SummaryWriter(str(out / "tb"))

    ep_returns, ep_lengths, ep_success = (collections.deque(maxlen=100) for _ in range(3))
    prim = collections.defaultdict(list)  # primitive statistics since the last log, like PrimitiveStats
    train_stats = collections.defaultdict(list)
    evals = {"timesteps": [], "results": [], "ep_lengths": [], "successes": []}
    best = (-1.0, -np.inf)
    t, next_log, next_eval, next_ckpt, t0 = 0, LOG_EVERY, args.eval_every, CHECKPOINT_EVERY, time.time()

    while t < args.steps:
        if t < args.learning_starts:
            actions = agent.random_action(args.envs)
        else:
            actions, _ = agent.predict(obs)
        next_obs, rewards, dones, infos = envs.step(actions)

        for i, info in enumerate(infos):
            # on episode end the VecEnv has already reset: the real last observation is in the info
            real_next = info["terminal_observation"] if dones[i] else next_obs[i]
            terminal = dones[i] and not info.get("TimeLimit.truncated", False)
            agent.buffer.add(obs[i], actions[i], rewards[i], real_next, float(terminal))

            name = info["primitive"]
            prim["primitives/aff"].append(info["aff"])
            prim["primitives/failure_rate"].append(float(not info["success"]))
            prim[f"primitives/success_{name}"].append(float(info["success"]))
            for p in PRIMITIVES:
                prim[f"primitives/chosen_{p}"].append(float(name == p))
            if "episode" in info:  # added by Monitor at the end of an episode
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])
                ep_success.append(float(info.get("is_success", False)))
        obs = next_obs
        t += args.envs

        if t >= args.learning_starts:
            for _ in range(args.utd * args.envs):
                for k, v in agent.update(t).items():
                    train_stats[k].append(v)

        if t >= next_log:
            next_log += LOG_EVERY
            for k, v in {**prim, **train_stats}.items():
                writer.add_scalar(k, np.mean(v), t)
            if ep_returns:
                writer.add_scalar("rollout/ep_rew_mean", np.mean(ep_returns), t)
                writer.add_scalar("rollout/ep_len_mean", np.mean(ep_lengths), t)
                writer.add_scalar("rollout/success_rate", np.mean(ep_success), t)
            fps = t / (time.time() - t0)
            writer.add_scalar("time/fps", fps, t)
            print(f"{t:7d} steps  fps {fps:4.1f}  success {np.mean(ep_success) if ep_success else 0:.2f}  "
                  f"reach {np.mean(prim['primitives/chosen_reach']):.2f}  "
                  f"alpha_s {np.mean(train_stats['train/alpha_s']) if train_stats else float('nan'):.3f}",
                  flush=True)
            prim.clear()
            train_stats.clear()

        if t >= next_eval:
            next_eval += args.eval_every
            r, n, s = evaluate(agent, eval_env, len(EVAL_SEEDS))
            for k, v in zip(evals, (t, r, n, s)):
                evals[k].append(v)
            np.savez(out / "evaluations.npz", **{k: np.array(v) for k, v in evals.items()})
            writer.add_scalar("eval/success_rate", s.mean(), t)
            writer.add_scalar("eval/mean_reward", r.mean(), t)
            writer.add_scalar("eval/mean_ep_length", n.mean(), t)
            print(f"  eval at {t}: success {s.mean():.2f}  reward {r.mean():.2f}  length {n.mean():.1f}", flush=True)
            if (s.mean(), r.mean()) > best:  # success first, reward breaks ties
                best = (s.mean(), r.mean())
                agent.save(out / "best_model.pt")

        if t >= next_ckpt:
            next_ckpt += CHECKPOINT_EVERY
            agent.save(out / f"hybrid_{t}_steps.pt")

    agent.save(out / "final.pt")
    writer.close()
    envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()

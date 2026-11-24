import argparse
import contextlib
import io
import time
from pathlib import Path
from ompl import util as ou

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from youbot_pnp.rl.env import YouBotPnPEnv

PRIMITIVES = ["grasp", "reach", "open"]
EVAL_SEEDS = range(10_000, 10_020)  # fixed tasks for evaluation, never used for training
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"  # <repo>/runs, independent of the working directory


class Quiet(gym.Wrapper):
    """swallows the planner's prints (Executing RRT path ...)"""

    def reset(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.env.reset(**kwargs)

    def step(self, action):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.env.step(action)


class FixedSeeds(gym.Wrapper):
    """every reset takes the next seed of a fixed list, so every evaluation runs the same tasks"""

    def __init__(self, env, seeds):
        super().__init__(env)
        self.seeds, self.i = list(seeds), 0

    def reset(self, **kwargs):
        kwargs["seed"] = self.seeds[self.i % len(self.seeds)]
        self.i += 1
        return self.env.reset(**kwargs)


def make_env(n_cubes, variant, seeds=None, curriculum=False):
    def _init():
        ou.setLogLevel(ou.LOG_WARN)  # OMPL's C++ logger bypasses Quiet (it doesn't use Python's stdout)
        env = Quiet(YouBotPnPEnv(n_cubes=n_cubes, variant=variant, curriculum=curriculum))
        if seeds is not None:
            env = FixedSeeds(env, seeds)
        return Monitor(env)  # records episode return and length
    return _init


class PrimitiveStats(BaseCallback):
    """averages since the last log output: which primitive is chosen, how often it succeeds, the affordance"""

    def _on_step(self):
        for info in self.locals["infos"]:  # one info per env
            name = info["primitive"]
            self.logger.record_mean("primitives/aff", info["aff"])
            self.logger.record_mean("primitives/failure_rate", float(not info["success"]))
            for p in PRIMITIVES:
                self.logger.record_mean(f"primitives/chosen_{p}", float(name == p))
            self.logger.record_mean(f"primitives/success_{name}", float(info["success"]))
        return True  # False would stop training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000, help="primitive steps, summed over all envs")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--curriculum", action="store_true", help="training episodes start with 1-n cubes still to move")
    parser.add_argument("--variant", default="random")
    parser.add_argument("--learning-starts", type=int, default=5_000, help="random steps before the first update")
    parser.add_argument("--utd", type=int, default=1, help="gradient steps per collected transition")
    parser.add_argument("--eval-every", type=int, default=5_000, help="primitive steps between evaluations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name, default: generated from the settings")
    args = parser.parse_args()

    torch.set_num_threads(2)  # leave the other cores to the env processes
    run = args.name or f"sac_{args.cubes}cube_{args.variant}_s{args.seed}_{time.strftime('%m%d_%H%M')}"
    out = str(RUNS_DIR / run)


    train_env = SubprocVecEnv([make_env(args.cubes, args.variant, curriculum=args.curriculum) for _ in range(args.envs)],
                              start_method="forkserver")
    eval_env = DummyVecEnv([make_env(args.cubes, args.variant, seeds=EVAL_SEEDS)])

    model = SAC(
        "MlpPolicy", train_env,
        learning_starts=args.learning_starts,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=args.utd * args.envs,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=out,
        device="cpu",
        seed=args.seed,
        verbose=1,
    )

    callbacks = CallbackList([
        PrimitiveStats(),
        EvalCallback(eval_env, n_eval_episodes=len(EVAL_SEEDS), eval_freq=max(1, args.eval_every // args.envs),
                     best_model_save_path=out, log_path=out, deterministic=True),
        CheckpointCallback(save_freq=max(1, 25_000 // args.envs), save_path=out, name_prefix="sac"),
    ])
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="tb", log_interval=20)
    model.save(f"{out}/final")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

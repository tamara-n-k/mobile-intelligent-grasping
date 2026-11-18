"""Watch a trained policy in the viewer (real time) and print every primitive it chooses.
   python youbot_pnp/scripts/watch_policy.py runs/<run>/best_model.zip --episodes 5"""

import argparse
import contextlib
import io

import mujoco.viewer
import numpy as np
from ompl import util as ou
from stable_baselines3 import SAC

from youbot_pnp.rl.env import YouBotPnPEnv
from youbot_pnp.scene.scene import CUBE_HALF_SIZE
from youbot_pnp.utils.viewer_utils import GOAL_ALPHA, geom_rgba, set_fixed_camera, show_goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to a .zip saved by train_sac.py")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--variant", default="random")
    parser.add_argument("--seed", type=int, default=10_000, help="first task seed; 10000-10019 are the eval tasks")
    args = parser.parse_args()

    ou.setLogLevel(ou.LOG_WARN)
    env = YouBotPnPEnv(n_cubes=args.cubes, variant=args.variant)
    model = SAC.load(args.model, device="cpu")
    m = env.world.robot.model

    with mujoco.viewer.launch_passive(m, env.world.robot.data) as viewer:
        with viewer.lock():
            set_fixed_camera(viewer, m)
        env.task.viewer = viewer  # the primitives now redraw the viewer and run in real time

        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            show_goals(viewer, [(env.cfg.goals[c], geom_rgba(m, f"{c}_geom"), GOAL_ALPHA) for c in env.cubes],
                       CUBE_HALF_SIZE)
            print(f"\nepisode {ep} (seed {seed})")
            done, ret = False, 0.0
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    obs, r, term, trunc, info = env.step(action)
                ret += r
                done = term or trunc
                target = env._to_workspace(np.clip(action[3:], -1, 1))[:3]
                print(f"  {info['primitive']:5s} target {np.round(target, 3)}  success {info['success']!s:5s}  "
                      f"aff {info['aff']:.2f}  reward {r:+.2f}")
            print(f"  return {ret:.2f}, solved {info['is_success']}")


if __name__ == "__main__":
    main()

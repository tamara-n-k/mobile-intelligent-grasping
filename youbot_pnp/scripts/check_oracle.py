"""Upper bound for the RL policy: a scripted policy with perfect parameters, run through env.step().

If the oracle does not reach 100%, no learned policy can: the primitives, the success predicate or the task
generator are the limit, not SAC.

    python youbot_pnp/scripts/check_oracle.py --cubes 1 --seeds 40
    python youbot_pnp/scripts/check_oracle.py --cubes 4 --seeds 60
"""

import argparse
import collections
import contextlib
import io

import numpy as np
from ompl import util as ou

from youbot_pnp.rl.env import YouBotPnPEnv, WS_LOW, WS_HIGH

PRIMITIVES = ["grasp", "reach", "open"]
REACH_DONE = 0.02  # m: closer than this to the goal, the oracle opens instead of reaching again


def to_action(name, xyz):
    """inverse of YouBotPnPEnv._to_workspace: primitive one-hot + parameters in [-1, 1]"""
    params = np.clip(2 * (np.asarray(xyz) - WS_LOW) / (WS_HIGH - WS_LOW) - 1, -1, 1)
    action = np.zeros(7, dtype=np.float32)
    action[PRIMITIVES.index(name)] = 1.0
    action[3:6] = params  # action[6] (yaw) is not used by the primitives
    return action


def oracle_step(env):
    """(primitive, target) a perfect policy would choose in the current state, None if all cubes are placed"""
    held = env._held_cube()
    if held is not None:
        goal = np.array(env.cfg.goals[held])
        far = np.linalg.norm(env.world.scene.cube_pos(held) - goal) > REACH_DONE
        return ("reach" if far else "open"), goal
    unplaced = env._unplaced()
    if not unplaced:
        return None
    return "grasp", env.world.scene.cube_pos(unplaced[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--variant", default="random")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--first-seed", type=int, default=0)
    args = parser.parse_args()

    ou.setLogLevel(ou.LOG_ERROR)
    env = YouBotPnPEnv(n_cubes=args.cubes, variant=args.variant)
    solved, first_failure = 0, collections.Counter()

    for seed in range(args.first_seed, args.first_seed + args.seeds):
        log, done = [], False
        with contextlib.redirect_stdout(io.StringIO()):  # the planner prints every move
            env.reset(seed=seed)
            while not done:
                choice = oracle_step(env)
                if choice is None:
                    break
                name, target = choice
                _, _, terminated, truncated, info = env.step(to_action(name, target))
                log.append((name, info["success"]))
                done = terminated or truncated
        ok = env._all_placed()
        solved += ok
        if not ok:
            failed = [name for name, success in log if not success]
            first_failure[failed[0] if failed else "all primitives ok, task not solved"] += 1
        steps = " ".join(f"{name[0]}{'+' if success else '-'}" for name, success in log)
        print(f"seed {seed}: {'OK  ' if ok else 'FAIL'} {len(log):2d} steps  {steps}", flush=True)

    print(f"\n{args.variant}, {args.cubes} cube(s): {solved}/{args.seeds} solved ({100 * solved / args.seeds:.0f}%)")
    if first_failure:
        print("first failing primitive per failed episode:", dict(first_failure))


if __name__ == "__main__":
    main()
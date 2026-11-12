"""Timing check (plan step 1): wall time per primitive and where it goes. Headless, no viewer."""

import argparse
import contextlib
import io
import time
from collections import defaultdict

import numpy as np

from youbot_pnp.rl.env import YouBotPnPEnv, WS_LOW, WS_HIGH

PRIMITIVES = ["grasp", "reach", "open"]


class Profiler:
    """Wraps methods of live objects and sums their exclusive wall time:
    time spent in a wrapped method called from inside another one is booked only to the inner one."""

    def __init__(self):
        self.times = defaultdict(float)
        self._stack = []  # one [label, wall time of wrapped children] per active call

    def wrap(self, obj, method, label):
        # puts a timer around a method of an existing object
        fn = getattr(obj, method)

        def timed(*args, **kwargs):
            self._stack.append([label, 0.0])
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                total = time.perf_counter() - t0
                _, children = self._stack.pop()
                self.times[label] += total - children
                if self._stack:
                    self._stack[-1][1] += total

        setattr(obj, method, timed)  # instance attribute shadows the class method

    def take(self):
        """return and reset what was collected since the last call"""
        out = dict(self.times)
        self.times.clear()
        return out


def instrument(env):
    task, p = env.task, Profiler()
    p.wrap(task.grasp_sampler, "find_valid_grasp", "grasp/place sampling")
    p.wrap(task.grasp_sampler, "find_valid_placement", "grasp/place sampling")
    p.wrap(task.planner, "find_path_rrt", "RRT (OMPL)")
    p.wrap(task.planner, "find_linear_path", "linear path")
    p.wrap(task.planner, "collision_free", "collision checks")
    p.wrap(task.kin, "solve", "IK")
    p.wrap(env.world.robot, "step", "physics (mj_step)")
    for method in ("execute_path", "servo_to", "run_for"):
        p.wrap(task, method, "sim loop overhead")
    return p


def action_for(name, xyz):
    """inverse of env._to_workspace: a 7-dim action that runs `name` at xyz"""
    a = -np.ones(7, dtype=np.float32)
    a[PRIMITIVES.index(name)] = 1.0
    a[3:6] = 2 * (np.asarray(xyz) - WS_LOW) / (WS_HIGH - WS_LOW) - 1
    return np.clip(a, -1, 1)  # a[6] (yaw) is unused: grasp takes the yaw from the sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = YouBotPnPEnv(n_cubes=1)
    prof = instrument(env)
    cube = env.cubes[0]
    rows, reset_wall = [], []  # rows: (primitive, success, wall, sim_time, breakdown)

    for ep in range(args.episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            env.reset(seed=ep)
            reset_wall.append(time.perf_counter() - t0)
            prof.take()
            targets = {"grasp": env.world.scene.cube_pos(cube), "reach": env.cfg.goals[cube], "open": WS_LOW}
            for name in PRIMITIVES:
                t0 = time.perf_counter()
                *_, info = env.step(action_for(name, targets[name]))
                rows.append((name, info["success"], time.perf_counter() - t0, info["sim_time"], prof.take()))
                if not info["success"]:
                    break
        print(f"episode {ep}: " + ", ".join(f"{r[0]} {'ok' if r[1] else 'FAIL'} {r[2]:.2f}s"
                                            for r in rows if r is not None)[-60:])

    print(f"\nreset: mean {np.mean(reset_wall):.2f}s wall")
    for name in PRIMITIVES:
        r = [x for x in rows if x[0] == name]
        if not r:
            continue
        wall, sim = np.array([x[2] for x in r]), np.array([x[3] for x in r])
        print(f"\n{name}: {sum(x[1] for x in r)}/{len(r)} succeeded | wall mean {wall.mean():.2f}s, "
              f"max {wall.max():.2f}s | sim mean {sim.mean():.2f}s | sim/wall {sim.sum() / wall.sum():.1f}x")
        parts = defaultdict(float)
        for x in r:
            for label, t in x[4].items():
                parts[label] += t
        parts["other (obs, reward, logic)"] = wall.sum() - sum(parts.values())
        for label, t in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"  {label:28s} {t / len(r):6.3f}s  {100 * t / wall.sum():5.1f}%")

    mean_step = np.mean([x[2] for x in rows])
    print(f"\nmean wall per primitive: {mean_step:.2f}s -> {3600 / mean_step:.0f} primitives/hour per env")
    print(f"MAPLE warm-up (30000 primitives) on 1 env: {30000 * mean_step / 3600:.1f} h (upper bound, "
          f"random primitives often fail early and are cheaper)")


if __name__ == "__main__":
    main()

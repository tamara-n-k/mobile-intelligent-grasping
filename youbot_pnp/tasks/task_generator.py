from dataclasses import dataclass, field
import numpy as np

from youbot_pnp.scene.scene import CUBES, CUBE_Z

# two rows of slots the arm already reaches: current cube row (x=0.93) and place row (x=0.86)
#   index:   x=0.93 -> 0..3, x=0.86 -> 4..7, each row at y = 0.26, 0.32, 0.38, 0.44
SLOTS = np.array([[x, y] for x in (0.93, 0.86) for y in (0.26, 0.32, 0.38, 0.44)])
START_JITTER = 0.01  # m, random offset of the start position inside its slot
# nominal task: cube starts from mujoco/environment.xml and goals = PLACE_POSITIONS in main.py (order of CUBES)
NOMINAL_START_SLOTS = [1, 3, 0, 2]
NOMINAL_GOAL_SLOTS = [7, 4, 6, 5]
VARIANTS = ["nominal", "random", "occupied", "swap"]


@dataclass
class TaskConfig:
    variant: str
    starts: dict          # cube -> (x, y, yaw)
    goals: dict           # cube -> (x, y, z)
    disturbances: dict = field(default_factory=dict)  # e.g. slip_prob, obs_noise_std (used later)


def generate_task(seed, variant="random", cubes=CUBES):
    """
    start and goal slot for every cube; the same seed and variant always give the same task.
      nominal:  the current scene and place positions
      random:   random start slots, goals on free slots only
      occupied: one cube's goal is another cube's start (no cycle): the order matters
      swap:     two cubes swap their slots: one of them needs a free slot as buffer first
    """
    rng = np.random.default_rng(seed)
    n = len(cubes)
    if variant in ("occupied", "swap") and n < 2:
        raise ValueError(f"variant {variant} needs at least 2 cubes")

    if variant == "nominal":
        start_slots, goal_slots = NOMINAL_START_SLOTS[:n], NOMINAL_GOAL_SLOTS[:n]
    else:
        start_slots = rng.choice(len(SLOTS), n, replace=False)
        if variant == "random":
            goal_slots = rng.choice(_free_slots(start_slots), n, replace=False)
        elif variant == "occupied":
            goal_slots = _goals_with_chain(rng, start_slots)
        elif variant == "swap":
            goal_slots = _goals_with_cycle(rng, start_slots)
        else:
            raise ValueError(f"unknown variant {variant}")

    jitter = 0.0 if variant == "nominal" else START_JITTER
    starts, goals = {}, {}
    for cube, s, g in zip(cubes, start_slots, goal_slots):
        x, y = SLOTS[s] + rng.uniform(-jitter, jitter, 2)
        yaw = 0.0 if variant == "nominal" else rng.uniform(0, np.pi / 2)
        starts[cube] = (float(x), float(y), float(yaw))
        goals[cube] = (float(SLOTS[g][0]), float(SLOTS[g][1]), CUBE_Z)
    return TaskConfig(variant, starts, goals)


def _free_slots(start_slots):
    return np.setdiff1d(np.arange(len(SLOTS)), start_slots)


def _goals_with_chain(rng, start_slots):
    """cube a's goal is cube b's start, all other goals are free slots: b has to be moved before a"""
    n = len(start_slots)
    a, b = rng.choice(n, 2, replace=False)
    goals = np.empty(n, dtype=int)
    goals[a] = start_slots[b]
    others = [i for i in range(n) if i != a]
    goals[others] = rng.choice(_free_slots(start_slots), len(others), replace=False)
    return goals


def _goals_with_cycle(rng, start_slots):
    """cubes a and b swap their slots, all other goals are free slots: one of them needs a buffer spot first"""
    n = len(start_slots)
    a, b = rng.choice(n, 2, replace=False)
    goals = np.empty(n, dtype=int)
    goals[a], goals[b] = start_slots[b], start_slots[a]
    others = [i for i in range(n) if i not in (a, b)]
    goals[others] = rng.choice(_free_slots(start_slots), len(others), replace=False)
    return goals

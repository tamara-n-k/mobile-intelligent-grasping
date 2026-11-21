"""Integration test: HRL primitives vs. the FSM on the nominal task."""

import argparse
import numpy as np

from youbot_pnp.fsm.pick_and_place_fsm import finger_contacts
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.rl.primitives import Grasp, Reach, Open
from youbot_pnp.scene.scene import CUBES
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.tasks.task_generator import generate_task
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.world import World

SETTLE_TIME = 0.5
PLACE_TOL = 0.01


def make_world_and_task(cfg, selected_cube):
    """Create a fresh simulation; park all cubes except the one being tested."""
    world = World()
    world.reset(cfg)

    for i, cube in enumerate(CUBES):
        if cube != selected_cube:
            world.scene.place_cube(cube, -1.0, 0.1 * i, 0.0)

    task = PickAndPlace(
        world.robot,
        MotionPlanner(world.robot),
        KinematicsSolver(world.robot),
    )
    task.run_for(SETTLE_TIME)
    return world, task


def cube_is_placed(world, cube, goal):
    distance = np.linalg.norm(world.scene.cube_pos(cube) - goal)
    released = not any(finger_contacts(world.robot, cube))
    return distance < PLACE_TOL and released


def run_hrl(cfg, cube):
    world, task = make_world_and_task(cfg, cube)
    goal = np.asarray(cfg.goals[cube], dtype=float)

    def held_cube():
        if all(finger_contacts(world.robot, cube)):
            return cube
        return None

    def unplaced_cubes():
        return [] if cube_is_placed(world, cube, goal) else [cube]

    primitives = {
        "grasp": Grasp(task, [cube], unplaced_cubes, held_cube),
        "reach": Reach(task, [cube], unplaced_cubes, held_cube),
        "open": Open(task, [cube], unplaced_cubes, held_cube),
    }

    # The grasp primitive uses x/y to select the cube, then reads its exact pose
    # directly from MuJoCo for grasp sampling and motion planning.
    cube_position = world.scene.cube_pos(cube)
    grasp_ok, grasp_time = primitives["grasp"].execute(cube_position)
    print(f"HRL grasp: success={grasp_ok}, sim_time={grasp_time:.2f}s")

    if not grasp_ok:
        return False, np.linalg.norm(world.scene.cube_pos(cube) - goal), False

    reach_ok, reach_time = primitives["reach"].execute(goal)
    print(f"HRL reach: success={reach_ok}, sim_time={reach_time:.2f}s")

    if not reach_ok:
        return False, np.linalg.norm(world.scene.cube_pos(cube) - goal), False

    open_ok, open_time = primitives["open"].execute(None)
    print(f"HRL open:  success={open_ok}, sim_time={open_time:.2f}s")

    final_distance = float(np.linalg.norm(world.scene.cube_pos(cube) - goal))
    released = not any(finger_contacts(world.robot, cube))
    return bool(grasp_ok and reach_ok and open_ok), final_distance, released


def run_fsm(cfg, cube):
    world, task = make_world_and_task(cfg, cube)
    goal = np.asarray(cfg.goals[cube], dtype=float)

    success = task.execute_pick_and_place(cube, goal)
    final_distance = float(np.linalg.norm(world.scene.cube_pos(cube) - goal))
    released = not any(finger_contacts(world.robot, cube))
    print(f"FSM:       success={success}, final_distance={1000 * final_distance:.2f}mm, released={released}")
    return bool(success), final_distance, released


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube", choices=CUBES, default=CUBES[0])
    args = parser.parse_args()

    cfg = generate_task(args.seed, "nominal", cubes=CUBES)
    cube = args.cube
    goal = np.asarray(cfg.goals[cube], dtype=float)

    print("\n=== NOMINAL TASK ===")
    print(f"Cube:  {cube}")
    print(f"Start: {np.round(cfg.starts[cube], 4)}")
    print(f"Goal:  {np.round(goal, 4)}\n")

    print("=== HRL PRIMITIVES ===")
    hrl_ok, hrl_distance, hrl_released = run_hrl(cfg, cube)

    print("\n=== FSM BASELINE ===")
    fsm_ok, fsm_distance, fsm_released = run_fsm(cfg, cube)

    difference = abs(hrl_distance - fsm_distance)
    print("\n=== COMPARISON ===")
    print(f"HRL final distance: {1000 * hrl_distance:.2f} mm")
    print(f"FSM final distance: {1000 * fsm_distance:.2f} mm")
    print(f"Absolute difference: {1000 * difference:.2f} mm")
    print(f"HRL released cube: {hrl_released}")
    print(f"FSM released cube: {fsm_released}")

    hrl_pass = hrl_ok and hrl_released and hrl_distance < PLACE_TOL
    fsm_pass = fsm_ok and fsm_released and fsm_distance < PLACE_TOL
    comparable = difference < PLACE_TOL
    print(f"\nHRL task passed: {hrl_pass}")
    print(f"FSM task passed: {fsm_pass}")
    print(f"Distances comparable (< {1000 * PLACE_TOL:.0f} mm difference): {comparable}")

    if not hrl_pass or not fsm_pass:
        raise SystemExit("FAIL: at least one implementation did not complete the nominal task")


if __name__ == "__main__":
    main()

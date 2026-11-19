import argparse
import numpy as np

from youbot_pnp.fsm.pick_and_place_fsm import finger_contacts
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.rl.primitives import Grasp, Reach, Open
from youbot_pnp.scene.scene import CUBES
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.tasks.task_generator import generate_task, VARIANTS
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

    context = {}

    primitives = {
        "grasp": Grasp(
            task, [cube], unplaced_cubes, held_cube, context=context
        ),
        "reach": Reach(
            task, [cube], unplaced_cubes, held_cube, context=context
        ),
        "open": Open(
            task, [cube], unplaced_cubes, held_cube, context=context
        ),
    }

    results = {}

    print("\n--- Grasp ---")

    cube_position = world.scene.cube_pos(cube)

    grasp_ok, grasp_time = primitives["grasp"].execute(cube_position)
    grasp_contact = all(finger_contacts(world.robot, cube))

    results["grasp"] = grasp_ok and grasp_contact

    print(f"Primitive success: {grasp_ok}")
    print(f"Both fingers in contact: {grasp_contact}")
    print(f"Simulation time: {grasp_time:.2f}s")

    if not results["grasp"]:
        print("Grasp failed. Skipping remaining HRL primitives.")
        return results, None, False, False

    print("\n--- Reach ---")

    reach_ok, reach_time = primitives["reach"].execute(goal)
    held_after_reach = all(finger_contacts(world.robot, cube))

    position_after_reach = world.scene.cube_pos(cube)
    distance_after_reach = float(
        np.linalg.norm(position_after_reach - goal)
    )

    results["reach"] = reach_ok and held_after_reach

    print(f"Primitive success: {reach_ok}")
    print(f"Cube still held: {held_after_reach}")
    print(f"Distance to goal while held: {1000 * distance_after_reach:.2f}mm")
    print(f"Simulation time: {reach_time:.2f}s")

    if not results["reach"]:
        print("Reach failed. Skipping Open.")
        return results, None, False, False

    print("\n--- Open ---")

    open_ok, open_time = primitives["open"].execute(None)

    released = not any(finger_contacts(world.robot, cube))

    print(f"Primitive success: {open_ok}")
    print(f"Cube released: {released}")
    print(f"Simulation time: {open_time:.2f}s")

    results["open"] = open_ok and released

    # Allow the cube to settle before checking its final position.
    task.run_for(SETTLE_TIME)

    final_position = world.scene.cube_pos(cube)
    final_distance = float(np.linalg.norm(final_position - goal))

    contacts_after_settle = finger_contacts(world.robot, cube)
    released_after_settle = not any(contacts_after_settle)
    within_tolerance = final_distance < PLACE_TOL

    placed = cube_is_placed(world, cube, goal)

    print("\n--- Placement diagnostics ---")
    print(f"Finger contacts after settling: {contacts_after_settle}")
    print(f"Released immediately after Open: {released}")
    print(f"Released after settling: {released_after_settle}")
    print(f"Distance within tolerance: {within_tolerance}")
    print(f"PLACE_TOL: {PLACE_TOL:.4f}m")
    print(f"Placement check: {placed}")
    results["placement"] = placed

    print("\n--- Final placement ---")
    print(f"Final cube position: {np.round(final_position, 4)}")
    print(f"Goal position:       {np.round(goal, 4)}")
    print(f"Final distance: {1000 * final_distance:.2f}mm")
    print(f"Released: {released}")
    print(f"Placed: {placed}")

    task_pass = all(results.values())

    print(f"\nHRL task passed: {task_pass}")

    return results, final_distance, released, task_pass


def run_fsm(cfg, cube):
    world, task = make_world_and_task(cfg, cube)
    goal = np.asarray(cfg.goals[cube], dtype=float)

    success = task.execute_pick_and_place(cube, goal)

    task.run_for(SETTLE_TIME)

    final_distance = float(
        np.linalg.norm(world.scene.cube_pos(cube) - goal)
    )

    released = not any(finger_contacts(world.robot, cube))
    placed = cube_is_placed(world, cube, goal)

    print("\n--- FSM result ---")
    print(f"Execution success: {success}")
    print(f"Final distance: {1000 * final_distance:.2f}mm")
    print(f"Released: {released}")
    print(f"Placed: {placed}")

    task_pass = success and released and placed

    print(f"FSM task passed: {task_pass}")

    return task_pass, final_distance, released


def main():
    parser = argparse.ArgumentParser(
        description="Compare HRL primitives against the FSM on a generated task."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to generate the task.",
    )

    parser.add_argument(
        "--cube",
        choices=CUBES,
        default=CUBES[0],
        help="Cube to evaluate.",
    )

    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="nominal",
        help="Task variant: nominal, random, occupied or swap.",
    )

    args = parser.parse_args()

    cfg = generate_task(args.seed, args.variant, cubes=CUBES)

    cube = args.cube
    start = cfg.starts[cube]
    goal = np.asarray(cfg.goals[cube], dtype=float)

    print(f"\n========== {args.variant.upper()} TASK ==========")
    print(f"Seed:  {args.seed}")
    print(f"Cube:  {cube}")
    print(f"Start: {np.round(start, 4)}")
    print(f"Goal:  {np.round(goal, 4)}")

    print("\n========== HRL PRIMITIVES ==========")

    hrl_results, hrl_distance, hrl_released, hrl_pass = run_hrl(
        cfg, cube
    )

    print("\n========== FSM BASELINE ==========")

    fsm_pass, fsm_distance, fsm_released = run_fsm(cfg, cube)

    print("\n========== COMPARISON ==========")

    if hrl_distance is not None:
        difference = abs(hrl_distance - fsm_distance)

        print(f"HRL final distance: {1000 * hrl_distance:.2f}mm")
        print(f"FSM final distance: {1000 * fsm_distance:.2f}mm")
        print(f"Absolute difference: {1000 * difference:.2f}mm")
        print(f"Distances comparable: {difference < PLACE_TOL}")
    else:
        print("HRL did not reach the placement stage.")

    print(f"HRL released cube: {hrl_released}")
    print(f"FSM released cube: {fsm_released}")

    print("\nHRL primitive results:")

    for primitive, result in hrl_results.items():
        print(f"  {primitive}: {result}")

    print(f"\nHRL task passed: {hrl_pass}")
    print(f"FSM task passed: {fsm_pass}")

    if not hrl_pass or not fsm_pass:
        raise SystemExit(
            f"FAIL: at least one implementation did not complete "
            f"the {args.variant} task (seed={args.seed})."
        )

    print(
        f"\nPASS: both implementations completed the "
        f"{args.variant} task (seed={args.seed})."
    )


if __name__ == "__main__":
    main()
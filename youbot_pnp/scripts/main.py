import mujoco.viewer
import numpy as np
import argparse

from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.planners.base_motion_planner import BaseMotionPlanner
from youbot_pnp.robot.robot import TABLE_BASE_POSE, FINGER_OPEN
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.tasks.move_base import MoveBase
from youbot_pnp.tasks.task_generator import generate_task, VARIANTS
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.utils.viewer_utils import CAMERA_NAME, set_fixed_camera
from youbot_pnp.world import World
from youbot_pnp.scene.scene import CUBES, CUBE_HALF_SIZE
from youbot_pnp.utils.viewer_utils import CAMERA_NAME, set_fixed_camera, show_goal, clear_markers, geom_rgba


TABLE_TOP = 0.13
# place positions (cube centers), one per cube in the order of CUBES: a second row in front of the cubes,
# shuffled so that every cube has to cross over
PLACE_POSITIONS = [
    [0.86, 0.44, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.26, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.38, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.32, TABLE_TOP + CUBE_HALF_SIZE],
]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nominal", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drive", action="store_true", help="drive the base to the table instead of starting there")
    args = parser.parse_args()

    cfg = generate_task(args.seed, args.variant)
    print(cfg)
    world = World()
    start_pose = (0.1, -0.5, 0.0) if args.drive else TABLE_BASE_POSE
    world.reset(cfg, base_pose=start_pose)
    robot = world.robot

    kin = KinematicsSolver(robot)
    base_planner = BaseMotionPlanner(robot)
    task = PickAndPlace(robot, MotionPlanner(robot), kin)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            set_fixed_camera(viewer, robot.model, CAMERA_NAME)
        task.viewer = viewer
        if args.drive:
            move_base = MoveBase(robot, base_planner, viewer=viewer)
            move_base.execute_move(goal_xy=TABLE_BASE_POSE[:2], target_yaw=TABLE_BASE_POSE[2])

        # initial settling: cubes come to rest on the table
        task.run_for(0.5)

        results = {}
        for cube in CUBES:
            show_goal(viewer, cfg.goals[cube], geom_rgba(robot.model, f"{cube}_geom"), CUBE_HALF_SIZE)
            results[cube] = task.execute_pick_and_place(object_name=cube, place_positions=cfg.goals[cube])
            print(f"Task {'succeeded' if results[cube] else 'failed'} for {cube}.")
        # move back to home position       
        task.execute_move(robot.home_pose)

        print("\nTask completed.")
        for cube in CUBES:
            error = np.linalg.norm(robot.data.body(cube).xpos - cfg.goals[cube])
            print(f"  {cube:12s} {'succeeded' if results[cube] else 'FAILED   '}  distance to target {1000 * error:.1f} mm")

        try:
            while viewer.is_running():
                task.run_for(1.0 / 60)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main()
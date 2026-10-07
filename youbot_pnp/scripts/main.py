import time

import mujoco.viewer
import numpy as np

from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.scripts.view_scene import CAMERA_NAME, set_fixed_camera
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.utils.kinematics_solver import KinematicsSolver

CUBES = ["cube_red", "cube_green", "cube_blue", "cube_yellow"]
TABLE_TOP = 0.13
CUBE_HALF_SIZE = 0.009
# place positions (cube centers), one per cube in the order of CUBES: a second row in front of the cubes,
# shuffled so that every cube has to cross over
PLACE_POSITIONS = [
    [0.86, 0.44, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.26, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.38, TABLE_TOP + CUBE_HALF_SIZE],
    [0.86, 0.32, TABLE_TOP + CUBE_HALF_SIZE],
]

def main():

    robot = YouBot()
    # teleport robot to home
    robot.reset()
    print("Home position reached: ", robot.get_arm_qpos())

    kin = KinematicsSolver(robot)
    task = PickAndPlace(robot, MotionPlanner(robot), kin)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            set_fixed_camera(viewer, robot.model, CAMERA_NAME)
        task.viewer = viewer

        # initial settling: cubes come to rest on the table
        task.run_for(0.5)

        results = {}
        for cube in CUBES:
            success = task.execute_pick_and_place(object_name=cube, place_positions=PLACE_POSITIONS)
            results[cube] = success
            if not success:
                print(f"Task failed for {cube}.")
            else:
                print(f"Task succeeded for {cube}.")
        # move back to home position       
        task.execute_move(robot.home_pose)

        print("\nTask completed.")
        for cube, target in zip(CUBES, PLACE_POSITIONS):
            error = np.linalg.norm(robot.data.body(cube).xpos - target)
            print(f"  {cube:12s} {'succeeded' if results[cube] else 'FAILED   '}  distance to target {1000 * error:.1f} mm")

        try:
            while viewer.is_running():
                task.run_for(1.0 / 60)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main()

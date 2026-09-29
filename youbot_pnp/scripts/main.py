import time

import mujoco.viewer
import numpy as np

from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.scripts.view_scene import CAMERA_NAME, match_free_camera
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.utils.ik_solver import IKSolver

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

if __name__ == "__main__":

    robot = YouBot()
    robot.reset()
    print("Home position reached: ", robot.get_arm_qpos())

    ik = IKSolver(robot)
    task = PickAndPlace(robot, MotionPlanner(robot), ik)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            match_free_camera(viewer.cam, robot.data, CAMERA_NAME)
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

        print("\nTask completed.")
        for cube, target in zip(CUBES, PLACE_POSITIONS):
            error = np.linalg.norm(robot.data.body(cube).xpos - target)
            print(f"  {cube:12s} {'succeeded' if results[cube] else 'FAILED   '}  distance to target {1000 * error:.1f} mm")

        try:
            while viewer.is_running():
                task.run_for(1.0 / 60)
        except KeyboardInterrupt:
            print("Exiting...")

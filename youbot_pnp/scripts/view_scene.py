import time

import mujoco
import mujoco.viewer
import numpy as np

from youbot_pnp.robot.robot import TABLE_BASE_POSE
from youbot_pnp.utils.viewer_utils import CAMERA_NAME, set_fixed_camera
from youbot_pnp.world import World

# base pose (x, y, yaw) to start the view from
BASE_X, BASE_Y, BASE_YAW = TABLE_BASE_POSE
ORBIT_TARGET = np.array([0.93, 0.35, 0.14])  # the mouse orbits around the view-ray point closest to the cubes


def main():
    robot = World().robot

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            set_fixed_camera(viewer, robot.model, CAMERA_NAME)

        while viewer.is_running():
            start = time.time()
            robot.step()
            viewer.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.time() - start)))

if __name__ == "__main__":
    main()

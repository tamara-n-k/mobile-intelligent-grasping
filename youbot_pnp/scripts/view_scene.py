import time

import mujoco
import mujoco.viewer
import numpy as np

from youbot_pnp.robot.robot import TABLE_BASE_POSE, YouBot

# base pose (x, y, yaw) and the camera from mujoco/environment.xml or youbot.xml to start the view from
BASE_X, BASE_Y, BASE_YAW = TABLE_BASE_POSE
CAMERA_NAME = "wrist_cam"  # rgbd_cam or "wrist_cam"
ORBIT_TARGET = np.array([0.93, 0.35, 0.14])  # the mouse orbits around the view-ray point closest to the cubes

def set_fixed_camera(viewer, model, camera_name):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id != -1:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        print(f"Warning: Camera: '{camera_name}' was not found in the model.")


def main():
    robot = YouBot()

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

import time

import mujoco
import mujoco.viewer
import numpy as np

from youbot_pnp.robot import TABLE_BASE_POSE, YouBot

# base pose (x, y, yaw) and the camera from mujoco/environment.xml or youbot.xml to start the view from
BASE_X, BASE_Y, BASE_YAW = TABLE_BASE_POSE
CAMERA_NAME = "wrist_cam"  # rgbd_cam or "wrist_cam"
ORBIT_TARGET = np.array([0.93, 0.35, 0.14])  # the mouse orbits around the view-ray point closest to the cubes


def match_free_camera(cam, data, camera_name):
    """put the free (mouse-controllable) camera at the pose of camera"""
    pos = data.cam(camera_name).xpos
    forward = -data.cam(camera_name).xmat.reshape(3, 3)[:, 2]  # mujoco cameras look along -z
    distance = max(np.dot(ORBIT_TARGET - pos, forward), 0.1)

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pos + distance * forward
    cam.distance = distance
    cam.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    cam.elevation = np.degrees(np.arcsin(forward[2]))


def main():
    robot = YouBot()
    robot.reset(base_pose=(BASE_X, BASE_Y, BASE_YAW))

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            match_free_camera(viewer.cam, robot.data, CAMERA_NAME)

        while viewer.is_running():
            start = time.time()
            robot.step()
            viewer.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()

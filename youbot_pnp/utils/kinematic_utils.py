import mujoco
import numpy as np

def rotation_error(R_current, R_target):
    """world-frame rotation vector that turns R_current into R_target"""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (R_target @ R_current.T).flatten())
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, quat, 1.0)
    return err


def top_down_rotation(yaw):
    """tcp orientation pointing straight down (z = -world z), gripper rotated by yaw about the vertical"""
    c, s = np.cos(yaw), np.sin(yaw)
    x_axis = np.array([c, s, 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


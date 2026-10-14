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


def yaw_to_quat(yaw):
    """quaternion (w, x, y, z) for a rotation by yaw about the vertical"""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def wrap_angle(a):
    """angle wrapped to [-pi, pi)"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def is_descendant(model, body_id, root_id):
    """True if body_id is root_id or lies below it in the kinematic tree"""
    while body_id != 0:
        if body_id == root_id:
            return True
        body_id = model.body_parentid[body_id]
    return False


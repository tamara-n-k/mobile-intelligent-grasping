import mujoco
import numpy as np
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.kinematic_utils import rotation_error

TCP_SITE = "tcp"


class IKSolver:
    """
    computes the inverse kinematics of the tcp site with damped least squares on the MuJoCo model.
    uses its own MjData, so solving never moves the simulated robot. targets are in world frame.
    """
    def __init__(self, robot: YouBot, pos_tol=1e-3, rot_tol=1e-2, max_iters=200, damping=1e-2, max_step=0.2):
        self.robot = robot
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(TCP_SITE).id

        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.max_iters = max_iters
        self.damping = damping
        self.max_step = max_step  # rad per iteration, keeps the linearization valid

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def forward_kinematics(self, q_arm):
        """tcp position and rotation matrix in world frame"""
        # take the current base pose from the live simulation
        self.data.qpos[:] = self.robot.data.qpos
        self.data.mocap_pos[:] = self.robot.data.mocap_pos
        self.data.mocap_quat[:] = self.robot.data.mocap_quat

        self.data.qpos[self.robot.arm_qadr] = q_arm
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jacSite
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def frame_error(self, q_arm, target_pos, target_rot):
        pos, rot = self.forward_kinematics(q_arm)
        return np.concatenate([target_pos - pos, rotation_error(rot, target_rot)])

    def solve(self, q_init, target_pos, target_rot):
        q = np.clip(np.asarray(q_init, dtype=float), self.robot.arm_lower, self.robot.arm_upper)
        dadr = self.robot.arm_dadr

        for _ in range(self.max_iters):
            err = self.frame_error(q, target_pos, target_rot)
            if np.linalg.norm(err[:3]) < self.pos_tol and np.linalg.norm(err[3:]) < self.rot_tol:
                return q

            # jacobian of the tcp w.r.t. the arm joints only
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
            J = np.vstack([self.jacp[:, dadr], self.jacr[:, dadr]])

            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), err)
            step = np.max(np.abs(dq))
            if step > self.max_step:
                dq *= self.max_step / step
            q = np.clip(q + dq, self.robot.arm_lower, self.robot.arm_upper)

        return None

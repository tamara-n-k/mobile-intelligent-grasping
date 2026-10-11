import os

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_PATH = os.path.join(REPO_ROOT, "mujoco", "scene.xml")

# define robot
ARM_JOINTS = [f"arm_joint_{i}" for i in range(1, 6)]
FINGER_JOINTS = ["gripper_finger_joint_l", "gripper_finger_joint_r"]
ARM_ACTUATORS = [f"arm{i}" for i in range(1, 6)]
FINGER_ACTUATORS = ["gripper_l", "gripper_r"]

# define object locations
TABLE_BASE_POSE = (0.52, 0.35, 0.0)  # x, y, yaw
BASE_HEIGHT = 0.0815  # wheel radius + caster drop, wheels resting on the floor

# arm folded up above the base, all joints well inside their limits
HOME_POSE = np.array([2.95, 1.0, -1.5, 2.0, 2.95])

FINGER_OPEN = 0.0115  # per finger -> 22.8mm gap
FINGER_CLOSED = 0.0


class YouBot:
    """
    holds the youBot model/data and maps arm and gripper joints by name, since qpos/qvel/ctrl
    also contain the base freejoint, the caster/wheel joints and the cubes
    """
    def __init__(self, xml_path=SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model

        arm_ids = [m.joint(n).id for n in ARM_JOINTS]
        finger_ids = [m.joint(n).id for n in FINGER_JOINTS]
        self.arm_qadr = m.jnt_qposadr[arm_ids]
        self.arm_dadr = m.jnt_dofadr[arm_ids]
        self.finger_qadr = m.jnt_qposadr[finger_ids]
        self.arm_act = np.array([m.actuator(n).id for n in ARM_ACTUATORS])
        self.finger_act = np.array([m.actuator(n).id for n in FINGER_ACTUATORS])
        self.n_arm_dofs = len(ARM_JOINTS)
        self.arm_lower, self.arm_upper = m.jnt_range[arm_ids].T

        base_joint = m.body("base_link").jntadr[0]
        self.base_qadr = m.jnt_qposadr[base_joint]
        self.base_mocap = m.body("base_anchor").mocapid[0]

        self.home_pose = HOME_POSE.copy()

    def get_base_pose(self):
        """
        Retorna la posición (x, y, z) y el cuaternión (w, x, y, z) de la base móvil.
        """
        pos = self.data.qpos[self.base_qadr:self.base_qadr + 3].copy()
        quat = self.data.qpos[self.base_qadr + 3:self.base_qadr + 7].copy()
        return pos, quat

    def get_base_xy_yaw(self):
        """
        Retorna la posición (x, y) y el ángulo yaw actual de la base calculado desde su cuaternión.
        """
        pos, quat = self.get_base_pose()
        w, x, y, z = quat
        # Conversion of quaternion to yaw angle (Z-axis rotation)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return pos[0], pos[1], yaw
        
    def place_base(self, x, y, yaw):
        """teleport the base; moves the freejoint and the mocap anchor the base is welded to"""
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        pos = np.array([x, y, BASE_HEIGHT])
        self.data.qpos[self.base_qadr:self.base_qadr + 3] = pos
        self.data.qpos[self.base_qadr + 3:self.base_qadr + 7] = quat
        self.data.mocap_pos[self.base_mocap] = pos
        self.data.mocap_quat[self.base_mocap] = quat

    def execute_base_path(self, path_xy, yaw=0.0, viewer=None):
        """
        Recorre una lista de waypoints (x, y) de forma fluida 
        actualizando el mocap y avanzando la simulación paso a paso.
        """
        for wp in path_xy:
            pos = np.array([wp[0], wp[1], BASE_HEIGHT])
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            
            # We smoothly update the anchor and the base's qpos.
            self.data.mocap_pos[self.base_mocap] = pos
            self.data.mocap_quat[self.base_mocap] = quat
            self.data.qpos[self.base_qadr:self.base_qadr + 3] = pos
            self.data.qpos[self.base_qadr + 3:self.base_qadr + 7] = quat

            # We advance the simulation by several waypoint steps to ensure smooth movement.
            for _ in range(3):
                self.step()
                if viewer is not None:
                    viewer.sync()

    def set_arm_qpos(self, q_arm):
        self.data.qpos[self.arm_qadr] = q_arm

    def get_arm_qpos(self):
        return self.data.qpos[self.arm_qadr].copy()

    # position actuator targets
    def set_arm_ctrl(self, q_arm):
        self.data.ctrl[self.arm_act] = np.clip(q_arm, self.arm_lower, self.arm_upper)

    def set_gripper_ctrl(self, finger_pos):
        self.data.ctrl[self.finger_act] = np.clip(finger_pos, FINGER_CLOSED, FINGER_OPEN)

    def reset(self, base_pose=TABLE_BASE_POSE, q_arm=None):
        mujoco.mj_resetData(self.model, self.data)
        q_arm = self.home_pose if q_arm is None else np.asarray(q_arm)
        self.place_base(*base_pose)
        self.set_arm_qpos(q_arm)
        self.data.qpos[self.finger_qadr] = FINGER_OPEN
        self.set_arm_ctrl(q_arm)
        self.set_gripper_ctrl(FINGER_OPEN)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        # gravity/coriolis compensation on the arm dofs only (not on the base freejoint)
        self.data.qfrc_applied[:] = 0
        self.data.qfrc_applied[self.arm_dadr] = self.data.qfrc_bias[self.arm_dadr]
        mujoco.mj_step(self.model, self.data)

import os

import mujoco

from youbot_pnp.robot.robot import TABLE_BASE_POSE, YouBot
from youbot_pnp.scene.scene import Scene

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_PATH = os.path.join(REPO_ROOT, "mujoco", "scene.xml")


class World:
    """loads the MuJoCo scene once and holds everything in it: the robot and the objects on the table"""
    def __init__(self, xml_path=SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot = YouBot(self.model, self.data)
        self.scene = Scene(self.model, self.data)

    def reset(self, task_cfg=None, base_pose=TABLE_BASE_POSE, q_arm=None):
        """full reset: everything back to the XML state, then the robot at base_pose and the cubes from the task"""
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset(base_pose, q_arm)
        if task_cfg is not None:
            self.scene.apply_task(task_cfg)
        mujoco.mj_forward(self.model, self.data)

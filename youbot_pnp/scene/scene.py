import mujoco

from youbot_pnp.utils.kinematic_utils import yaw_to_quat

# cubes in mujoco/environment.xml
CUBES = ["cube_red", "cube_green", "cube_blue", "cube_yellow"]
TABLE_TOP = 0.13
CUBE_HALF_SIZE = 0.009
CUBE_Z = TABLE_TOP + CUBE_HALF_SIZE  # cube center resting on the table
DROP_HEIGHT = 0.001  # cubes are placed slightly above the table so they do not start in contact


class Scene:
    """
    the objects around the robot (cubes on the table): places them and reads their poses.
    model/data are loaded and owned by the World
    """
    def __init__(self, model, data, cubes=CUBES):
        self.model = model
        self.data = data
        self.cubes = list(cubes)
        joints = {cube: model.joint(f"{cube}_joint") for cube in self.cubes}
        self._qadr = {cube: joint.qposadr[0] for cube, joint in joints.items()}
        self._dadr = {cube: joint.dofadr[0] for cube, joint in joints.items()}

    def place_cube(self, name, x, y, yaw, z=CUBE_Z + DROP_HEIGHT):
        """teleport a cube (freejoint: position + quaternion) and stop it"""
        qadr, dadr = self._qadr[name], self._dadr[name]
        self.data.qpos[qadr:qadr + 3] = (x, y, z)
        self.data.qpos[qadr + 3:qadr + 7] = yaw_to_quat(yaw)
        self.data.qvel[dadr:dadr + 6] = 0.0

    def apply_task(self, task_cfg):
        """put the cubes at the start poses of a TaskConfig"""
        for cube, (x, y, yaw) in task_cfg.starts.items():
            self.place_cube(cube, x, y, yaw)
        mujoco.mj_forward(self.model, self.data)

    def cube_pos(self, name):
        return self.data.body(name).xpos.copy()

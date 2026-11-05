import gymnasium as gym
from gymnasium import spaces
import numpy as np

from youbot_pnp.fsm.pick_and_place_fsm import finger_contacts
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.rl.primitives import StubPrimitive
from youbot_pnp.robot.robot import FINGER_OPEN
from youbot_pnp.scene.scene import CUBES, CUBE_HALF_SIZE, TABLE_TOP
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.tasks.task_generator import generate_task
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.world import World

LAMBDA = 1.0
PLACE_TOL = 0.01
WS_LOW, WS_HIGH = np.array([0.82, 0.22, 0.13]), np.array([0.98, 0.48, 0.16])    # actions
OBS_LOW, OBS_HIGH = np.array([0.40, 0.00, 0.10]), np.array([1.00, 0.70, 0.60])  # observations


class YouBotPnPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_cubes=1, variant="random"):
        self.cubes = CUBES[:n_cubes]
        self.variant = variant
        self.max_steps = 5 * n_cubes
        self.world = World()
        robot = self.world.robot
        self.kin = KinematicsSolver(robot)
        self.task = PickAndPlace(robot, MotionPlanner(robot), self.kin)
        self.primitives = {name: StubPrimitive() for name in ("grasp", "reach", "open")}

        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        obs_dim = 9 * n_cubes + 7
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        # starts a new episode: generates a task, resets sim
        super().reset(seed=seed)
        task_seed = int(self.np_random.integers(2**31))
        self.cfg = generate_task(task_seed, self.variant, cubes=self.cubes)
        self.world.reset(self.cfg)
        self._park_unused_cubes()
        self.task.run_for(0.5)
        self.n_steps, self.last_success = 0, True
        return self._obs(), {} # returns first obs

    def step(self, action):
        # runs one whole primitive and returns the result
        action = np.clip(action, -1, 1)
        name = ["grasp", "reach", "open"][int(np.argmax(action[:3]))]
        target = self._to_workspace(action[3:])       # map p1-p4 to x, y, z, yaw

        aff = self._affordance(name, target)          # before the robot moves
        success, sim_time = self.primitives[name].execute(target)

        self.n_steps += 1
        self.last_success = success
        reward = self._task_reward(success) - LAMBDA * (1 - aff)
        terminated = self._all_placed() or self._cube_lost()
        truncated = self.n_steps >= self.max_steps
        info = {"primitive": name, "success": success, "aff": aff, "sim_time": sim_time}
        return self._obs(), reward, terminated, truncated, info

    def _is_placed(self, cube):
        close = np.linalg.norm(self.world.scene.cube_pos(cube) - self.cfg.goals[cube]) < PLACE_TOL
        return close and not any(finger_contacts(self.world.robot, cube))

    def _park_unused_cubes(self):
        for i, cube in enumerate(CUBES[len(self.cubes):]):
            self.world.scene.place_cube(cube, -1.0, 0.1 * i, 0.0, z=CUBE_HALF_SIZE + 0.001)

    def _to_workspace(self, p):
        xyz = WS_LOW + (p[:3] + 1) / 2 * (WS_HIGH - WS_LOW)
        yaw = (p[3] + 1) / 2 * (np.pi / 2) % (np.pi / 2)
        return np.append(xyz, yaw)

    def _affordance(self, name, target):
        return 1.0  # placeholder until aff_score exists

    def _held_cube(self):
        for cube in self.cubes:
            if all(finger_contacts(self.world.robot, cube)):
                return cube
        return None

    def _all_placed(self):
        return all(self._is_placed(c) for c in self.cubes)

    def _cube_lost(self):
        return any(self.world.scene.cube_pos(c)[2] < TABLE_TOP for c in self.cubes)

    def _task_reward(self, success):
        reward = float(sum(self._is_placed(c) for c in self.cubes))
        held = self._held_cube()
        if held is not None:
            d = np.linalg.norm(self.world.scene.cube_pos(held) - self.cfg.goals[held])
            reward += 0.25 + 1 - np.tanh(10 * d)
        if self._all_placed():
            reward += 10.0
        if not success:
            reward -= 0.1
        return float(reward)

    def _obs(self):
        robot, scene = self.world.robot, self.world.scene
        unit = lambda p: 2 * (np.asarray(p) - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1
        yaw_feat = lambda R: [np.cos(4 * np.arctan2(R[1, 0], R[0, 0])),
                            np.sin(4 * np.arctan2(R[1, 0], R[0, 0]))]
        obs = []
        for cube in self.cubes:
            R = robot.data.body(cube).xmat.reshape(3, 3)
            obs += [*unit(scene.cube_pos(cube)), *yaw_feat(R), *unit(self.cfg.goals[cube])]
        tcp_pos, tcp_rot = self.kin.forward_kinematics(robot.get_arm_qpos())
        obs += [*unit(tcp_pos), *yaw_feat(tcp_rot)]
        obs += [robot.data.qpos[robot.finger_qadr].sum() / (2 * FINGER_OPEN) * 2 - 1]
        held = self._held_cube()
        obs += [float(c == held) for c in self.cubes]
        obs += [float(self.last_success)]
        return np.array(obs, dtype=np.float32)


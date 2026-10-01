import time

import numpy as np

from youbot_pnp.fsm.pick_and_place_fsm import PickAndPlaceStateMachine
from youbot_pnp.planners.grasp_planner import GraspSampler
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.kinematics_solver import KinematicsSolver

MAX_JOINT_VEL = 0.5  # rad/s when executing paths, lower = slower (hardware limit is ~1.57 rad/s)
SERVO_KI = 2.0  # 1/s, integral gain of the final servo (see servo_to)
VIEWER_FPS = 60  # viewer.sync() waits for a screen refresh, so only sync at frame rate


class PickAndPlace:
    """
    runs the task on the robot: plans moves with the motion planner and executes them in the simulation
    """
    def __init__(self, robot: YouBot, planner: MotionPlanner, k_solver: KinematicsSolver, viewer=None):
        self.robot = robot
        self.planner = planner
        self.kin = k_solver
        self.viewer = viewer  # mujoco passive viewer, or None to run headless
        self.grasp_sampler = GraspSampler(self.robot, self.kin, self.planner)
        self.place_index = 0

        self._last_sync_sim = -np.inf
        self._last_sync_wall = time.time()

    # ---------- simulation / viewer ----------

    def sync_viewer(self):
        """redraw at VIEWER_FPS and keep the simulation at real time while watching"""
        if self.viewer is None or not self.viewer.is_running():
            return
        data = self.robot.data
        sim_elapsed = data.time - self._last_sync_sim
        if sim_elapsed < 1.0 / VIEWER_FPS:
            return
        ahead = sim_elapsed - (time.time() - self._last_sync_wall)
        if 0 < ahead < 1.0:
            time.sleep(ahead)
        self.viewer.sync()
        self._last_sync_sim = data.time
        self._last_sync_wall = time.time()

    def run_for(self, duration):
        """step the simulation for a duration in seconds (independent of the timestep)"""
        for _ in range(max(1, round(duration / self.robot.model.opt.timestep))):
            self.robot.step()
            self.sync_viewer()

    # ---------- execution ----------

    def servo_to(self, q_arm, finger_pos=None, timeout=2.0, tolerance=0.01, ki=SERVO_KI):
        """servo to a single target with the position actuators until it is reached or the timeout expires"""
        robot = self.robot
        q_arm = np.asarray(q_arm)
        if finger_pos is not None:
            robot.set_gripper_ctrl(finger_pos)

        # joint friction stops a pure position servo short of the target (error ~ frictionloss / kp);
        # the integral term pushes the command further until the remaining error is gone
        offset = np.zeros_like(q_arm)
        dt = robot.model.opt.timestep
        for _ in range(round(timeout / dt)):
            error = q_arm - robot.get_arm_qpos()
            if np.linalg.norm(error) < tolerance:
                return True  # early exit once the target is reached
            offset += ki * dt * error
            robot.set_arm_ctrl(q_arm + offset)
            robot.step()
            self.sync_viewer()
        return np.linalg.norm(q_arm - robot.get_arm_qpos()) < tolerance

    def execute_path(self, path, finger_pos=None, max_joint_vel=MAX_JOINT_VEL, max_tracking_error=0.2):
        """
        streams the waypoints as a trajectory: each intermediate waypoint gets as much time as the largest joint
        needs at max_joint_vel, without waiting for convergence. only the final waypoint has to be reached.
        """
        robot = self.robot
        if finger_pos is not None:
            robot.set_gripper_ctrl(finger_pos)
        prev = robot.get_arm_qpos()
        for q_arm in path[:-1]:
            robot.set_arm_ctrl(q_arm)
            self.run_for(np.max(np.abs(q_arm - prev)) / max_joint_vel)
            prev = q_arm
            # the arm lagging far behind the commanded path means it is blocked (e.g. by a collision)
            error = np.linalg.norm(robot.get_arm_qpos() - q_arm)
            if error > max_tracking_error:
                print(f"Arm lags behind the path. Tracking error: {error:.4f}")
                return False

        if not self.servo_to(path[-1]):
            print(f"Servo failed to reach the final waypoint. Error: {np.linalg.norm(robot.get_arm_qpos() - path[-1]):.4f}")
            return False
        return True

    def execute_move(self, q_goal_arm, object_name=None, linear=False, finger_pos=None, max_joint_vel=MAX_JOINT_VEL):
        """plan from the current arm state to q_goal_arm (RRT, or straight line in joint space) and execute it"""
        q_start_arm = self.robot.get_arm_qpos()
        # contacts with the object being grasped/carried are allowed
        ignore = [object_name] if object_name else []

        if linear:
            print(f"Executing linear path to: {np.round(q_goal_arm, 3)}")
            path = self.planner.find_linear_path(q_start_arm, q_goal_arm, ignore=ignore)
        else:
            print(f"Executing RRT path to: {np.round(q_goal_arm, 3)}")
            path = self.planner.find_path_rrt(q_start_arm, q_goal_arm, ignore=ignore)

        if not path:
            return False
        return self.execute_path(path, finger_pos=finger_pos, max_joint_vel=max_joint_vel)

    # ---------- pick and place ----------

    def execute_pick_and_place(self, object_name, place_positions):
        selected_place_pos = self._select_place_position(place_positions)

        pick_configs = self._compute_pick_configurations(object_name)
        if pick_configs is None:
            return False
        q_grasp, q_pick_approach, q_pick_lift = pick_configs

        place_configs = self._compute_place_configurations(selected_place_pos, q_pick_lift, object_name)
        if place_configs is None:
            return False
        q_place_approach, q_place_drop = place_configs

        print("\n--- Initializing Pick and Place State Machine Engine ---")
        fsm = PickAndPlaceStateMachine(
            task_runner=self,
            object_name=object_name,
            q_pick_approach=q_pick_approach,
            q_grasp=q_grasp,
            q_pick_lift=q_pick_lift,
            q_place_approach=q_place_approach,
            q_place_drop=q_place_drop
        )
        return fsm.run()

    def _select_place_position(self, place_positions):
        """a single (x, y, z) position, or cycles through a list of them (one per call)"""
        positions = np.asarray(place_positions, dtype=float)
        if positions.shape == (3,):
            return positions
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("Expected a single place position with shape (3,) or a list/array of positions with shape (N, 3)")

        index = self.place_index % len(positions)
        selected_position = positions[index]
        self.place_index += 1
        print(f"Selected place position {index + 1}/{len(positions)}: {selected_position}")
        return selected_position

    def _compute_pick_configurations(self, object_name):
        print(f"\n--- Sampling valid grasp for {object_name} ---")
        object_pos = self.robot.data.body(object_name).xpos.copy()

        # the sampler already checks the approach pose above the grasp and the straight move down
        q_grasp, _, q_approach = self.grasp_sampler.find_valid_grasp(object_pos, object_name)
        if q_grasp is None:
            print(f"No valid pick grasp found for {object_name}")
            return None

        # approach and lift use the same pose above the object
        return q_grasp, q_approach, q_approach

    def _compute_place_configurations(self, place_pos, q_pick_lift, object_name):
        print(f"\n--- Calculating place targets for {np.round(place_pos, 3)} ---")
        q_place_drop, _, q_place_approach = self.grasp_sampler.find_valid_placement(
            place_pos=place_pos,
            q_seed=q_pick_lift,
            object_name=object_name
        )
        if q_place_drop is None:
            print(f"No valid placement found for {np.round(place_pos, 3)}")
            return None

        return q_place_approach, q_place_drop

import time

import numpy as np

from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.ik_solver import IKSolver

MAX_JOINT_VEL = 0.5  # rad/s when executing paths, lower = slower (hardware limit is ~1.57 rad/s)
SERVO_KI = 2.0  # 1/s, integral gain of the final servo (see servo_to)
VIEWER_FPS = 60  # viewer.sync() waits for a screen refresh, so only sync at frame rate


class PickAndPlace:
    """
    runs the task on the robot: plans moves with the motion planner and executes them in the simulation
    """
    def __init__(self, robot: YouBot, planner: MotionPlanner, ik_solver: IKSolver, viewer=None):
        self.robot = robot
        self.planner = planner
        self.ik = ik_solver
        self.viewer = viewer  # mujoco passive viewer, or None to run headless

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

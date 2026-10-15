import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og

from youbot_pnp.utils.kinematic_utils import is_descendant, yaw_to_quat

WHEELS = ["wheel_link_fl", "wheel_link_fr", "wheel_link_bl", "wheel_link_br"]

class BaseValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, collision_func):
        super().__init__(si)
        self.collision_func = collision_func

    def isValid(self, state):
        return self.collision_func(state)


class BaseMotionPlanner:
    def __init__(self, robot, planning_time=2.0, step_size=0.1, check_resolution=0.01):
        self.robot = robot
        self.model = robot.model
        m = self.model
        
        self.planner_data = mujoco.MjData(m)

        robot_root = m.body("base_link").id
        self.robot_bodies = {b for b in range(m.nbody) if is_descendant(m, b, robot_root)}
        self.wheel_bodies = {m.body(n).id for n in WHEELS}
        self.world_body = 0
        self.check_yaw = None  # yaw used to validate each state (None: the one the base currently has)

        self.space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -1.0)
        bounds.setHigh(0, 2.0)
        bounds.setLow(1, -1.0)
        bounds.setHigh(1, 2.0)
        self.space.setBounds(bounds)

        self.si = ob.SpaceInformation(self.space)
        self.svc = BaseValidityChecker(self.si, self._state_valid)
        self.si.setStateValidityChecker(self.svc)
        self.si.setStateValidityCheckingResolution(check_resolution)
        self.si.setup()

        self.pdef = ob.ProblemDefinition(self.si)
        self.pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.si))
        self.planner = og.RRTConnect(self.si)
        self.planner.setRange(step_size)
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()
        self.planning_time = planning_time

    def _contact_allowed(self, b1, b2):
        pair = {b1, b2}
        # Allow wheels to be in contact with the floor (world)
        if pair & self.wheel_bodies and self.world_body in pair:
            return True
        r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
        if not r1 and not r2:
            return True  # Contacts between objects external to the robot
        if r1 and r2:
            return True  # Self-contacts (e.g., fingers): the arm remains stationary while the base moves
        return False     # Robot vs. environment (table, walls, cubes...)

    def _update_planner_data(self, state_xy):
        d = self.planner_data
        d.qpos[:] = self.robot.data.qpos
        d.mocap_pos[:] = self.robot.data.mocap_pos
        d.mocap_quat[:] = self.robot.data.mocap_quat

        base_qadr = self.robot.base_qadr
        d.qpos[base_qadr] = state_xy[0]
        d.qpos[base_qadr + 1] = state_xy[1]
        if self.check_yaw is not None:
            d.qpos[base_qadr + 3:base_qadr + 7] = yaw_to_quat(self.check_yaw)

        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)

    def collision_free(self, state_xy):
        self._update_planner_data(state_xy)
        m, d = self.model, self.planner_data
        
        for con in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if not self._contact_allowed(b1, b2):
                return False
        return True

    def _state_valid(self, state):
        return self.collision_free([state[0], state[1]])

    def find_path_rrt(self, start_xy, goal_xy, yaw=None):
        self.check_yaw = yaw
        start, goal = self.space.allocState(), self.space.allocState()
        start[0], start[1] = start_xy[0], start_xy[1]
        goal[0], goal[1] = goal_xy[0], goal_xy[1]

        if not self.collision_free(start_xy) or not self.collision_free(goal_xy):
            print("CRITICAL: The base's start point or finish point is in collision.")
            return None

        self.pdef.clearSolutionPaths()
        self.pdef.clearStartStates()
        self.pdef.setStartAndGoalStates(start, goal)
        self.planner.clear()

        if not self.planner.solve(self.planning_time) or not self.pdef.hasExactSolution():
            print("No path was found for the base.")
            return None

        path = self.pdef.getSolutionPath()
        og.PathSimplifier(self.si).simplify(path, 1.0)

        states = path.getStates()
        return [np.array([s[0], s[1]]) for s in states]
import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og

from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.kinematic_utils import is_descendant

WHEELS = ["wheel_link_fl", "wheel_link_fr", "wheel_link_bl", "wheel_link_br"]
FINGERS = ["gripper_finger_link_l", "gripper_finger_link_r"]


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, collision_func):
        super().__init__(si)
        self.collision_func = collision_func

    def isValid(self, state):
        return self.collision_func(state)


class MotionPlanner:
    """
    takes joint positions and plans a collision-free arm trajectory from start to goal with OMPL RRTConnect.
    collision checks run on a separate MjData, so planning never moves the live simulation.
    """
    def __init__(self, robot: YouBot, planning_time=2.0, step_size=0.2, check_resolution=0.003):
        self.robot = robot
        self.model = robot.model
        m = self.model
        # separate MuJoCo data used only for planning queries
        self.planner_data = mujoco.MjData(m)

        # body classification for the contact rules
        robot_root = m.body("base_link").id
        self.robot_bodies = {b for b in range(m.nbody) if is_descendant(m, b, robot_root)}
        self.wheel_bodies = {m.body(n).id for n in WHEELS}
        self.finger_bodies = {m.body(n).id for n in FINGERS}
        self.world_body = 0
        self.ignored_bodies = set()  # e.g. the cube that is being grasped or carried

        # configuration space: the 5 arm joints within their limits
        n = robot.n_arm_dofs
        self.space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i in range(n):
            bounds.setLow(i, robot.arm_lower[i])
            bounds.setHigh(i, robot.arm_upper[i])
        self.space.setBounds(bounds)

        # setup space information
        self.si = ob.SpaceInformation(self.space)
        self.svc = ValidityChecker(self.si, self._state_valid)
        self.si.setStateValidityChecker(self.svc)
        # fraction of the space extent between collision checks along an edge;
        # OMPL's default (1%) is ~0.1 rad here, which lets the gripper skip over the small cubes
        self.si.setStateValidityCheckingResolution(check_resolution)
        self.si.setup()

        # problem definition
        self.pdef = ob.ProblemDefinition(self.si)
        self.pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.si))
        self.planner = og.RRTConnect(self.si)
        self.planner.setRange(step_size)
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()
        self.planning_time = planning_time

    def set_ignored_bodies(self, body_names=()):
        """bodies whose contacts are ignored, e.g. the cube while grasping/carrying it"""
        self.ignored_bodies = {self.model.body(n).id for n in body_names}

    # ---------- collision checking ----------

    def _contact_allowed(self, b1, b2):
        if b1 in self.ignored_bodies or b2 in self.ignored_bodies:
            return True
        r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
        if not r1 and not r2:
            return True  # contacts between objects (cube on table) don't concern the arm
        pair = {b1, b2}
        if pair & self.wheel_bodies and self.world_body in pair:
            return True  # wheels resting on the floor
        if pair <= self.finger_bodies:
            return True  # fingers touching each other when closed
        return False  # robot vs environment, or robot vs itself

    def _update_planner_data(self, q_arm):
        d = self.planner_data
        # take base pose and object positions from the live simulation, then set the arm
        d.qpos[:] = self.robot.data.qpos
        d.mocap_pos[:] = self.robot.data.mocap_pos
        d.mocap_quat[:] = self.robot.data.mocap_quat
        d.qpos[self.robot.arm_qadr] = q_arm
        # kinematics + collision detection only; no dynamics needed
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)

    def forbidden_contacts(self, q_arm):
        """names of all colliding body pairs for q_arm (for debugging)"""
        self._update_planner_data(q_arm)
        m, d = self.model, self.planner_data
        contacts = set()  # one entry per body pair, even if they touch at several points
        for con in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if not self._contact_allowed(b1, b2):
                contacts.add((m.body(b1).name or "world", m.body(b2).name or "world"))
        return sorted(contacts)

    def collision_free(self, q_arm):
        self._update_planner_data(q_arm)
        m, d = self.model, self.planner_data
        for con in d.contact[:d.ncon]:
            if not self._contact_allowed(m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]):
                return False
        return True

    def _state_valid(self, state):
        return self.collision_free([state[i] for i in range(self.robot.n_arm_dofs)])

    # ---------- planning ----------

    def _validate(self, q_start, q_goal):
        for name, q in [("Start", q_start), ("Goal", q_goal)]:
            if not self.collision_free(q):
                print(f"CRITICAL: {name} state in collision!")
                for b1, b2 in self.forbidden_contacts(q):
                    print(f" CONTACT: {b1} <--> {b2}")
                return False
        return True

    def find_path_rrt(self, q_start, q_goal, ignore=(), n_waypoints=200):
        """collision-free joint path from q_start to q_goal as a list of arrays, or None"""
        self.set_ignored_bodies(ignore)
        q_start, q_goal = np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float)
        # integration leaves the arm ~1e-8 outside its limits (e.g. -7e-9 with range [0, ...]) and OMPL rejects that start
        q_start = np.clip(q_start, self.robot.arm_lower, self.robot.arm_upper)
        if not self._validate(q_start, q_goal):
            return None

        start, goal = self.space.allocState(), self.space.allocState()
        for i in range(self.robot.n_arm_dofs):
            start[i] = q_start[i]
            goal[i] = q_goal[i]

        # clear OMPL's internal state before planning
        self.pdef.clearSolutionPaths()
        self.pdef.clearStartStates()
        self.pdef.setStartAndGoalStates(start, goal)
        self.planner.clear()

        if not self.planner.solve(self.planning_time):
            print("No path found at all")
            return None
        # only use the path if it actually reached the goal
        if not self.pdef.hasExactSolution():
            print("Found only an APPROXIMATE path (not safe to execute)")
            return None

        path = self.pdef.getSolutionPath()
        og.PathSimplifier(self.si).simplify(path, 1.0)
        path.interpolate(n_waypoints)
        print(f"Found EXACT path with {path.getStateCount()} states")
        states = path.getStates()
        return [np.array([s[i] for i in range(self.robot.n_arm_dofs)]) for s in states]

    def find_linear_path(self, q_start, q_goal, ignore=(), steps=100):
        """straight line in joint space (e.g. short descend/lift moves), or None if it collides"""
        self.set_ignored_bodies(ignore)
        q_start, q_goal = np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float)
        if not self._validate(q_start, q_goal):
            return None

        path = []
        for i in range(steps):
            alpha = i / (steps - 1)
            q_interp = (1 - alpha) * q_start + alpha * q_goal
            if not self.collision_free(q_interp):
                print(f"Collision detected at step {i} of linear path!")
                return None
            path.append(q_interp)
        return path
import numpy as np

from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.utils.kinematic_utils import top_down_rotation

GRASP_OFFSET = 0.004  # tcp slightly above the cube center, keeps the fingertips off the table
APPROACH_HEIGHT = 0.05  # approach/lift pose above grasp and place; higher needs arm_joint_4 beyond its limit


class GraspSampler:
    """
    samples top-down grasp/place poses for the cubes, solves the IK and picks the best collision-free one.
    a pose is a (position, rotation matrix) tuple in world frame.
    """
    def __init__(self, robot: YouBot, solver: KinematicsSolver, planner: MotionPlanner):
        self.robot = robot
        self.solver = solver
        self.planner = planner

        # standard seed for IK: the home pose (arm folded up, elbow up)
        self.default_seed = robot.home_pose.copy()

    def _calculate_joint_penalty(self, q):
        q_limit_min = self.robot.arm_lower
        q_limit_max = self.robot.arm_upper
        q_center = (q_limit_min + q_limit_max) / 2.0
        # normalized by the joint ranges, which differ a lot on the youBot (2.7 - 5.9 rad)
        return np.sum(((q - q_center) / (q_limit_max - q_limit_min)) ** 2)

    def _object_yaw(self, object_name):
        """yaw of the cube's faces; the cube looks the same every 90 degrees, so it is returned in [0, pi/2)"""
        R = self.robot.data.body(object_name).xmat.reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0]) % (np.pi / 2)

    def _create_target_pose(self, pos, yaw):
        # top-down: tcp z-axis points straight down, the fingers close along the gripper y-axis
        pos = np.array(pos, dtype=np.float64).flatten() + np.array([0.0, 0.0, GRASP_OFFSET])
        return pos, top_down_rotation(yaw)

    def _linear_move_free(self, q_a, q_b, ignore, steps=50):
        """straight joint-space move between two configurations is collision-free"""
        self.planner.set_ignored_bodies(ignore)
        free = all(self.planner.collision_free(q_a + a * (q_b - q_a)) for a in np.linspace(0, 1, steps))
        self.planner.set_ignored_bodies()
        return free

    def _approach_config(self, q_target, target_pos, target_rot, approach_ignore, object_name):
        """
        configuration APPROACH_HEIGHT above the target with the same orientation, seeded from the target
        solution so both share the elbow/wrist configuration; None if it is unreachable, collides,
        or the straight move down to the target collides
        """
        q_approach = self.solver.solve(q_target, target_pos + np.array([0.0, 0.0, APPROACH_HEIGHT]), target_rot)
        if q_approach is None:
            return None
        self.planner.set_ignored_bodies(approach_ignore)
        free = self.planner.collision_free(q_approach)
        self.planner.set_ignored_bodies()
        if not free or not self._linear_move_free(q_approach, q_target, [object_name]):
            return None
        return q_approach

    def find_valid_grasp(self, object_pos, object_name):
        candidates = []
        # the fingers must close on two opposite faces: the 4 yaws aligned with the cube
        # (every 90 degrees, since arm_joint_5 cannot turn a full circle, not all of them are reachable)
        cube_yaw = self._object_yaw(object_name)
        rolls = cube_yaw + np.arange(4) * np.pi / 2

        for roll in rolls:
            target_pos, target_rot = self._create_target_pose(object_pos, roll)
            q_sol = self.solver.solve(self.default_seed, target_pos, target_rot)

            if q_sol is not None:
                # ignore the cube so the fingers may touch it
                self.planner.set_ignored_bodies([object_name])
                is_safe = self.planner.collision_free(q_sol)
                self.planner.set_ignored_bodies()
                if not is_safe:
                    continue
                # the approach (= lift) pose above must be reachable and must not touch the cube yet
                q_approach = self._approach_config(q_sol, target_pos, target_rot, [], object_name)
                if q_approach is None:
                    continue
                # no hand clearance term: collision_free already rejects every contact of the hand
                joint_penalty = self._calculate_joint_penalty(q_sol)
                score = -joint_penalty

                candidates.append({'q': q_sol, 'pose': (target_pos, target_rot), 'q_approach': q_approach,
                                   'score': score})

        if not candidates:
            return None, None, None

        # Pick the winner
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        return best['q'], best['pose'], best['q_approach']

    def find_valid_placement(self, place_pos, q_seed, object_name=None):
        candidates = []
        # the cube can be set down at any of its 4 symmetric yaws, aligned with the table
        rolls = np.arange(4) * np.pi / 2

        for roll in rolls:
            target_pos, target_rot = self._create_target_pose(place_pos, roll)
            q_sol = self.solver.solve(q_seed, target_pos, target_rot)

            if q_sol is not None:
                if object_name is not None:
                    self.planner.set_ignored_bodies([object_name])
                is_safe = self.planner.collision_free(q_sol)
                self.planner.set_ignored_bodies()
                if not is_safe:
                    continue
                # the carried cube is in the hand, so it is ignored for the approach as well
                ignore = [object_name] if object_name is not None else []
                q_approach = self._approach_config(q_sol, target_pos, target_rot, ignore, object_name)
                if q_approach is None:
                    continue

                joint_penalty = self._calculate_joint_penalty(q_sol)
                # prefer the yaw that turns the wrist least from the carrying pose
                wrist_turn = abs(q_sol[4] - q_seed[4])

                score = -(joint_penalty * 1.0) - (wrist_turn * 1.0)

                candidates.append({'q': q_sol, 'pose': (target_pos, target_rot), 'q_approach': q_approach,
                                   'score': score})

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        return best['q'], best['pose'], best['q_approach']

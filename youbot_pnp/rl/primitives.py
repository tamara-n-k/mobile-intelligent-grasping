import numpy as np
from youbot_pnp.fsm.pick_and_place_fsm import (finger_contacts, GRIPPER_OPEN_TIME,
                                               GRIPPER_CLOSE_TIME, MAX_SLIP)
from youbot_pnp.planners.grasp_planner import APPROACH_HEIGHT
from youbot_pnp.robot.robot import FINGER_OPEN, FINGER_CLOSED

SNAP_TOL = 0.02  # m

class Primitive:
    def __init__(self, task, cubes, unplaced, held):
        self.task, self.robot = task, task.robot
        self.cubes, self.unplaced, self.held = cubes, unplaced, held

    # Exiting a collision before planning
    def execute(self, target):
        t0 = self.robot.data.time
        self._free_start()
        ok = self._run(target)
        if not ok:
            self._recover()
        return ok, self.robot.data.time - t0
    
    def _free_start(self):
        held = self.held()
        self.task.planner.set_ignored_bodies([held] if held else [])
        q = self.robot.get_arm_qpos()
        if self.task.planner.collision_free(q):
            return
        tcp, R = self.task.kin.forward_kinematics(q)
        q_up = self.task.kin.solve(q, tcp + [0, 0, APPROACH_HEIGHT], R)
        if q_up is not None:
            self.task.servo_to(q_up, finger_pos=None if held else FINGER_OPEN)

    def _open(self):
        self.robot.set_gripper_ctrl(FINGER_OPEN)
        self.task.run_for(GRIPPER_OPEN_TIME)

    def _recover(self):            # leave the robot in a state the next plan can start from
        self._open()

    def _nearest(self, cubes, xyz):
        pos = {c: self.robot.data.body(c).xpos for c in cubes}
        return min(pos, key=lambda c: np.linalg.norm(pos[c][:2] - xyz[:2]), default=None), pos


class Grasp(Primitive):
    def _run(self, target):
        if self.held() is not None:
            return False                              # gripper already occupied
        cube, pos = self._nearest(self.unplaced(), target)
        if cube is None or np.linalg.norm(pos[cube][:2] - target[:2]) > SNAP_TOL:
            return False                              # nothing near the target: no motion
        q_grasp, _, q_app = self.task.grasp_sampler.find_valid_grasp(pos[cube].copy(), cube)
        if q_grasp is None:
            return False
        self._open()
        if not self.task.execute_move(q_app, finger_pos=FINGER_OPEN): return False
        if not self.task.execute_move(q_grasp, cube, linear=True):    return False
        self.robot.set_gripper_ctrl(FINGER_CLOSED)
        self.task.run_for(GRIPPER_CLOSE_TIME)
        if not all(finger_contacts(self.robot, cube)):  return False
        before = self._offset(cube)
        if not self.task.execute_move(q_app, cube, linear=True, finger_pos=FINGER_CLOSED): return False
        return np.linalg.norm(self._offset(cube) - before) < MAX_SLIP

    def _offset(self, cube):
        tcp, _ = self.task.kin.forward_kinematics(self.robot.get_arm_qpos())
        return self.robot.data.body(cube).xpos - tcp


class Reach(Primitive):
    def _run(self, target):
        held = self.held()
        q_drop, _, q_app = self.task.grasp_sampler.find_valid_placement(
            target[:3], self.robot.get_arm_qpos(), held)
        if q_drop is None:
            return False
        fp = FINGER_CLOSED if held else None
        return (self.task.execute_move(q_app, held, finger_pos=fp) and
                self.task.execute_move(q_drop, held, linear=True, finger_pos=fp))

    def _recover(self):
        pass                                   # don't open: that would drop the cube


class Open(Primitive):
    def _run(self, target=None):
        self._open()
        tcp, R = self.task.kin.forward_kinematics(self.robot.get_arm_qpos())
        q_up = self.task.kin.solve(self.robot.get_arm_qpos(), tcp + [0, 0, APPROACH_HEIGHT], R)
        if q_up is None:
            return False
        near, _ = self._nearest(self.cubes, tcp)  # cube next to the fingers
        return self.task.execute_move(q_up, near, linear=True, finger_pos=FINGER_OPEN)

'''
The parameter policy (your low-level policy) outputs an approximate x, y that selects a cube, and grasp then replaces it with the cube's pose from the simulator state.

1. Task policy: chooses to run grasp.
2. Parameter policy: outputs a target, which only has to land within 2 cm of a cube.
3. grasp: finds the nearest cube that isn't placed yet and checks the 2 cm limit.
4. Simulator state: provides the exact position and yaw used for the IK, planning and motion.
'''
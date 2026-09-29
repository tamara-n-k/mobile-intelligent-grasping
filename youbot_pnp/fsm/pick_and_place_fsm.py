from enum import Enum, auto

import numpy as np

from youbot_pnp.robot.robot import FINGER_CLOSED, FINGER_OPEN

GRIPPER_OPEN_TIME = 0.5  # s for the fingers to open
GRIPPER_CLOSE_TIME = 1.0  # s for the fingers to close on the cube
MAX_SLIP = 0.005  # m the object may move relative to the gripper while lifting


class TaskState(Enum):
    IDLE = auto()
    OPEN_GRIPPER = auto()
    APPROACH_PICK = auto()
    DESCEND_PICK = auto()
    GRASPING = auto()
    LIFT_PICK = auto()

    APPROACH_PLACE = auto()
    DESCEND_PLACE = auto()
    RELEASE = auto()
    RETRACT_PLACE = auto()

    HOME_RETURN = auto()

    COMPLETED = auto()
    FAILED = auto()

class PickAndPlaceStateMachine:
    def __init__(self, task_runner, object_name,
                 q_pick_approach, q_grasp, q_pick_lift,
                 q_place_approach, q_place_drop):

        self.task = task_runner
        self.robot = task_runner.robot
        self.object_name = object_name

        # Pick joint configurations
        self.q_pick_approach = q_pick_approach
        self.q_grasp = q_grasp
        self.q_pick_lift = q_pick_lift

        # Place joint configurations
        self.q_place_approach = q_place_approach
        self.q_place_drop = q_place_drop

        self.state = TaskState.IDLE

    def run(self):
        self.state = TaskState.OPEN_GRIPPER

        while self.state not in [TaskState.COMPLETED, TaskState.FAILED]:
            self.update_fsm()
        return self.state == TaskState.COMPLETED


    def update_fsm(self):
        state_handlers = {
            TaskState.OPEN_GRIPPER:   self._handle_open_gripper,
            TaskState.APPROACH_PICK:  self._handle_approach_pick,
            TaskState.DESCEND_PICK:   self._handle_descend_pick,
            TaskState.GRASPING:       self._handle_grasping,
            TaskState.LIFT_PICK:      self._handle_lift_pick,
            TaskState.APPROACH_PLACE: self._handle_approach_place,
            TaskState.DESCEND_PLACE:  self._handle_descend_place,
            TaskState.RELEASE:        self._handle_release,
            TaskState.RETRACT_PLACE:  self._handle_retract_place,
            TaskState.HOME_RETURN:     self._move_home
        }

        handler = state_handlers.get(self.state)
        if handler:
            handler()


    def _handle_open_gripper(self):
        print("FSM: Opening gripper...")
        # the arm keeps its last position target, run_for only lets the fingers move
        self.robot.set_gripper_ctrl(FINGER_OPEN)
        self.task.run_for(GRIPPER_OPEN_TIME)
        self.state = TaskState.APPROACH_PICK


    def _handle_approach_pick(self):
        print(f"FSM: Approaching {self.object_name}...")
        # the object is not ignored here: the approach must not touch it
        if self.task.execute_move(self.q_pick_approach, linear=False):
            self.state = TaskState.DESCEND_PICK
        else:
            self.state = TaskState.FAILED


    def _handle_descend_pick(self):
        print(f"FSM: Descending onto {self.object_name}...")
        success = self.task.execute_move(self.q_grasp, self.object_name, linear=True)
        if success:
            self.state = TaskState.GRASPING
        else:
            self.state = TaskState.FAILED


    def _handle_grasping(self):
        print("FSM: Closing jaws...")
        self.robot.set_gripper_ctrl(FINGER_CLOSED)
        self.task.run_for(GRIPPER_CLOSE_TIME)

        if self.check_grasp():
            print("FSM: Grasp verified successfully!")
            self.state = TaskState.LIFT_PICK
        else:
            print("FSM: CRITICAL - Grasp failed validation check.")
            self.state = TaskState.FAILED


    def _handle_lift_pick(self):
        print("FSM: Elevating structural target payload...")
        # object position relative to the gripper before lifting (not in the Panda version)
        offset_before = self._object_offset()
        if not self.task.execute_move(self.q_pick_lift, self.object_name, linear=False, finger_pos=FINGER_CLOSED):
            self.state = TaskState.FAILED
            return
        # the object must have come up with the gripper instead of slipping out
        slip = np.linalg.norm(self._object_offset() - offset_before)
        if slip > MAX_SLIP:
            print(f"FSM: CRITICAL - {self.object_name} slipped {1000 * slip:.1f} mm while lifting.")
            self.state = TaskState.FAILED
            return
        self.state = TaskState.APPROACH_PLACE


    # placing sequence starts here
    def _handle_approach_place(self):
        print(f"FSM: Transit to target place approach...")
        if self.task.execute_move(self.q_place_approach, self.object_name, linear=False, finger_pos=FINGER_CLOSED):
            self.state = TaskState.DESCEND_PLACE
        else:
            self.state = TaskState.FAILED


    def _handle_descend_place(self):
        print(f"FSM: Linear descent to the place position...")
        if self.task.execute_move(self.q_place_drop, self.object_name, linear=True, finger_pos=FINGER_CLOSED):
            self.state = TaskState.RELEASE
        else:
            self.state = TaskState.FAILED


    def _handle_release(self):
        print("FSM: Opening jaws to release payload...")
        self.robot.set_gripper_ctrl(FINGER_OPEN)
        self.task.run_for(GRIPPER_OPEN_TIME)
        self.state = TaskState.RETRACT_PLACE


    def _handle_retract_place(self):
        print("FSM: Retracting arm straight up back to safety...")
        # the open fingers are still next to the object, so it is ignored
        if self.task.execute_move(self.q_place_approach, self.object_name, linear=True, finger_pos=FINGER_OPEN):
            self.state = TaskState.HOME_RETURN
        else:
            self.state = TaskState.FAILED


    def _move_home(self):
        print("FSM: Moving to home position...")
        if self.task.execute_move(self.robot.home_pose, linear=False):
            self.state = TaskState.COMPLETED
        else:
            self.state = TaskState.FAILED

    def _object_offset(self):
        """object position minus tcp position, in world frame"""
        tcp_pos, _ = self.task.ik.forward_kinematics(self.robot.get_arm_qpos())
        return self.robot.data.body(self.object_name).xpos - tcp_pos

    #### Grasp verification using contact information from Mujoco
    def check_grasp(self):
        data = self.robot.data
        model = self.robot.model

        touch_f1, touch_f2 = False, False

        # Loop through every active contact point
        for i in range(data.ncon):
            contact = data.contact[i]

            # get body ids and convert to body names
            b1_name = model.body(model.geom_bodyid[contact.geom1]).name
            b2_name = model.body(model.geom_bodyid[contact.geom2]).name

            if self.object_name in (b1_name, b2_name):
                if "gripper_finger_link_l" in (b1_name, b2_name):
                    touch_f1 = True
                if "gripper_finger_link_r" in (b1_name, b2_name):
                    touch_f2 = True

        print(f"Grasp check: Left finger contact: {touch_f1}, Right finger contact: {touch_f2}")
        return touch_f1 and touch_f2

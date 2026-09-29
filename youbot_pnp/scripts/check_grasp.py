import mujoco
import mujoco.viewer
import numpy as np

from youbot_pnp.planners.grasp_planner import GraspSampler
from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.robot.robot import FINGER_CLOSED, FINGER_OPEN, YouBot
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.utils.ik_solver import IKSolver

CUBE = "cube_red"
SHOW_VIEWER = True
CLOSE_TIME = 1.0  # s to let the fingers close on the cube


def finger_contacts(robot, object_name):
    """which fingers touch the object (like check_grasp in the Panda FSM)"""
    m, d = robot.model, robot.data
    left = right = False
    for con in d.contact[:d.ncon]:
        names = {m.body(m.geom_bodyid[con.geom1]).name, m.body(m.geom_bodyid[con.geom2]).name}
        if object_name in names:
            left |= "gripper_finger_link_l" in names
            right |= "gripper_finger_link_r" in names
    return left, right


def run(task, sampler, robot):
    cube_start = robot.data.body(CUBE).xpos.copy()

    q_grasp, pose, q_approach = sampler.find_valid_grasp(cube_start, CUBE)
    if q_grasp is None:
        print(f"No valid grasp found for {CUBE}")
        return
    print(f"grasp yaw {np.degrees(np.arctan2(pose[1][1, 0], pose[1][0, 0])):.0f} deg, q_grasp {np.round(q_grasp, 2)}")

    ok = (task.execute_move(q_approach, finger_pos=FINGER_OPEN)
          and task.execute_move(q_grasp, object_name=CUBE, linear=True))
    if not ok:
        print("could not reach the grasp pose")
        return

    # close the fingers and let them settle on the cube
    robot.set_gripper_ctrl(FINGER_CLOSED)
    task.run_for(CLOSE_TIME)
    left, right = finger_contacts(robot, CUBE)
    fingers = robot.data.qpos[robot.finger_qadr]
    print(f"closed: left contact {left}, right contact {right}, finger positions {np.round(1000 * fingers, 1)} mm "
          f"(0 = fully closed, {1000 * FINGER_OPEN:.1f} = open)")

    # lift straight up with closed fingers
    tcp_before = task.ik.forward_kinematics(robot.get_arm_qpos())[0]
    cube_before = robot.data.body(CUBE).xpos.copy()
    ok = task.execute_move(q_approach, object_name=CUBE, linear=True, finger_pos=FINGER_CLOSED)
    task.run_for(0.5)
    tcp_rise = task.ik.forward_kinematics(robot.get_arm_qpos())[0][2] - tcp_before[2]
    cube_rise = robot.data.body(CUBE).xpos[2] - cube_before[2]
    lifted = ok and cube_rise > 0.95 * tcp_rise  # the cube must follow the gripper, not slip
    print(f"lift: gripper rose {1000 * tcp_rise:.1f} mm, cube rose {1000 * cube_rise:.1f} mm -> "
          f"{'GRASP HOLDS' if lifted else 'cube not lifted'}")


if __name__ == "__main__":
    robot = YouBot()
    robot.reset()
    ik = IKSolver(robot)
    planner = MotionPlanner(robot)
    sampler = GraspSampler(robot, ik, planner)
    task = PickAndPlace(robot, planner, ik)

    if SHOW_VIEWER:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            task.viewer = viewer
            run(task, sampler, robot)
            while viewer.is_running():
                task.run_for(1.0 / 60)
    else:
        run(task, sampler, robot)

import time

import mujoco.viewer
import numpy as np

from youbot_pnp.planners.motion_planner import MotionPlanner
from youbot_pnp.tasks.pick_and_place import PickAndPlace
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.utils.kinematic_utils import top_down_rotation
from youbot_pnp.world import World

CUBES = ["cube_red", "cube_green", "cube_blue", "cube_yellow"]
EXECUTE_CUBE = "cube_red"  # this one is executed in simulation after planning
SHOW_VIEWER = True
APPROACH_HEIGHT = 0.05  # pre-grasp above the cube; higher needs arm_joint_4 beyond its limit
GRASP_OFFSET = 0.004  # tcp slightly above the cube center, keeps the fingertips off the table


def top_down_ik(kin, planner, pos, q_seed):
    """top-down IK trying the 4 cube-symmetric gripper yaws, returns the first collision-free solution"""
    for k in range(4):
        q = kin.solve(q_seed, pos, top_down_rotation(k * np.pi / 2))
        if q is not None and planner.collision_free(q):
            return q
    return None


if __name__ == "__main__":
    world = World()
    world.reset()
    robot = world.robot
    kin = KinematicsSolver(robot)
    planner = MotionPlanner(robot)
    task = PickAndPlace(robot, planner, kin)

    # the collision checker itself: home must be free, the arm folded back onto its own base must not be
    print("home collision free:", planner.collision_free(robot.home_pose))
    q_bad = np.array([2.95, 0.1, -4.1, 2.0, 2.95])
    print("folded pose collision free:", planner.collision_free(q_bad), planner.forbidden_contacts(q_bad))

    plans = {}
    for cube in CUBES:
        cube_pos = robot.data.body(cube).xpos
        planner.set_ignored_bodies()  # pre-grasp must not touch any cube
        q_pre = top_down_ik(kin, planner, cube_pos + [0, 0, APPROACH_HEIGHT], robot.home_pose)
        if q_pre is None:
            print(f"{cube}: no collision-free pre-grasp IK")
            continue
        planner.set_ignored_bodies([cube])  # fingers may touch the cube they grasp
        q_grasp = top_down_ik(kin, planner, cube_pos + [0, 0, GRASP_OFFSET], q_pre)
        if q_grasp is None:
            print(f"{cube}: no grasp IK")
            continue

        t = time.time()
        path_approach = planner.find_path_rrt(robot.home_pose, q_pre)
        path_descend = planner.find_linear_path(q_pre, q_grasp, ignore=[cube])
        print(f"{cube}: approach {'ok' if path_approach else 'FAILED'}, descend {'ok' if path_descend else 'FAILED'} "
              f"({time.time() - t:.2f}s)")
        if path_approach and path_descend:
            plans[cube] = (q_pre, q_grasp)

    if EXECUTE_CUBE not in plans:
        raise SystemExit(f"nothing to execute for {EXECUTE_CUBE}")
    q_pre, q_grasp = plans[EXECUTE_CUBE]
    cube_start = robot.data.body(EXECUTE_CUBE).xpos.copy()

    def run():
        ok = task.execute_move(q_pre) and task.execute_move(q_grasp, object_name=EXECUTE_CUBE, linear=True)
        tcp_err = np.linalg.norm(kin.forward_kinematics(robot.get_arm_qpos())[0] - cube_start - [0, 0, GRASP_OFFSET])
        print(f"execute {EXECUTE_CUBE}: reached grasp pose {ok}, tcp error {1000 * tcp_err:.1f}mm, "
              f"cube moved {1000 * np.linalg.norm(robot.data.body(EXECUTE_CUBE).xpos - cube_start):.1f}mm")
        ok = task.execute_move(q_pre, object_name=EXECUTE_CUBE, linear=True) and task.execute_move(robot.home_pose)
        print(f"back home: {ok}")

    if SHOW_VIEWER:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            task.viewer = viewer
            run()
            while viewer.is_running():
                task.run_for(1.0 / 60)
    else:
        run()

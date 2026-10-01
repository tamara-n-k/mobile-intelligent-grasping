import numpy as np
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.utils.kinematics_solver import KinematicsSolver
from youbot_pnp.utils.kinematic_utils import top_down_rotation


if __name__ == "__main__":
    # sanity checks: FK -> IK -> FK round trip and top-down reach of every cube
    robot = YouBot()
    robot.reset()
    kin = KinematicsSolver(robot)

    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(20):
        q_true = rng.uniform(robot.arm_lower, robot.arm_upper)
        pos, rot = kin.forward_kinematics(q_true)
        ok += kin.solve(robot.home_pose, pos, rot) is not None
    print(f"round trip from home pose: {ok}/20 random configurations recovered")

    # the gripper grasps a cube the same way every 90 degrees
    for cube in ["cube_red", "cube_green", "cube_blue", "cube_yellow"]:
        target = robot.data.body(cube).xpos + [0, 0, 0.004]
        for k in range(4):
            q = kin.solve(robot.home_pose, target, top_down_rotation(k * np.pi / 2))
            if q is None:
                print(f"{cube:12s} yaw {90 * k:3d}: no solution")
                continue
            pos, _ = kin.forward_kinematics(q)
            print(f"{cube:12s} yaw {90 * k:3d}: q={np.round(q, 2)}  pos err={1000 * np.linalg.norm(pos - target):.2f}mm")

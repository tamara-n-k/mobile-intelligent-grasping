import mujoco
import numpy as np

from youbot_pnp.planners.base_motion_planner import BaseMotionPlanner
from youbot_pnp.robot.robot import YouBot
from youbot_pnp.tasks.sim_task import SimTask

MAX_BASE_VEL = 0.3  # m/s when moving the base
MAX_BASE_YAW_VEL = 0.5  # rad/s when correcting the yaw
MAX_BASE_ACC = 0.5  # m/s², acceleration ramp (keeps the wheels from slipping)
LOOKAHEAD = 0.15  # m, how far ahead along the path the point the controller aims at is
K_YAW = 2.0  # 1/s, yaw gain
K_VEL = 1.5  # 1/s, speed drops in proportion to the remaining distance (smooth braking)
MIN_VEL = 0.03  # m/s, minimum approach speed (wheel friction stalls the base below this)
POS_TOL = 0.002  # m, arrival tolerance
YAW_TOL = 0.01  # rad
CONTROL_PERIOD = 0.01  # s between controller updates (100 Hz)

WHEELS = ["fl", "fr", "bl", "br"]


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def simplify_polyline(points, eps=2e-3):
    """Ramer-Douglas-Peucker: keeps only the vertices of the path (OMPL interpolates it into N collinear points)"""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-12:
        dists = np.linalg.norm(points - a, axis=1)
    else:
        dists = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / norm
    i = int(np.argmax(dists))
    if dists[i] <= eps:
        return np.array([a, b])
    left = simplify_polyline(points[:i + 1], eps)
    right = simplify_polyline(points[i:], eps)
    return np.vstack([left[:-1], right])


class BaseMotion(SimTask):
    """
    Drives the mobile base with its wheels: plans the path with OMPL and executes it with a controller, the same
    way PickAndPlace does for the arm (path -> actuator commands with a speed limit -> final servo).

    Every wheel has a velocity actuator (rolling) and a position actuator (steering, caster_joint), so the base is
    holonomic (swerve drive): the controller computes the velocity (vx, vy, w) the base should have and steers/spins
    each wheel to achieve it. The base_lock weld is disabled while moving; on arrival it is enabled again at the
    pose reached, and the base stays fixed while the arm works.
    """
    def __init__(self, robot: YouBot, base_planner: BaseMotionPlanner, viewer=None):
        super().__init__(robot, viewer)
        self.planner = base_planner

        m = robot.model
        self.wheel_act = np.array([m.actuator(f"wheel_{k}").id for k in WHEELS])
        self.steer_act = np.array([m.actuator(f"caster_{k}").id for k in WHEELS])
        self.steer_qadr = np.array([m.jnt_qposadr[m.joint(f"caster_joint_{k}").id] for k in WHEELS])
        # position of each wheel relative to the base (x, y) and the wheel radius, both read from the model
        self.wheel_xy = np.array([m.body(f"caster_link_{k}").pos[:2] for k in WHEELS])
        wheel_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("wheel_link_fl").id)
        self.wheel_radius = float(m.geom_size[wheel_geom][0])
        self.lock_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "base_lock")

    # ---------- base lock ----------

    def unlock_base(self):
        """release the base from the mocap so it can move with its wheels"""
        self.robot.data.eq_active[self.lock_id] = 0

    def lock_base(self):
        """stop the wheels and weld the base again at the pose it has right now"""
        data = self.robot.data
        data.ctrl[self.wheel_act] = 0.0
        data.ctrl[self.steer_act] = data.qpos[self.steer_qadr]  # the wheels stay as they are
        pos, quat = self.robot.get_base_pose()
        data.mocap_pos[self.robot.base_mocap] = pos
        data.mocap_quat[self.robot.base_mocap] = quat
        data.eq_active[self.lock_id] = 1

    # ---------- wheel control ----------

    def set_base_velocity(self, v_body, omega):
        """
        desired base velocity (vx, vy in the base frame, w in rad/s) -> commands for the 4 wheels: each wheel is
        steered towards its own contact-point velocity and spun at the speed that gives. it does not push until it
        is aligned (cos of the steering error), so it never drags sideways.
        """
        data = self.robot.data
        for i, (x, y) in enumerate(self.wheel_xy):
            vx = v_body[0] - omega * y
            vy = v_body[1] + omega * x
            speed = np.hypot(vx, vy)
            steer = data.qpos[self.steer_qadr[i]]
            if speed < 1e-4:
                data.ctrl[self.steer_act[i]] = steer
                data.ctrl[self.wheel_act[i]] = 0.0
                continue
            error = wrap_angle(np.arctan2(vy, vx) - steer)
            if abs(error) > np.pi / 2:  # shorter to point the wheel the other way and roll it backwards
                error = wrap_angle(error + np.pi)
                speed = -speed
            data.ctrl[self.steer_act[i]] = steer + error
            data.ctrl[self.wheel_act[i]] = speed * max(0.0, np.cos(error)) / self.wheel_radius

    # ---------- execution ----------

    def execute_path(self, path, target_yaw=0.0, max_vel=MAX_BASE_VEL, max_yaw_vel=MAX_BASE_YAW_VEL,
                     max_tracking_error=0.15):
        """
        follows the path (list of (x, y)) with the controller: aims at a point LOOKAHEAD ahead along the path, at
        limited speed and with an acceleration ramp, and holds the yaw at target_yaw. at the end it brakes and
        requires reaching the goal (like servo_to for the arm). returns False if the base drifts away from the path
        or does not arrive.
        """
        robot = self.robot
        pts = simplify_polyline(path)
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        goal = pts[-1]
        # the path is extended LOOKAHEAD past its end so the point being aimed at does not vanish on arrival
        direction = seg[-1] / seg_len[-1] if seg_len[-1] > 1e-9 else np.zeros(2)
        ext = np.vstack([pts, goal + LOOKAHEAD * direction])
        ext_cum = np.concatenate([cum, [total + LOOKAHEAD]])

        dt = robot.model.opt.timestep
        n_sub = max(1, round(CONTROL_PERIOD / dt))
        timeout = 3.0 * total / max_vel + 10.0
        t_start = robot.data.time

        self.unlock_base()
        v_cmd = np.zeros(2)  # commanded velocity in world coordinates (with the acceleration ramp)
        s = 0.0  # progress along the path
        reached = False
        while robot.data.time - t_start < timeout:
            x, y, yaw = robot.get_base_xy_yaw()
            p = np.array([x, y])

            # progress along the path: projection of the base onto the remaining segments, never going backwards
            best = None
            for i in range(len(pts) - 1):
                if cum[i + 1] < s:
                    continue
                t = np.clip(np.dot(p - pts[i], seg[i]) / seg_len[i] ** 2, 0.0, 1.0)
                dist = np.linalg.norm(p - (pts[i] + t * seg[i]))
                if best is None or dist < best[0]:
                    best = (dist, cum[i] + t * seg_len[i])
            cross_track, s = best[0], max(s, best[1])

            if cross_track > max_tracking_error:
                print(f"La base se aleja de la ruta. Error de seguimiento: {cross_track:.3f}")
                break

            dist_goal = np.linalg.norm(goal - p)
            yaw_error = wrap_angle(target_yaw - yaw)
            if dist_goal < POS_TOL and abs(yaw_error) < YAW_TOL and np.linalg.norm(v_cmd) < 0.01:
                reached = True
                break

            carrot = np.array([np.interp(min(s + LOOKAHEAD, total + LOOKAHEAD), ext_cum, ext[:, 0]),
                               np.interp(min(s + LOOKAHEAD, total + LOOKAHEAD), ext_cum, ext[:, 1])])
            to_carrot = carrot - p
            # near the end, aim straight at the goal and brake in proportion to the remaining distance
            if dist_goal < LOOKAHEAD:
                to_carrot = goal - p
            speed = min(max_vel, max(MIN_VEL, K_VEL * dist_goal)) if dist_goal > POS_TOL else 0.0
            norm = np.linalg.norm(to_carrot)
            v_des = speed * to_carrot / norm if norm > 1e-9 else np.zeros(2)

            # acceleration ramp
            dv = v_des - v_cmd
            max_dv = MAX_BASE_ACC * CONTROL_PERIOD
            if np.linalg.norm(dv) > max_dv:
                dv *= max_dv / np.linalg.norm(dv)
            v_cmd = v_cmd + dv

            omega = np.clip(K_YAW * yaw_error, -max_yaw_vel, max_yaw_vel)
            c, sn = np.cos(yaw), np.sin(yaw)
            v_body = np.array([c * v_cmd[0] + sn * v_cmd[1], -sn * v_cmd[0] + c * v_cmd[1]])  # world -> base
            self.set_base_velocity(v_body, omega)

            for _ in range(n_sub):
                robot.step()
                self.sync_viewer()

        self.lock_base()
        if not reached:
            x, y, yaw = robot.get_base_xy_yaw()
            print(f"La base no llegó al destino. Error: {np.linalg.norm(goal - [x, y]):.4f} m, "
                  f"yaw: {wrap_angle(target_yaw - yaw):.4f} rad")
        return reached

    def execute_move(self, goal_xy, target_yaw=0.0, max_vel=MAX_BASE_VEL):
        """
        Plans the base path from its current position and executes it with the wheel controller.
        """
        start_x, start_y, _ = self.robot.get_base_xy_yaw()
        start_xy = (start_x, start_y)

        print(f"\n--- Planificando base desde {np.round(start_xy, 2)} hasta {np.round(goal_xy, 2)} ---")
        # the base holds target_yaw while moving, so the path is validated with that yaw
        path = self.planner.find_path_rrt(start_xy, goal_xy, yaw=target_yaw)

        if path is None:
            print("CRITICAL: No se encontró trayectoria válida para la base.")
            return False

        print("Ejecutando movimiento de la base móvil...")
        if not self.execute_path(path, target_yaw=target_yaw, max_vel=max_vel):
            return False

        print("¡Base móvil colocada en destino con éxito!")
        return True
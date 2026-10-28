import mujoco
import numpy as np

# camera from mujoco/environment.xml or youbot.xml to start the view from
CAMERA_NAME = "rgbd_cam"  # rgbd_cam or "wrist_cam"

# goal markers: transparent "ghost cubes" drawn into the passive viewer's user scene.
# they are visual only: no contacts, not in model/data, so the planner never sees them
GOAL_ALPHA = 0.2  # transparency of the next goal
OTHER_GOAL_ALPHA = 0.1  # transparency of the remaining goals (show_goals)
GOAL_MARGIN = 0.002  # m, the marker is slightly larger than the cube so it stays visible around it


def set_fixed_camera(viewer, model, camera_name=CAMERA_NAME):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id != -1:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        print(f"Warning: Camera: '{camera_name}' was not found in the model.")


def geom_rgba(model, geom_name):
    """color of a geom; the cubes get theirs from a material, so the material color is used if there is one"""
    geom_id = model.geom(geom_name).id
    mat_id = model.geom_matid[geom_id]
    rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
    return rgba.copy()


def clear_markers(viewer):
    """remove all markers drawn with show_goal / show_goals"""
    if viewer is None:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0


def show_goal(viewer, pos, rgba, half_size):
    """draw one ghost cube at pos (replaces the previous markers); does nothing when running headless"""
    show_goals(viewer, [(pos, rgba, GOAL_ALPHA)], half_size)


def show_goals(viewer, goals, half_size):
    """
    draw a ghost cube per goal (replaces the previous markers); goals is a list of (pos, rgba, alpha).
    the markers stay until they are replaced or cleared and appear with the next viewer.sync()
    """
    if viewer is None:
        return
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        for pos, rgba, alpha in goals[:scn.maxgeom]:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.full(3, half_size + GOAL_MARGIN),
                pos=np.asarray(pos, dtype=float),
                mat=np.eye(3).flatten(),
                rgba=np.array([*rgba[:3], alpha], dtype=np.float32),
            )
            scn.ngeom += 1

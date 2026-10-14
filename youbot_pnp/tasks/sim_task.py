import time

import numpy as np

from youbot_pnp.robot.robot import YouBot

VIEWER_FPS = 60  # viewer.sync() waits for a screen refresh, so only sync at frame rate


class SimTask:
    """
    base for the tasks: steps the simulation and keeps the passive viewer at real time
    """
    def __init__(self, robot: YouBot, viewer=None):
        self.robot = robot
        self.viewer = viewer  # mujoco passive viewer, or None to run headless

        self._last_sync_sim = -np.inf
        self._last_sync_wall = time.time()

    def sync_viewer(self):
        """redraw at VIEWER_FPS and keep the simulation at real time while watching"""
        if self.viewer is None or not self.viewer.is_running():
            return
        data = self.robot.data
        sim_elapsed = data.time - self._last_sync_sim
        if sim_elapsed < 1.0 / VIEWER_FPS:
            return
        ahead = sim_elapsed - (time.time() - self._last_sync_wall)
        if 0 < ahead < 1.0:
            time.sleep(ahead)
        self.viewer.sync()
        self._last_sync_sim = data.time
        self._last_sync_wall = time.time()

    def run_for(self, duration):
        """step the simulation for a duration in seconds (independent of the timestep)"""
        for _ in range(max(1, round(duration / self.robot.model.opt.timestep))):
            self.robot.step()
            self.sync_viewer()

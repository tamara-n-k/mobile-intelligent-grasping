import imageio
import mujoco


class VideoRecorder:
    """renders offscreen frames at a fixed fps (in sim time) and writes them to an mp4"""
    def __init__(self, model, camera, path, fps=30, width=1280, height=720):
        # the offscreen buffer defaults to 640x480, so it has to be enlarged for HD
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera
        self.fps = fps
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        self._last_frame_time = -float("inf")

    def capture(self, data):
        if data.time - self._last_frame_time < 1.0 / self.fps:
            return
        self.renderer.update_scene(data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self._last_frame_time = data.time

    def close(self):
        self.writer.close()
        self.renderer.close()

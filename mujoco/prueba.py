import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_link_5")
p_body = data.xpos[body_id]
R_body = data.xmat[body_id].reshape(3, 3)

# valores que copiaste con "Copy camera" (mundo)
p_cam_world = np.array([0.160, 0.011, 0.534])
x_world = np.array([-0.329, -0.944, 0.000])
y_world = np.array([0.467, -0.162, 0.869])

p_local = R_body.T @ (p_cam_world - p_body)
x_local = R_body.T @ x_world
y_local = R_body.T @ y_world

print("pos:", p_local)
print("xyaxes:", np.concatenate([x_local, y_local]))
import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Cargar modelo (asegúrate de incluir el robot dentro del entorno o incluirlo en el XML)
# Si agrupaste todo en un archivo principal:
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Inicializar visor interactivo
viewer = mujoco.viewer.launch_passive(model, data)

# Indexar actuadores por nombre para fácil acceso
actuator_names = [model.actuator(i).name for i in range(model.nu)]
def set_actuator(name, value):
    if name in actuator_names:
        idx = model.actuator(name).id
        data.ctrl[idx] = value

# Parámetros del ciclo de control
dt = model.opt.timestep
phase = 0
phase_timer = 0.0

# Posiciones de descanso y agarre para las articulaciones del brazo
ARM_HOME = [0.0, 0.4, -0.8, 0.4, 0.0]
ARM_PICK = [0.0, 1.2, -1.5, 0.8, 0.0]  # Ajuste de inclinación hacia el suelo

GRIPPER_OPEN = 0.0115
GRIPPER_CLOSE = 0.0000

while viewer.is_running():
    step_start = time.time()
    
    # Obtener posición actual del cubo azul directamente desde el entorno
    blue_cube_id = model.body("cube_blue").id
    cube_pos = data.xpos[blue_cube_id]  # [x, y, z]

    # Posición actual de la base del robot (base_link)
    base_id = model.body("base_link").id
    base_pos = data.xpos[base_id]
    
    # -------------------------------------------------------------
    # MÁQUINA DE ESTADOS SECUENCIAL
    # -------------------------------------------------------------
    
    # FASE 0: Posición Inicial del Brazo y Apertura de Pinza
    if phase == 0:
        for i, val in enumerate(ARM_HOME):
            set_actuator(f"arm{i+1}", val)
        set_actuator("gripper_l", GRIPPER_OPEN)
        set_actuator("gripper_r", GRIPPER_OPEN)
        
        phase_timer += dt
        if phase_timer > 2.0:
            phase = 1
            phase_timer = 0.0
            print("[FASE 1] Moviendo base hacia el cubo azul...")

    # FASE 1: Navegación de la Base hacia el Cubo (Control P simple)
    elif phase == 1:
        # Calcular error de posición (objetivo: frenar a ~0.2m antes del cubo para alcanzarlo con el brazo)
        target_x = cube_pos[0] - 0.22
        target_y = cube_pos[1]
        
        err_x = target_x - base_pos[0]
        err_y = target_y - base_pos[1]
        dist = np.sqrt(err_x**2 + err_y**2)
        
        # Mapeo a comandos de velocidad mecanum/omnidireccional simple
        vx = np.clip(3.0 * err_x, -4.0, 4.0)
        vy = np.clip(3.0 * err_y, -4.0, 4.0)
        
        # Control simple de ruedas para desplazamiento 2D
        set_actuator("wheel_fl", vx - vy)
        set_actuator("wheel_fr", vx + vy)
        set_actuator("wheel_bl", vx + vy)
        set_actuator("wheel_br", vx - vy)
        
        if dist < 0.03:
            # Frenar ruedas
            for w in ["wheel_fl", "wheel_fr", "wheel_bl", "wheel_br"]:
                set_actuator(w, 0.0)
            phase = 2
            phase_timer = 0.0
            print("[FASE 2] Bajando brazo hacia el cubo...")

    # FASE 2: Bajar el brazo hacia la ubicación del cubo
    elif phase == 2:
        for i, val in enumerate(ARM_PICK):
            set_actuator(f"arm{i+1}", val)
            
        phase_timer += dt
        if phase_timer > 3.0:
            phase = 3
            phase_timer = 0.0
            print("[FASE 3] Cerrando pinza...")

    # FASE 3: Cerrar la pinza para sujetar el cubo
    elif phase == 3:
        set_actuator("gripper_l", GRIPPER_CLOSE)
        set_actuator("gripper_r", GRIPPER_CLOSE)
        
        phase_timer += dt
        if phase_timer > 1.5:
            phase = 4
            phase_timer = 0.0
            print("[FASE 4] Elevando el cubo a la posición inicial...")

    # FASE 4: Elevar el brazo manteniendo la pinza cerrada
    elif phase == 4:
        for i, val in enumerate(ARM_HOME):
            set_actuator(f"arm{i+1}", val)
            
        set_actuator("gripper_l", GRIPPER_CLOSE)
        set_actuator("gripper_r", GRIPPER_CLOSE)

    # Paso de simulación física en MuJoCo
    mujoco.mj_step(model, data)
    viewer.sync()

    # Mantener tasa de refresco en tiempo real
    time_until_next_step = dt - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
youbot_description
==================

Robot description in form of URDF files and meshes

## Setup

The MuJoCo tooling in this repo (viewer, mesh conversion, rendering) and the
`youbot_pnp` pick-and-place package run in a conda environment named `hrl`.
Python must be 3.10–3.13, since OMPL has no wheels for 3.14 yet.

```
conda env create -f environment.yml
conda activate hrl
```

Without conda, a plain virtualenv works too:

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [`requirements.txt`](requirements.txt): exact, tested package versions. It also installs
  `youbot_pnp` in editable mode, so it can be imported from anywhere and code changes apply
  without reinstalling.
- [`pyproject.toml`](pyproject.toml): the `youbot_pnp` package definition and its minimum
  dependencies.
- [`environment.yml`](environment.yml): the Python version, plus `requirements.txt`.

To update an existing `hrl` env after `requirements.txt` changed:

```
conda activate hrl
pip install -r requirements.txt
```

Quick check that everything works:

```
python youbot_pnp/scripts/check_ik.py
python youbot_pnp/scripts/view_scene.py
```

## MuJoCo Model

Alongside the ROS URDF/xacro description, this repo has a MuJoCo model of the
youBot (base + arm + gripper), built from the same meshes, under
[`mujoco/`](mujoco):

- [`mujoco/youbot.xml`](mujoco/youbot.xml) — the robot alone: full-detail
  meshes for visual geoms, decimated `_convex` meshes for collision, and a
  freejoint base so it can drive and rest on the ground under physics.
- [`mujoco/environment.xml`](mujoco/environment.xml) — a reusable ground
  plane + lighting fragment.
- [`mujoco/scene.xml`](mujoco/scene.xml) — combines the two via MJCF
  `<include>`; this is the file to load for simulation.

```
conda activate hrl
python -m mujoco.viewer --mjcf=mujoco/scene.xml
```

![youBot rendered in MuJoCo](docs/youbot_render.png)

## Simulation Setup & Technical Decisions

During the integration and tuning process in MuJoCo, several physical and technical challenges were resolved:

  1. Wheel-Ground Physics & Base Stabilization

**Initial issue:** Originally, the wheels penetrated the ground plane. Using a standard freejoint allowed the robot to rest on the surface, but introduced an undesirable constant physical drift in space.
**Problem:** Having an unpredictable spatial drift (varying in magnitude and direction) degraded calculation accuracy for kinematics and planning experiments.
**Solution:** We removed the freejoint and addressed the root cause: the wheel collision geometries overlap with the chassis base by approximately 3 cm. Without explicit contact exclusions, the physics solver continuously attempts to resolve this internal penetration by spinning the wheels. Adding explicit contact exclusion pairs (<exclude>) between the chassis base and each wheel stabilized the physics completely without requiring a floating base.

  2. Environment & Table Setup for Cube Manipulation
     
**Reachability testing:** Initially, target cubes were spawned at arbitrary locations and varying heights to verify that the robot's workspace and arm kinematics could reach all target coordinates.
**Optimized stage:** Once reachability was confirmed across all points, a dedicated table with specific dimensions was designed and introduced to the scene. Placing all targets on this fixed-height surface simplifies training/learning workflows and allows using a static external camera with a reliable line of sight.

  3. On-Robot Camera Setup (Gripper Monitoring)
  
**Purpose:** To verify visual contacts between the end-effector (gripper) and the target cubes in real-time.
**Mounting:** An extended tripod mount was added directly onto the robot structure, placing an onboard camera at an optimal vantage point to observe the gripper fingers and grasping interactions without visual obstruction.

<img width="968" height="757" alt="Screenshot From 2026-09-16 03-24-47" src="https://github.com/user-attachments/assets/4631f37a-28c1-4b0a-be9f-ac799cdc79f9" />

<img width="968" height="757" alt="Screenshot From 2026-09-16 03-25-03" src="https://github.com/user-attachments/assets/faed3f64-91f4-4d7d-86d6-7287257c2671" />

<img width="968" height="757" alt="Screenshot From 2026-09-16 03-25-14" src="https://github.com/user-attachments/assets/8909b438-da30-46f9-9ff0-bfc54eeeba78" />


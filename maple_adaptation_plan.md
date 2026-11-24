# Adapting MAPLE to the youBot pick-and-place

## Overview

In this project we adapt the **MAPLE** architecture to our pick-and-place task for the KUKA youBot. We first implemented the task using a Finite State Machine (FSM). This implementation is being extended to a Hierarchical Reinforcement Learning (HRL) architecture, taking Nasiriany et al. as a reference:

https://arxiv.org/pdf/2110.03655

### MAPLE architecture

1. The policy is split into two components:
   - **Task policy:** selects which primitive to execute next.
   - **Parameter policy:** generates the continuous arguments for the selected primitive, such as a Cartesian target and grasp yaw.
   - Both policies are trained together using SAC.

2. **Affordance reward:** provides a learning signal when the selected parameters are close to useful regions. For example, grasp targets close to a cube receive a higher affordance score, making exploration more feasible.

3. **Primitive library:** provides reusable actions such as grasp, reach and release, each with a common execution interface.

4. **Gymnasium environment:** wraps the robot simulation, primitives, observations, actions and reward computation.

## What we take from MAPLE and what we change

We keep the existing FSM handlers and motion-planning code as **low-level primitives**, while learning **which primitive to execute next and where**.

Instead of following a hard-coded sequence as in `pick_and_place_fsm.py`, the HRL policy determines the sequence of primitives and their parameters.

```text
observation ──► task policy ──► which primitive? (grasp / reach / open)
             └─► parameter policy ──► where? (x, y, z, yaw in [-1, 1])
                                                |
                                                v
                                  primitive.execute()
                                     IK + RRT + servo
                                                |
                                                v
                                  task reward + affordance
```

One RL step corresponds to one complete primitive execution, not one physics step.

The original MAPLE implementation is available in the robosuite fork:

https://github.com/ARISE-Initiative/robosuite/tree/maple

---

## Primitives

The primitive library provides the low-level actions available to the HRL agent. Each primitive encapsulates the existing motion-planning and robot-execution pipeline and exposes a common interface:

```python
success, sim_time = primitive.execute(target)
```

One RL action executes one complete primitive rather than one physics step.

### Primitive library

| Primitive | Parameters | Main behavior | Success condition |
|---|---|---|---|
| `grasp(x, y, z, yaw)` | 4 | Lift if necessary → open → approach (RRT) → descend (linear) → close → lift | Both fingers establish contact with the selected cube, and the cube remains held after lifting. |
| `reach(x, y, z)` | 3 | Lift if necessary → compute a valid placement configuration → approach above target → descend linearly | Motion succeeds and, when carrying a cube, both fingers remain in contact with it. |
| `open()` | 0 | Open fingers → retract the arm to the previously validated placement approach configuration | The cube is released, with no remaining finger contact, after retraction. |

### Grasp

`Grasp` reuses the existing grasp-planning pipeline:

- Selects the cube associated with the target.
- Computes a valid grasp configuration using grasp sampling and inverse kinematics.
- Plans and executes the approach.
- Descends linearly towards the cube.
- Closes the fingers.
- Lifts the cube.
- Uses `finger_contacts` to verify that the cube is actually held.

A successful motion alone is not sufficient: the primitive must establish a valid grasp.

The primitive uses the target coordinates to select the relevant cube, then reads its exact pose from the MuJoCo simulation for grasp sampling and motion planning.

### Reach

`Reach` moves the robot towards a Cartesian target, normally the goal position of the currently held cube.

It uses:

```python
find_valid_placement(target, current_qpos, held_cube)
```

to calculate a valid placement configuration and an approach configuration.

The primitive then:

1. Determines which cube is currently held using `finger_contacts`.
2. Computes a valid approach and placement configuration.
3. Performs an initial lift if necessary.
4. Moves above the target.
5. Descends linearly towards the placement configuration.

**Important validation improvement:**

Previously, a successful trajectory execution could cause `Reach` to return `True` without checking whether the cube was still being carried.

The primitive now verifies finger contact after the movement:

```python
if held is not None:
    return all(finger_contacts(self.robot, held))
```

If the cube is no longer held, `Reach` returns `False`, even if the arm trajectory itself was executed successfully.

This distinction is essential because successful trajectory execution does not guarantee successful object transport.

**Recovery behavior:**

`Reach` overrides the default recovery behavior and does not open the gripper when a movement fails.

This is important because the default `Primitive._recover()` opens the fingers. Applying that behavior to a failed `Reach` while carrying a cube could unintentionally drop the object.

### Open

`Open` releases the currently held cube and retracts the arm to a safe approach configuration.

The primitive uses the placement approach configuration calculated by `Reach` and stored in a shared context between primitives.

Its current execution sequence is:

1. Checks whether a cube is being held.
2. Opens the gripper.
3. Retrieves the validated placement approach configuration from the shared context.
4. Retracts the arm to that configuration using a linear movement, while keeping the gripper open.
5. Checks that neither finger remains in contact with the cube.
6. Returns success only if the retraction and release verification succeed.

If no cube is held, the gripper is opened but the primitive returns `False`, because no release operation has been completed.

**Important release validation improvement:**

An earlier implementation checked whether the cube had been released immediately after opening the gripper. In some scenarios, residual contact with one finger remained briefly after opening. This caused `Open` to return `False` before the arm could retract.

The current implementation performs the retraction before checking the final release state. This allows the arm to complete its movement even when residual finger contact remains immediately after opening.

`Open` also overrides `_recover()` to avoid unnecessarily repeating the opening action.

This makes the release result observable and meaningful to the learning algorithm.

### Initial lift and collision avoidance

Both `Grasp` and `Reach` use an initial vertical lift when the gripper is empty and the TCP is close to a cube.

If the gripper is empty and the TCP is within approximately 3 cm of a cube, the arm first moves vertically upwards by `APPROACH_HEIGHT`, using a linear path and ignoring that cube.

This avoids moving sideways directly from a low pose next to a placed cube.

The behavior was introduced because collision-checked sideways RRT movements could still knock a cube.

Observed displacement:

- Without the initial lift: more than 5 mm in 9/30 test runs.
- With the initial lift: at most 0.53 mm in the same type of tests.

The initial lift is therefore retained as part of the primitive execution logic.

### Parameters and workspace

Parameters are normalized to `[-1, 1]` and mapped to the robot's table workspace.

Example workspace:

- x ∈ [0.82, 0.98] m
- y ∈ [0.22, 0.48] m
- z ∈ [0.13, 0.16] m

The workspace should be tuned using IK reachability checks.

Yaw is wrapped to `[0, π/2)`, following `_object_yaw`, because the cube is symmetric and `arm_joint_5` cannot turn fully.

### Snapping: rough parameters from the policy, exact positions from the simulator

The cube positions and goals are **inputs** to the policy (the observation), not its output. The parameters are the policy's **output**, and that output is never exact:

```
simulator ──exact──▶ observation ──▶ neural network ──▶ action (parameters) ──▶ primitive
                     (cube positions,                    (primitive choice +
                      goals, TCP, ...)                    x, y, z, yaw)
                      exact numbers                       the network's guess
```

1. **Observation (exact):** `_obs()` in `rl/env.py` reads cube positions, goals, TCP pose and so on from MuJoCo.
2. **Policy (learned, approximate):** the SAC network maps the observation to 7 numbers in `[-1, 1]`: 3 for the primitive choice and 4 for x, y, z, yaw. It does not copy a cube position into its output; it has to *learn* to produce something close to it.
3. **Action → workspace:** `_to_workspace` scales the 4 parameters to metres. The result is the `target` that the primitive receives.
4. **Primitive (exact again):** if `target` is close enough to a meaningful point, the primitive replaces it with that point's exact position from the simulator. This is the **snap**.

The policy's output is imprecise for three reasons:

- **Approximation error:** a 256×256 MLP never fits the mapping exactly.
- **Exploration noise:** during training SAC samples its action from a Gaussian around the network output, on purpose.
- **Early training:** at the start the parameters are essentially random.

The gripper, on the other hand, needs millimetre precision: it opens only 22.8 mm around an 18 mm cube, and the yaw has to match. The snap splits the work: **the policy decides *which* cube or goal and roughly *where*, and the simulator supplies the precision.**

**Grasp (implemented).** `Grasp._run` takes the unplaced cube nearest to `target` in xy. If it is within `SNAP_TOL = 0.02` m, the grasp is planned to that cube's exact position and yaw. If not, the primitive returns `False` without moving.

**Reach (planned, Run 9 in `sac_training_report.md`).** If a cube is held and `target` is within `REACH_SNAP = 0.03` m in xy of that cube's goal, `target` is replaced by the exact goal (z included). Otherwise `reach` goes to `target` as given. `Reach` gets the goal through a `goal_of` callback from the env (`goal_of=lambda c: self.cfg.goals[c]`), the same way it already gets `held` and `unplaced`, because `cfg` changes at every reset. `PLACE_TOL` stays at 1 cm. Without the snap, a random reach places the cube only about 0.25 % of the time; with the 3 cm snap circle, about 7 %.

**Why a tolerance, and not always snap to the nearest point?** The tolerance decides *whether* the snap happens:

- **The parameter keeps its meaning.** A primitive means "do X *at this location*". A grasp aimed at empty table fails, and that failure is an informative signal. If every grasp snapped, every grasp would succeed, and the policy would never learn where the cubes are.
- **Reach must be able to go elsewhere.** In `occupied` and `swap`, a cube sometimes has to go to a free buffer slot that is *not* its goal. A reach that always snapped to the goal would make that impossible. With the tolerance, it snaps only when the policy is already aiming near the goal.
- **Choosing the size.** A small tolerance keeps the action precise but is hard to hit by chance. A large one is easy to hit but can blur which of two neighbouring targets was meant. Slots are 6 cm apart, so 3 cm is about the largest reach tolerance that never confuses two neighbouring goals.

**The affordance has to agree with the snap.** The reward contains `− λ · (1 − aff)`, and for `reach` the affordance measures the distance from `target` to the goal (`aff = 1` within `tol`, then `1 − tanh(10 · (d − tol))`). With the snap, any target inside the circle gives the same perfect placement, so the affordance must not penalise it:

| Policy's target | Result with snap | aff with today's settings | Penalty |
|---|---|---|---|
| 2.5 cm off in xy, z on the goal | placed exactly | 0.80 | −0.20 |
| exactly above the goal, z = 0.16 m | placed exactly (z is overwritten) | ≈ 0.17 | −0.83 |

Therefore, together with the reach snap:

- `env._affordance` uses `tol = 0.03` for `reach`, so every target inside the snap circle gets `aff = 1`. Outside it, the penalty grows from the circle's edge and still guides the policy towards the goal.
- `aff_score` measures `reach` in xy only (`n = 2`), because the snap overwrites z.

The snap makes a good-enough reach *succeed*, and the affordance change makes it also *score* as good. (`grasp` has the same mismatch, with a 2 cm snap and a 0.5 cm affordance tolerance. It learns anyway, so it is left unchanged for now, to change one thing at a time.)

### Why reliable primitive success signals matter for RL

Primitive success is used by the environment to determine the reward and the next state of the agent.

Incorrect success signals can directly damage training:

- If `Grasp` reports success without establishing finger contact, subsequent actions may operate under an incorrect assumption about the held object.
- If `Reach` reports success after dropping a cube, the agent receives misleading feedback about its transport action.
- If `Open` reports success without an actual release, the agent cannot distinguish a completed placement from an unsuccessful one.

The current implementation makes primitive outcomes more closely reflect the physical state of the task.

It is important to distinguish between:

- **Primitive success:** whether the individual operation was completed correctly.
- **Task success:** whether the cube is released and within the placement tolerance of its goal.

A successful `Reach` does not necessarily mean that the cube has been placed. Final placement must be checked independently.

Every primitive returns `(success, sim_time)`, allowing the environment to use the result for reward computation and execution-time logging.

### Primitive integration validation

To verify that the HRL primitives preserve the behavior of the original FSM, an integration test was implemented:

`youbot_pnp/scripts/check_primitives.py`

The test executes the HRL primitive sequence independently and compares it with the original FSM implementation.

The HRL sequence is:

```text
Grasp → Reach → Open
```

Both implementations are evaluated in separate, freshly initialized simulation worlds using the same task configuration.

The test explicitly checks:

- Grasp success and contact with both fingers.
- Reach success and confirmation that the cube remains held.
- Open success and confirmation that the cube has been released.
- Final placement within `PLACE_TOL = 0.01` m.
- Final placement distance for both implementations.
- Whether the task is successfully completed.

The cube is allowed to settle before the final placement measurement.

**Important distinction:** this test executes a predefined HRL primitive sequence. It validates primitive integration and task execution, but it does not evaluate a learned high-level policy selecting primitives autonomously.

#### Initial nominal results

The original nominal integration test successfully completed the task with both implementations.

| Metric | HRL primitives | FSM |
|---|---:|---:|
| Grasp | Success | Success |
| Reach | Success | Success |
| Release | Success | Success |
| Final distance to goal | 1.66 mm | 1.66 mm |
| Cube released | Yes | Yes |
| Cube placed | Yes | Yes |

The absolute difference between the final placement distances was 0.00 mm at the reported precision.

#### Extended randomized results

The integration test was subsequently extended to support multiple task variants through the shared task generator.

Ten seeds were evaluated for each of the following variants:

| Variant | HRL | FSM |
|---|---:|---:|
| `random` | 10/10 | 10/10 |
| `occupied` | 10/10 | 10/10 |
| `swap` | 10/10 | 10/10 |
| **Total** | **30/30** | **30/30** |

Both implementations completed all 30 evaluated tasks.

These scenarios test different aspects of primitive integration:

- **Random:** different cube start configurations, orientations and goals.
- **Occupied:** a goal position is initially occupied by another cube.
- **Swap:** two cubes exchange their starting positions, requiring the use of a buffer position.

The `occupied` and `swap` scenarios are particularly useful because they introduce task configurations in which the order of operations matters.

#### Conclusions

The results provide evidence that the current primitive library can execute the tested pick-and-place configurations reliably and reproduce the task behavior of the FSM baseline.

The validation also demonstrates that the revised success checks work across more than the nominal configuration.

However, these results do not establish that a learned HRL policy can discover the appropriate primitive sequence. The tested HRL sequence is predefined.

**Current limitation:** autonomous primitive selection, learned sequencing and recovery behavior still need to be evaluated using the trained high-level policy.

---

## Environment (Gymnasium)

### Observation

The observation vector contains:

- Cube positions and yaw.
- Goal positions for each cube.
- TCP position and yaw.
- Finger opening.
- Held-cube one-hot encoding.
- Success flag of the last primitive.

### Action

The policy outputs 7 parameters in `[-1, 1]`:

```text
[grasp, reach, open, p1, p2, p3, p4]
```

1. `argmax(score)` selects the primitive.
2. The primitive uses only the parameters it needs:
   - Grasp uses p1–p4 as x, y, z, yaw.
   - Reach uses p1–p3 as x, y, z.
   - Open uses none.
3. Unused parameters are ignored.
4. Parameters are mapped to the workspace.

### Reward

The reward is computed using the state after the primitive execution.

| Component | Value | Purpose |
|---|---:|---|
| Cube on goal | +1 per cube | Main progress signal |
| Cube grasped | +0.25 | Intermediate progress, only for cubes not yet placed |
| Distance held cube to its goal | `1 - tanh(10d)` | Dense guidance while carrying |
| All cubes placed | +10 | Clear success signal; episode ends |
| Primitive failed | -0.1 | For example, no IK solution, no RRT path or blocked arm |
| Affordance penalty | `-λ(1-aff)` | Evaluates the selected parameters before robot movement |

The affordance score is computed as follows:

- **Grasp keypoints:** all cubes not yet placed.
- **Reach keypoint:** the goal of the currently held cube.
- If the target is within the tolerance around a keypoint, `aff = 1`.
- Otherwise, `aff = 1 - tanh(10 × distance)`.

Choose λ between 1 and 2 initially, so that the affordance penalty does not overpower the +1 reward for placing a cube. MAPLE uses λ = 15, but this value may be too high for our reward scale.

**Important:** potential-based reward shaping should also be evaluated. The potential function can provide a more consistent progress signal across successive primitive executions.

### Episode

- Terminates when all cubes are placed.
- Truncates after `5 × n_cubes` primitive steps.

---

## Task Design

The FSM and HRL policy use the same low-level primitives. Therefore, anything that breaks a primitive, such as an unreachable target or an IK failure, affects both approaches.

HRL can provide an advantage when the **order of the primitives** or the **reaction to what happened** matters.

The task variants are therefore designed to test sequencing and recovery, not merely motion planning.

### 1. Variation

Needed for RL, while the FSM should still be able to handle it:

- Random cube start positions and yaw within the reachable area.
- Random goals for each cube.

### 2. Sequencing problems

These are the tasks where HRL may provide an advantage:

- **Occupied goal slot:** the goal of cube A is where cube B is currently located. B must be moved away first.
- **Swap:** A's goal is B's start and B's goal is A's start. One cube needs to be moved to a buffer position first.

An FSM can only solve these tasks with a hand-coded rearrangement planner, which is the type of sequencing problem HRL is intended to learn.

### 3. Disturbances

Recovery matters in these scenarios:

- **Grasp slip:** the cube drops during transport with probability p (10–20%).
- **Cube displacement:** a cube is moved a few millimeters after the FSM has computed its configurations.
- **Observation noise:** 2–5 mm noise on cube positions.

---

## Shared Task Generator

A shared task generator has been implemented:

`youbot_pnp/tasks/task_generator.py`

It maps a seed and a task variant to cube start poses and goals.

The supported variants are:

```python
VARIANTS = ["nominal", "random", "occupied", "swap"]
```

The generator uses a fixed set of table slots:

```python
SLOTS = np.array([
    [x, y]
    for x in (0.93, 0.86)
    for y in (0.26, 0.32, 0.38, 0.44)
])
```

The current implementation defines:

- `nominal`: fixed start and goal slots matching the original scene.
- `random`: randomly selected start slots and goals on free slots.
- `occupied`: a goal is another cube's starting position, creating an ordering constraint.
- `swap`: two cubes exchange their starting slots, requiring a buffer position.

Randomized variants use:

- `START_JITTER = 0.01` m.
- Random yaw in `[0, π/2]`.
- A fixed random seed for reproducibility.

The generator currently supports one, two or four cubes depending on the selected variant. The `occupied` and `swap` variants require at least two cubes.

**Current limitation:** disturbance settings are represented in the `TaskConfig` data structure but are not yet implemented as active simulation disturbances. The current generator also uses a fixed set of eight slots rather than a larger set of additional buffer positions.

### Before the HRL implementation

The intended evaluation procedure is:

1. Generate shared task configurations using `task_generator.py`.
2. Run the FSM on approximately 50 seeds per variant.
3. Log success rates and reasons for failure.
4. If failures are caused by primitive limitations (no IK solution, unreachable targets), shrink the sampling area.
5. If failures are caused by ordering or missing recovery, retain the variant because this is where HRL may help.
6. Freeze the task definition before final training and comparison.

---

## Implementation Steps

### 1. Timing Check

The timing check estimates how long training will take and identifies the computational bottlenecks.

MAPLE collects 30,000 primitive steps before training starts, then adds 3,000 exploration and 3,000 evaluation steps per epoch.

A single grasp contains:

- RRT planning.
- Execution at maximum joint velocity.
- Final `servo_to`.
- Gripper closing time.
- Linear descent and lift with collision checks.

The timing check helps determine whether some methods need simplification or whether primitives should be designed more coarsely.

Potential optimizations:

- Run headless (`viewer=None`); `sync_viewer` slows simulation down to real time.
- Try `find_linear_path` first and RRT only as fallback.
- Lower planning time if appropriate.
- Use parallel environments.

**Done when:** wall time per primitive and the main computational bottleneck are known.

#### Results (2026-11-12, completed)

Script: `youbot_pnp/scripts/check_timing.py`

Configuration: one cube, 20 seeds, headless. Primitives were executed through `env.step()` with hand-picked parameters (true cube position and goal).

| Primitive | Success | Wall mean (max) | Sim mean | Sim / wall |
|---|---:|---:|---:|---:|
| Grasp | 20/20 | 0.30 s (0.54 s) | 7.60 s | 25× |
| Reach | 20/20 | 0.17 s (0.25 s) | 3.31 s | 20× |
| Open | 20/20 | 0.01 s (0.02 s) | 0.50 s | 50× |

Where the time goes (exclusive wall time, grasp / reach):

- Physics (`mj_step`): 47% / 41%.
- Collision checks (`collision_free`, mostly inside RRT and linear path checks): 45% / 48%.
- Everything else (OMPL itself, IK, grasp sampling, simulation-loop overhead, observation and reward): under 10%.

Conclusions:

- No simplification of the primitives is currently needed.
- `check_resolution=0.003` accounts for approximately half of the collision-checking cost, but is retained because it prevents the gripper from skipping through cubes.
- Approximately 0.16 s per primitive, or 22,000 primitives/hour per environment.
- The 30k warm-up steps take at most approximately 1.4 hours on one environment, and less in practice because random primitives often fail before the robot moves (approximately 0.06 s per step in `check_rl_env.py`).
- With `SubprocVecEnv` (4 environments), execution is approximately 3–3.5× faster.
- SAC network updates take only a few milliseconds on CPU, so more than one gradient step per environment step (`gradient_steps > 1`) is affordable.

### 2. Primitives

Each primitive is a small class that turns parameters into a robot motion.

Implemented:

- `grasp(x, y, z, yaw)`: lift first if the empty gripper is around a cube, then execute the existing open, approach, descent, close and lift sequence.
- `reach(x, y, z)`: lift first if necessary, then compute a valid placement configuration, move above the target and descend linearly.
- `open()`: open the gripper and retract the arm to the validated placement approach configuration.
- The policy provides coordinates rather than cube names.
- The planner uses the target position to identify the relevant cube during grasp, or the currently held cube during reach.
- The held cube is determined using `finger_contacts`.
- If IK or RRT fails, the primitive returns `success = False`.
- `Reach` verifies that a carried cube remains held after transport.
- `Reach` does not open the gripper during recovery.
- `Open` verifies that the cube has actually been released after retracting the arm.
- `Open` does not repeat the opening action during recovery.
- A shared context transfers the validated placement approach configuration from `Reach` to `Open`.

**Integration test:**

`youbot_pnp/scripts/check_primitives.py`

The script compares the predefined HRL primitive sequence with the original FSM implementation.

Results:

- Nominal task: both implementations pass.
- Random: 10/10 successful tasks for both implementations.
- Occupied: 10/10 successful tasks for both implementations.
- Swap: 10/10 successful tasks for both implementations.
- Overall: 30/30 successful tasks for each implementation.

**Status: primitive integration validation completed for the three randomized variants.**

### 3. Affordance

Create:

```python
aff_score(primitive, target, keypoints, threshold) -> [0, 1]
```

Use `_compute_aff_reward_and_success` in robosuite's `skills.py` as a reference.

Keypoints:

- Grasp: unplaced cubes.
- Reach: goal of the held cube.
- Open: no spatial keypoint; affordance is always 1.

**Done when:**

- Unit tests pass.
- A target exactly on a cube gives an affordance score of 1.
- A target 25 mm away gives approximately 0.8 with a 5 mm tolerance.

### 4. Design Environment

Gymnasium environment around `PickAndPlace` and the primitive library.

- Write observation and task reward as separate methods.
- `reset()`: reset the robot, generate cube poses and goals, and run the simulation for 0.5 s so the cubes settle.
- `step(action)`:
  1. Select the primitive using `argmax`.
  2. Compute the affordance score before robot movement.
  3. Execute the primitive.
  4. Compute the reward.
  5. Generate the next observation.
  6. Determine `terminated` and `truncated`.
- Normalize observations to approximately `[-1, 1]`.
- Include success, affordance score and selected primitive in `info` for logging.

**Done when:**

- Gymnasium's `check_env` passes.
- 100 episodes with random actions run without crashing.

### 5. Training: Quick Version

Use SB3 SAC with a flat 7-dimensional action space. The `argmax` of the first three values selects the primitive.

- Start with one cube.
- Use parallel environments with `SubprocVecEnv`.
- Log success rate, affordance score, primitive selection frequency and failure rate.

**Target:** more than 80% success on the one-cube task.

If the agent does not learn, inspect reward and observation design before increasing training time.

#### Previous training experiments

Several training runs were performed to investigate reward design and policy learning.

**Initial state-based reward:**

- The agent achieved 0% task success at 100k steps.
- It learned to repeatedly select grasp and reach actions without completing the task.
- This indicated that the reward could be exploited without making actual placement progress.

**Potential-based reward:**

- Improved evaluation success to approximately 55% at 100k steps.

**Subsequent training run:**

At 100k steps:

| Metric | Result |
|---|---:|
| Evaluation success rate | 70% |
| Evaluation reward | 7.79 ± 4.90 |
| Mean episode length | 3.60 |
| Affordance | 0.991 |
| Successful grasp rate | 0.941 |
| Successful reach rate | 1.000 |
| Successful open rate | 1.000 |
| Failure rate | 0.0208 |
| Grasp selection frequency | 0.354 |
| Reach selection frequency | 0.333 |
| Open selection frequency | 0.313 |

Run:

`runs/sac_1cube_random_s0_1119_0124`

The agent learned a meaningful primitive sequence, but the overall task success rate remained below the target of 80%.

**Checkpoint evaluation:**

A separate evaluation of checkpoints at 25k, 50k, 75k and 100k steps used five evaluation seeds (10000–10004).

| Training steps | Successful episodes | Observation |
|---|---:|---|
| 25k | 0/5 | The agent had not yet learned to complete the task. |
| 50k | 1/5 | The agent began discovering the correct sequence. |
| 75k | 5/5 | The agent completed all five episodes, including a recovery after a failed reach. |
| 100k | 4/5 | Overall behavior remained strong, but one episode failed during release/placement or recovery. |

These results suggest that the agent progressively discovers the task sequence, but the learned behavior is not yet fully robust.

**Important distinction:** primitive-level success does not necessarily imply task-level success. For example, a successful arm trajectory does not guarantee that the cube has been transported, released and placed correctly. This motivated the additional contact and release validation implemented in the primitive library.

**Experimental qualification:** these are historical results from earlier primitive implementations. They must not be treated as results obtained with the currently validated primitive library.

The next experiment should retrain the quick SAC baseline using the current implementation and compare its performance with these previous results.

### 6. Training: MAPLE Version

Port `PAMDPPolicy` and `SACHybridTrainer` from the MAPLE repository into the project.

- Task policy: categorical choice of primitive.
- Parameter policy: one network per primitive.
- Two entropy terms, with high initial entropy over primitive selection.
- Train both policy components together.
- This does not fit directly into SB3's standard flat SAC action space, so implement a dedicated training loop using CleanRL's `sac_continuous_action.py` as a template.

The MAPLE-style implementation should preserve the distinction between discrete primitive selection and continuous primitive parameters.

**Done when:** it works on one cube and can be compared with the quick SAC implementation from Step 5.

### 7. Harder Tasks

Use a curriculum:

1. One cube.
2. Two cubes.
3. Four cubes.

Introduce task variants where the FSM may fail or require additional planning:

- Random goal permutations.
- Goal slots occupied by other cubes.
- Cube pose noise.
- Grasp slips.
- Recovery after failed primitive execution.

The objective is to test whether HRL learns sequencing and recovery behavior that is not explicitly programmed into the FSM.

### 8. Evaluation

Run the FSM and the learned policy on the same randomized tasks using identical seeds.

Metrics:

- Task success rate.
- Number of primitives per episode.
- Simulation time.
- Wall-clock execution time.
- Primitive-level success and failure rates.
- Recovery frequency.

Ablations:

- No affordance reward.
- Quick SAC vs. MAPLE SAC.
- Per-step reward vs. potential-based reward.
- Original primitive success checks vs. contact- and release-validated primitives.

The last ablation is particularly relevant because the reliability of primitive outcomes directly affects the reward and transition signals used by SAC.

---

## Hardware

- GPU: NVIDIA T400, 4 GB.
- RAM: 32 GB.
- CPU: Intel Xeon W3-2423.

---

## Current Project Status and Next Steps

| Component | Status |
|---|---|
| Original FSM pick-and-place | Implemented |
| Shared task generator | Implemented |
| Primitive library (Grasp, Reach, Open) | Implemented |
| Initial vertical lift | Implemented and tested |
| Reach contact validation | Implemented |
| Safe Reach recovery | Implemented |
| Open release validation | Implemented |
| Open retraction to validated approach configuration | Implemented and tested |
| Nominal HRL–FSM integration test | Passed |
| Random primitive integration testing | Passed (10/10) |
| Occupied primitive integration testing | Passed (10/10) |
| Swap primitive integration testing | Passed (10/10) |
| Overall randomized integration testing | Passed (30/30 for HRL and FSM) |
| Primitive timing analysis | Completed |
| Initial SAC training | Completed |
| More than 80% SAC success target | Not yet achieved |
| Retraining with the current primitive implementation | Pending |
| Gymnasium validation with updated primitives | Pending |
| Disturbance implementation | Pending |
| Multi-cube learned HRL training | Pending |
| MAPLE-specific hybrid policy and trainer | Pending |
| Full FSM vs. learned HRL evaluation | Pending |

### Immediate next steps

1. Review and commit the validated primitive implementation and integration tests.
2. Verify that the Gymnasium environment correctly handles the updated `Open` behavior, primitive success signals and shared context.
3. Run environment validation, including `check_env` and random-action episodes.
4. Retrain the quick SAC baseline using the current primitive implementation.
5. Compare the new learning curves and task success against the historical training runs.
6. Evaluate whether the learned policy can complete the one-cube task reliably.
7. Once the one-cube baseline is stable, introduce multi-cube sequencing tasks.
8. Implement the MAPLE-specific hybrid policy and trainer.
9. Compare the quick SAC and MAPLE-style approaches against the FSM using identical task configurations and evaluation seeds.
10. Evaluate disturbances, recovery behavior and generalization to unseen configurations.

**The current stage has established a reliable primitive library across several task configurations. The next objective is to move from validating a predefined primitive sequence to training and evaluating a policy that can autonomously select primitives and their parameters.**
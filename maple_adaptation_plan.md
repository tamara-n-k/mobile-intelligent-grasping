# Adapting MAPLE to the youBot pick-and-place

## Overview

In this project we adapt the **MAPLE** architecture for our pick and place task for the Youbot. We have first implemented the task with a Finite State Machine. This will be extended to a Hierarchical Reinforcement Learning architecture taking Nasiriany et al. as a reference: https://arxiv.org/pdf/2110.03655

**MAPLE architecture**: 
1. the policy is split into 2:
- task policy: which primitive to run next
- parameter policy: continuous argument for the primitive: Cartesian target, grasp yaw
- the policies are trained together with SAC
2. affordance reward: gives a bonus when the parameters land near a useful region: grasp position -> reward makes exploration feasible
3. build primitive library: reach, grasp, release..., each primitive has execute interface
4. use gymnasium env

## What we take from MAPLE and what we change
We will keep the FSM handlers as **primitives** and learn **which primitive runs next and where**. So instead of following a hard-coded sequence as in `pick_and_place_fsm.py` we will train a policy which determines the order of the primitives.

```
observation ──► task policy ──► which primitive?   (grasp / reach / open)
            └─► param policy ─► where?             (x, y, z, yaw in [-1, 1])
                                   │
                                   ▼
                  primitive.execute()  →  IK + RRT + servo (existing code)
                                   │
                                   ▼
                  reward = task reward − affordance penalty
```

One RL step = one whole primitive (not one physics step).

**MAPLE** code is published on a fork of the robosuite github: https://github.com/ARISE-Initiative/robosuite/tree/maple


## Primitives

| Primitive              | Params | Built from (existing code)                                         |
|------------------------|--------|--------------------------------------------------------------------|
| `grasp(x, y, z, yaw)`  | 4      | open → approach (RRT) → descend (linear) → close → lift, `finger_contacts` check |
| `reach(x, y, z)`       | 3      | IK at top-down pose → approach above → descend (linear)            |
| `open()`               | 0      | open fingers → retract up (`_handle_release` + `_handle_retract_place`)                                        |

- Params in `[-1, 1]` are mapped to the table workspace, e.g. x ∈ [0.82, 0.98], y ∈ [0.22, 0.48],
  z ∈ [0.13, 0.16] (tune with an IK reachability check).
- Yaw is wrapped to `[0, π/2)` like `_object_yaw`, because the cube is symmetric and `arm_joint_5` cannot turn fully.
- Every primitive returns `(success, sim_time)`.

## Environment (Gymnasium)

- **Observation (what the policy sees):** The observation vector will contain: cube positions + yaw, goal position per cube, TCP position + yaw, finger opening,
  held-cube one-hot, success flag of the last primitive
- **Action (what the policy outputs):** 7 parameters in range [-1, 1]:
[grasp, reach, open, p1, p2, p3, p4]
1. argmax(score) picks the primitive
2. the primitive only uses the parameter it needs:
   - grasp uses p1–p4 as x, y, z, yaw
   - reach uses p1–p3 as x, y, z
   - open uses none The unused params are ignored.
3. the params are mapped to the workspace
- **Reward** (rated on the state after the primitive):

|Part|Value|Purpose|
|-----|-----|------|
|Cube on Goal|+1 per cube|Main progress signal|
|Cube grasped| +0.25| intermediate step: only for cubes not yet placed|
|Distance held cube to its goal| 1- tanh (10d) |dense guidance while carrying|
|All cubes placed| +10, episode ends| clear success signal|
primitive failed| -0.1| e.g. no IK solution, no RRT path, arm blocked|
|Affordance penalty| -λ (1-aff)| judges only the chosen params before robot moves|
Affordance score: 
- grasp keypoints: all cubes not yet placed
- reach keypoint: goal of the held cube
- if target within tolerance around keypoint, aff = 1, else aff = 1 - tanh(10 distance)
- choose lambda between 1-2 so it doesnt overstimulate the +1 per placed cube (MAPLE uses λ = 15)

NOTE: check potential based scores as well

- **Episode:** 
- terminates when all cubes are placed
- truncates after 5 * n_cubes primitives 

## Task Design

The FSM and the policy use the same primitives, so anything that breaks a primitive (e.g. no IK solution) breaks both. HRL can only be better where the **order of the primitives** or the **reaction to what happened** matters. So the task variants are chosen for that, not just to make the FSM fail.

**1. Variation** (needed for RL, the FSM should still handle it):
- random cube start positions and yaw in the reachable area
- random goal for each cube

**2. Sequencing problems** (HRL can win here):
- occupied goal slot: the goal of cube A is where cube B is, B has to be moved away first
- swap: A's goal is B's start and B's goal is A's start, needs a move to a buffer spot
- an FSM can only solve this with a hand-coded rearrangement planner, which is what HRL should replace

**3. Disturbances** (recovery matters):
- grasp slip: the cube drops during transport with probability p (10–20 %)
- cube is moved a few mm after the FSM computed its configurations
- observation noise of 2–5 mm on the cube positions


**Before the HRL implementation:**
1. write a shared task generator (`youbot_pnp/tasks/task_generator.py`): seed + variant → cube poses, goals, disturbance settings. FSM and RL env both use it
   - min spacing between cubes about 4 cm so the gripper fits
   - goals: fixed slots on the table (`PLACE_POSITIONS`) plus 2–3 free buffer spots
2. run both FSMs on about 50 seeds per variant, log success rate and the reason for each failure
3. failure caused by a primitive (no IK, unreachable) → shrink the sampling area. failure caused by order or missing recovery → keep the variant, here HRL can help
4. freeze the task definition before training

### Implementation Steps:
#### 1. Timing Check
This will tell us how long training will take: 
MAPLE collects 30000 primitive steps before training even starts, then it adds 3000 exploration and 3000 evaluation steps per epoch.
A single grasp contains:
- RRT plan
- execution at max_joint_vel
- final servo_to
- gripper close time
- linear descend and lift with collision checks
The timing check tells us where the time goes and therefore we can choose to simplify some methods etc to shorten training time. Or we might want to design the primitives more coarsely.
- run headless (viewer=None), sync_viewer slows the simulation down to real time
- possible fixes: find_linear_path first and RRT only as fallback, lower planning_time, parallel envs

Done when: we know the wall time per primitive and which part takes the most time.

#### 2. Primitives 
Each primitive is a small class that turns parameters into a motion.
- grasp(x, y, z, yaw): like _handle_open_gripper through _handle_lift_pick.
- reach(x, y, z): move above the target, then descend linearly, ignoring the held cube.
- open(): _handle_release + _handle_retract_place.
- the policy only gives coordinates, not a cube name: for the planner ignore list we take the cube nearest to the target (grasp) or the held cube (reach), the held cube comes from finger_contacts
- if IK or RRT fails, the primitive does nothing and returns success = False

Done when: grasp, reach and open with hand-picked parameters (e.g. the true cube position) give the same result as the FSM.
#### 3. Affordance:
create function, ```aff_score(primitive, target, keypoints, threshold) → [0, 1]```. Use _compute_aff_reward_and_success in robosuite's skills.py as an example

grasp keypoints: unplaced cubes
reach keypoint: goal of the held cube
open: none, always 1
Done when: unit tests pass. A target exactly on a cube gives 1, and 25 mm away gives about 0.8 (with 5 mm tolerance).

#### 4. Design Environment:
Gymnasium env around PickAndPlace and the primitives
- write obs vector and task_reward as separate methods
- reset(): robot.reset(), random cube poses and random goals, run_for(0.5) so the cubes settle
- step(action): pick the primitive with argmax, compute the affordance score before the robot moves, execute the primitive, then compute reward, obs, terminated and truncated
- normalize the obs to around [-1, 1]
- put success, aff score and the chosen primitive into info for logging

Done when: gymnasium's check_env passes and 100 episodes with random actions run without crashing.

#### 5. Training, quick version:
- SB3 SAC on the flat 7-dim action, argmax of the first 3 values picks the primitive
- start with 1 cube
- parallel envs with SubprocVecEnv
- log success rate, aff score, how often each primitive is chosen and the failure rate

Done when: more than 80 % success on 1 cube. If it does not learn, check reward and obs first.

#### 6. Training, MAPLE version:
port PAMDPPolicy and SACHybridTrainer from the maple repo into our project
- task policy: categorical choice of the primitive
- parameter policy: one network per primitive
- two entropy terms, high entropy over the primitives at the start
- this does not fit into SB3, so we write a small training loop (CleanRL's sac_continuous_action.py as template)

Done when: it works on 1 cube and we can compare it with step 5.

#### 7. Harder tasks:
- curriculum: 1 → 2 → 4 cubes
- task variants where the FSM fails: random goal permutations, goal slot occupied by another cube, cube pose noise, grasp slips
8. Evaluation:
run FSM and policy on the same randomized tasks (same seeds)
- metrics: success rate, primitives per episode, sim time and wall time
- ablations: no affordance reward, quick SAC vs. MAPLE SAC, per-step vs. potential based reward

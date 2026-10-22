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


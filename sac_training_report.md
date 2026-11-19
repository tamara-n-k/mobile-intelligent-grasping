# SAC training on 1 cube: results and next steps

Status as of 2026-11-19. Plan step 5 ("training, quick version") in [maple_adaptation_plan.md](maple_adaptation_plan.md).

**Summary:** SB3 SAC has now reached **70% evaluation success after 100k steps**, improving on the previous 55% result. During the final training iterations, the success rate fluctuated between 70% and 76%. The final evaluation achieved 70%, with a mean episode reward of 7.79 and an average episode length of 3.60 steps.

The primitive-level metrics are encouraging: reach and open both report 100% success, while grasp reports 94.1%. The affordance score is 0.991 and the failure rate is 2.08%. However, these individual primitive metrics do not yet translate into an overall task success rate above 80%.

The next priority is to investigate the gap between primitive-level and full-task success. Before changing SAC hyperparameters or launching another long training run, we should inspect the reward function, the success condition and the relationship between primitive execution and task completion.

## Setup

| | |
|---|---|
| Script | `youbot_pnp/scripts/train_sac.py`, results in `runs/<name>/` |
| Task | `YouBotPnPEnv(n_cubes=1, variant="random")`, up to 5 primitives per episode (as documented for the previous runs; confirm whether unchanged) |
| Algorithm | SB3 SAC, `MlpPolicy` [256, 256], flat 7-dim action (argmax of the first 3 values picks the primitive) |
| Training | 100k primitive steps, 8 parallel envs (`SubprocVecEnv`), 5k random warm-up steps, 1 gradient step per transition, `ent_coef="auto"` (as documented for the previous runs; confirm whether unchanged) |
| Evaluation | Every 5k steps on 20 fixed tasks (seeds 10000–10019, not used for training), deterministic policy (as documented for the previous runs; confirm whether unchanged) |
| Wall time | About 3 hours for 100k steps in this run (10,893 seconds elapsed) |

## Run 1: state-based reward → reward hacking

`runs/sac_1cube_random_s0_1117_1910`

The reward paid for the current state at every step: +1 per placed cube, and `0.25 + 1 − tanh(10 d)` while a cube is held.

| Metric (100k steps) | Value |
|---|---|
| `eval/success_rate` | **0%** |
| `eval/mean_reward` | 5.13 |
| episode length | Always 5 |
| `chosen_grasp / reach / open` | 0.2 / 0.8 / **0** |

The policy learned to grasp the cube, carry it to the goal and then keep holding it with 4 reaches, never opening. Hand-picked actions on 10 seeds confirmed the cause:

| Sequence | Return | Solved |
|---|---:|---:|
| grasp, reach ×4 (hold at the goal) | 5.31 | 0/10 |
| grasp, reach, open (place) | 11.54 | 9/10 |

Placing paid twice as much, but the policy never discovered it. A random reach lands within 1 cm of the goal only about 0.75% of the time, so the +10 for placing never appeared during exploration. Holding, meanwhile, paid a safe ~1.25 per step. Once exploration dropped (`ent_coef` 0.035), the policy stayed there.

## Run 2: potential-based reward → 55%

`runs/sac_1cube_potential_s0`

The reward now pays only for **progress**, `Φ(after) − Φ(before)`, with:

- Φ = 1 per placed cube, plus up to `0.25 + 0.5 · (1 − tanh(10 d))` for a held cube at distance `d` from its goal (at most 0.75 < 1).
- +10 when all cubes are placed, −0.1 for a failed primitive, −5 for a cube that falls off the table, −λ·(1 − aff) affordance penalty.

Holding still now earns 0. The same hand-picked check:

| Sequence | Return | Solved |
|---|---:|---:|
| grasp, reach ×4 (hold at the goal) | 0.74 | 0/10 |
| grasp, reach, open (place) | 9.90 | 9/10 |
| grasp, open, … (drop at the start) | −3.00 | 0/10 |

### Learning curve (eval on the 20 fixed tasks)

| Steps | Success | Episode length | Mean reward |
|---|---:|---:|---:|
| 5k–35k | 0% | 5.0 | −3.4 → −0.4 |
| 40k–60k | 0–10% | 5.0 | Around 0 |
| 65k–80k | 15–25% | 4.7 → 4.5 | 1.5–2.9 |
| 85k–95k | 35–45% | 4.3 → 4.1 | 3.9–5.0 |
| **100k** | **55%** | **3.9** | **6.0** |

There was no success for 35k steps, until the policy discovered "open at the goal". From then on, success rose steadily and had not levelled off when the run ended.

Final training statistics: `aff` 0.97, `success_grasp` 0.91, `success_reach` 1.0, `chosen_grasp / reach / open` 0.41 / 0.36 / 0.23, `ent_coef` 0.024.

### Failure analysis (final model, 20 eval tasks)

| Seed | Result | Final distance | Primitive sequence (reach error = distance of the reach target to the goal) |
|---|---|---:|---|
| 10000 | solved | 4.4 mm | grasp → reach (4 mm) → open |
| 10001 | solved | 9.5 mm | grasp → reach (9 mm) → open |
| 10002 | failed | 10.0 mm | grasp → reach (13 mm) → open → grasp → reach (10 mm) → *out of steps* |
| 10003 | solved | 7.9 mm | grasp → reach (9 mm) → open |
| 10004 | failed | 112 mm | grasp → open → grasp → open → grasp |
| 10005 | solved | 6.3 mm | grasp → reach (6 mm) → open |
| 10006 | solved | 7.2 mm | grasp → reach (6 mm) → open |
| 10007 | solved | 5.8 mm | grasp → reach (6 mm) → open |
| 10008 | failed | 222 mm | grasp → reach (222 mm) → open → grasp → open |
| 10009 | solved | 7.7 mm | grasp → reach (7 mm) → open |
| 10010 | failed | 4.6 mm | grasp → reach (11 mm) → open → grasp → reach (4 mm) → *out of steps* |
| 10011 | solved | 6.7 mm | grasp → reach (6 mm) → open |
| 10012 | failed | 216 mm | grasp → reach (216 mm) → open → grasp → open |
| 10013 | failed | 17.9 mm | grasp → reach (2 mm) → open (finger still touching) → grasp FAIL → grasp FAIL |
| 10014 | solved | 8.3 mm | grasp → reach (6 mm) → open |
| 10015 | failed | 223 mm | grasp → reach (223 mm) → open → grasp → open |
| 10016 | solved | 4.4 mm | grasp → reach (4 mm) → open |
| 10017 | solved | 6.7 mm | grasp → reach (6 mm) → open |
| 10018 | solved | 9.5 mm | grasp → reach (8 mm) → open |
| 10019 | failed | 11.2 mm | grasp → reach (13 mm) → open → grasp → reach (11 mm) → *out of steps* |

Four failure causes:

1. **Reach is barely precise enough (main cause).** A cube counts as placed within `PLACE_TOL = 10 mm`. All 11 solved tasks had a reach error of 4–9.5 mm, and 4 of the 9 failures missed by 10–13 mm. The reward barely distinguishes 4 mm from 13 mm (affordance 1.0 vs 0.92, Φ differs by 0.04), so the policy learns precision only from the sparse +10.
2. **No room for a retry.** With `max_steps = 5`, a failed attempt (3 steps) leaves only 2 steps, and a retry needs 3. Seeds 10002, 10010 and 10019 retried correctly but ran out of steps. For the same reason, training never rewards recovery.
3. **Some goals not learned yet (3 tasks).** In 10008, 10012 and 10015 the reach goes about 22 cm off, and in 10004 the policy opens without reaching. This is likely to improve with more training.
4. **Finger still touching after open (1 task).** In 10013 the reach was at 2 mm, but a finger still touched the cube after `open`, so `_is_placed` was false and the policy grasped again and pushed the cube away. This is the known 2/30 case from the plan.

## Run 3: latest SAC training → 70%

**Run directory:** Not yet recorded.  
**Training script:** `youbot_pnp/scripts/train_sac.py`  
**Total training steps:** 100,000.

The latest training run improves the previous evaluation result from 55% to **70%**. The training logs show some variation in success during the final iterations, ranging from 70% to 76%, rather than a consistent improvement at every checkpoint.

### Final evaluation

| Metric | Value |
|---|---:|
| `eval/success_rate` | **70%** |
| `eval/mean_reward` | 7.79 ± 4.90 |
| `eval/mean_ep_length` | 3.60 ± 0.92 |
| `primitives/aff` | 0.991 |
| `primitives/success_grasp` | 0.941 |
| `primitives/success_reach` | 1.000 |
| `primitives/success_open` | 1.000 |
| `primitives/failure_rate` | 0.0208 |
| `primitives/chosen_grasp` | 0.354 |
| `primitives/chosen_reach` | 0.333 |
| `primitives/chosen_open` | 0.313 |
| `train/ent_coef` | 0.0194 |
| `train/actor_loss` | -8.89 |
| `train/critic_loss` | 0.517 |
| `train/learning_rate` | 0.0003 |
| `train/n_updates` | 94,996 |

### Training behaviour near 100k steps

| Total timesteps | Success rate | Episode reward | Episode length | Aff | Failure rate |
|---:|---:|---:|---:|---:|---:|
| 99,660 | 74% | 8.17 | 3.53 | 0.947 | 0.0526 |
| 99,732 | 76% | 8.37 | 3.50 | 0.956 | 0.0556 |
| 99,800 | 73% | 8.05 | 3.56 | 0.972 | 0.0147 |
| 99,880 | 73% | 8.01 | 3.56 | 0.952 | 0.0500 |
| 99,952 | 70% | 7.69 | 3.62 | 0.955 | 0.0694 |
| **100,000 (evaluation)** | **70%** | **7.79** | **3.60** | **0.991** | **0.0208** |

The final evaluation reports:

- `success_grasp = 0.941`
- `success_reach = 1.0`
- `success_open = 1.0`
- `chosen_grasp / reach / open = 0.354 / 0.333 / 0.313`

The three primitives are selected at relatively similar frequencies. There is no obvious sign in these metrics that the policy has stopped selecting one of the primitives entirely.

### Interpretation

The new run is a clear improvement over the previous 55% result:

- Evaluation success increased by 15 percentage points.
- Mean episode reward increased from 6.0 to 7.79.
- Mean episode length decreased from 3.9 to 3.6 steps.
- The reported grasp success increased from 0.91 to 0.941.
- The affordance score increased from 0.97 to 0.991.

However, the target of **more than 80% success has not yet been reached**.

The most interesting observation is the difference between primitive-level success and full-task success. Reach and open both report 100% success, and grasp reports 94.1%, but the overall task success is only 70%.

This suggests that the next investigation should focus on how primitive-level success is defined and how the environment determines whether the complete pick-and-place task has succeeded. A primitive can finish successfully without necessarily completing the intended task.

The primitive selection frequencies are also relatively balanced. Therefore, there is no immediate evidence that the main issue is simply the policy avoiding one primitive.

**Important:** The latest logs alone do not establish whether the reward function or episode configuration changed relative to Run 2. The run directory, reward configuration and environment parameters should be recorded before making a direct causal comparison.

### OMPL / nanobind shutdown warning

At the end of the run, the process printed warnings such as:

`nanobind: leaked 6 instances!`

The reported objects include OMPL planners, state spaces, validity checkers and problem definitions.

This appears to be a resource/reference-management warning from the Python bindings during shutdown. It is not, by itself, evidence of an SAC training failure and does not explain the 70% success rate. It should be investigated separately if it causes resource accumulation or affects repeated runs.

## Visual Analysis of Policy Evolution

The following videos show the evolution of the SAC policy at different
training checkpoints. All checkpoints are evaluated using the same
initial conditions whenever possible.

### 25k timesteps

[Watch execution](media/sac/1cube/sac_025k.mp4)

**Observations:**
- The policy has not learned a reliable pick-and-place sequence yet.
- It frequently repeats grasp and open actions.
- None of the five observed episodes completed the task.

### 50k timesteps

[Watch execution](media/sac/1cube/sac_050k.mp4)

**Observations:**
- The first successful execution appears.
- The policy still shows inconsistent behaviour across episodes.
- Only one of the five observed episodes completed the task.

### 75k timesteps

[Watch execution](media/sac/1cube/sac_075k.mp4)

**Observations:**
- All five observed episodes completed the task.
- The policy generally follows the grasp → reach → open sequence.
- One execution recovered after an unsuccessful reach attempt.

### 100k timesteps

[Watch execution](media/sac/1cube/sac_100k.mp4)

**Observations:**
- Four of the five observed episodes completed the task.
- One execution failed despite the individual primitives reporting success.
- This suggests that successful primitive execution does not necessarily
  imply successful task completion.

## Comparison of the three runs

| Metric | Run 1: state-based | Run 2: potential-based | Run 3: latest |
|---|---:|---:|---:|
| Training steps | 100k | 100k | 100k |
| Evaluation success | 0% | 55% | **70%** |
| Mean evaluation reward | 5.13 | 6.0 | **7.79** |
| Mean episode length | 5.0 | 3.9 | **3.60** |
| Main observation | Reward hacking | Learned grasp → reach → open | Better overall performance, but below target |

The comparison shows substantial progress from the initial state-based reward to the latest result. However, because the exact configuration of Run 3 has not yet been recorded, the improvement should be treated as an empirical result rather than attributed to a specific reward modification.

## Recommended next steps

The priority is now to understand why the task success rate remains below 80%, rather than immediately increasing the training budget.

1. **Inspect the reward implementation.** Verify the potential calculation, progress reward, terminal reward and penalties. Check whether the agent can receive a high return without actually placing the cube.
2. **Inspect the success condition.** Verify exactly what `_is_placed` checks: cube position, placement tolerance, release state and finger contact. Compare this with the primitive-level success metrics.
3. **Analyse failed evaluation episodes.** Replay failed seeds with `watch_policy.py` and record the primitive sequence, cube position, target position, grasp state and final placement error. This will show whether the remaining failures are due to inaccurate placement, incorrect primitive sequencing, failed release or episode termination.
4. **Check the episode limit.** Run 2 identified retries that ran out of steps. Confirm whether Run 3 still uses `max_steps = 5 * n_cubes`. If so, test a longer episode limit separately, while keeping the placement tolerance unchanged.
5. **Improve reward shaping only after identifying the bottleneck.** If placement precision is still the main issue, test a sharper distance-based potential near the target. If the problem is release or task completion, improve the corresponding state representation and reward instead of increasing the reach reward indiscriminately.
6. **Run controlled comparisons.** Keep the same evaluation seeds and deterministic evaluation procedure. Change one factor at a time and compare against the 70% baseline.
7. **Reach and validate the >80% target.** Require the improvement to persist across several evaluations. Then repeat with additional training seeds to check that the result is reproducible.

What not to change:

- **Don't raise `PLACE_TOL`.** It would make the task easier than for the FSM and break the comparison in plan step 8.
- **Don't change several reward coefficients and SAC hyperparameters simultaneously.** That would make it difficult to identify which modification actually improves learning.
- **Don't use training reward as a substitute for task success.** The main target remains the independently evaluated success rate.

## Open decisions for later steps

- **Fixed observation size.** `obs_dim = 9 · n_cubes + 7` changes with the number of cubes, so a 1-cube model can't continue training on 2 or 4 cubes. With fixed 4 cube slots plus an "active" flag, the curriculum in plan step 7 could reuse models. This must be decided before models are meant to carry over.
- **MAPLE version (plan step 6).** SB3 SAC has one entropy term for all 7 action dimensions. If the primitive choice stops exploring too early again, for example on harder tasks, the separate entropy for the primitive choice in MAPLE's `PAMDPPolicy` / `SACHybridTrainer` is the structural fix.
- **Buffer slots for `occupied` / `swap`.** The reach affordance currently rewards only the held cube's goal. Moves to a free buffer slot need extra reach keypoints while the goal is blocked.

## Tools

| Script | Purpose |
|---|---|
| `youbot_pnp/scripts/train_sac.py` | Training; logs `eval/success_rate`, `primitives/*` (choice, success, aff) to TensorBoard |
| `youbot_pnp/scripts/watch_policy.py` | Replays a saved model in the viewer on the eval tasks and prints each primitive |
| `youbot_pnp/scripts/check_timing.py` | Wall time per primitive (plan step 1) |
| `tensorboard --logdir runs` | Learning curves of all runs |
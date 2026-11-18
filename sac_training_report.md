# SAC training on 1 cube: results and next steps

Status as of 2026-11-18. Plan step 5 ("training, quick version") in [maple_adaptation_plan.md](maple_adaptation_plan.md).

**Summary:** with a potential-based reward, SB3 SAC learns grasp → reach → open and solves **55 %** of the eval tasks after 100k steps. The target is > 80 %. Success was still rising when the run ended. Most failures are reaches that miss the goal by just over the 10 mm placement tolerance, or retries that run out of episode steps. The next run needs two small env changes (a longer episode and sharper shaping near the goal) and 300k steps.

## Setup

| | |
|---|---|
| Script | `youbot_pnp/scripts/train_sac.py`, results in `runs/<name>/` |
| Task | `YouBotPnPEnv(n_cubes=1, variant="random")`, up to 5 primitives per episode |
| Algorithm | SB3 SAC, `MlpPolicy` [256, 256], flat 7-dim action (argmax of the first 3 values picks the primitive) |
| Training | 100k primitive steps, 8 parallel envs (`SubprocVecEnv`), 5k random warm-up steps, 1 gradient step per transition, `ent_coef="auto"` |
| Evaluation | every 5k steps on 20 fixed tasks (seeds 10000–10019, not used for training), deterministic policy |
| Wall time | about 1.5 h per 100k steps |

## Run 1: state-based reward → reward hacking

`runs/sac_1cube_random_s0_1117_1910`

The reward paid for the current state at every step: +1 per placed cube, and `0.25 + 1 − tanh(10 d)` while a cube is held.

| Metric (100k steps) | Value |
|---|---|
| `eval/success_rate` | **0 %** |
| `eval/mean_reward` | 5.13 |
| episode length | always 5 |
| `chosen_grasp / reach / open` | 0.2 / 0.8 / **0** |

The policy learned to grasp the cube, carry it to the goal and then keep holding it with 4 reaches, never opening. Hand-picked actions on 10 seeds confirmed the cause:

| Sequence | Return | Solved |
|---|---|---|
| grasp, reach ×4 (hold at the goal) | 5.31 | 0/10 |
| grasp, reach, open (place) | 11.54 | 9/10 |

Placing paid twice as much, but the policy never discovered it. A random reach lands within 1 cm of the goal only about 0.75 % of the time, so the +10 for placing never appeared during exploration. Holding, meanwhile, paid a safe ~1.25 per step. Once exploration dropped (`ent_coef` 0.035), the policy stayed there.

## Run 2: potential-based reward → 55 %

`runs/sac_1cube_potential_s0`

The reward now pays only for **progress**, `Φ(after) − Φ(before)`, with

- Φ = 1 per placed cube, plus up to `0.25 + 0.5 · (1 − tanh(10 d))` for a held cube at distance `d` from its goal (at most 0.75 < 1),
- +10 when all cubes are placed, −0.1 for a failed primitive, −5 for a cube that falls off the table, −λ·(1 − aff) affordance penalty.

Holding still now earns 0. The same hand-picked check:

| Sequence | Return | Solved |
|---|---|---|
| grasp, reach ×4 (hold at the goal) | 0.74 | 0/10 |
| grasp, reach, open (place) | 9.90 | 9/10 |
| grasp, open, … (drop at the start) | −3.00 | 0/10 |

### Learning curve (eval on the 20 fixed tasks)

| Steps | Success | Episode length | Mean reward |
|---|---|---|---|
| 5k–35k | 0 % | 5.0 | −3.4 → −0.4 |
| 40k–60k | 0–10 % | 5.0 | around 0 |
| 65k–80k | 15–25 % | 4.7 → 4.5 | 1.5–2.9 |
| 85k–95k | 35–45 % | 4.3 → 4.1 | 3.9–5.0 |
| **100k** | **55 %** | **3.9** | **6.0** |

There was no success for 35k steps, until the policy discovered "open at the goal". From then on, success rose steadily and had not levelled off when the run ended.

Final training statistics: `aff` 0.97, `success_grasp` 0.91, `success_reach` 1.0, `chosen_grasp / reach / open` 0.41 / 0.36 / 0.23, `ent_coef` 0.024.

### Failure analysis (final model, 20 eval tasks)

| Seed | Result | Final distance | Primitive sequence (reach error = distance of the reach target to the goal) |
|---|---|---|---|
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
2. **No room for a retry.** With `max_steps = 5`, a failed attempt (3 steps) leaves only 2 steps, and a retry needs 3. Seeds 10002, 10010 and 10019 retried correctly but ran out of steps (in 10010 the second reach was at 4 mm). For the same reason, training never rewards recovery.
3. **Some goals not learned yet (3 tasks).** In 10008, 10012 and 10015 the reach goes about 22 cm off, and in 10004 the policy opens without reaching. This is likely to improve with more training.
4. **Finger still touching after open (1 task).** In 10013 the reach was at 2 mm, but a finger still touched the cube after `open`, so `_is_placed` was false and the policy grasped again and pushed the cube away. This is the known 2/30 case from the plan.

## Recommended next steps

Make one change at a time and test it before the next one.

1. **Longer episodes.** In `youbot_pnp/rl/env.py`, `__init__`, replace `self.max_steps = 5 * n_cubes` with `self.max_steps = 8 * n_cubes`. That allows one full retry plus some slack. Eval numbers are then not directly comparable to run 2.
2. **Sharper shaping close to the goal.** In `_potential()`, split the distance term into a wide and a steep part, keeping the maximum at 0.75:
   ```python
   phi += 0.25 + 0.25 * (1 - np.tanh(10 * d)) + 0.25 * (1 - np.tanh(50 * d))
   ```
   The steep term separates 4 mm (0.24) from 13 mm (0.19). If precision is still the bottleneck afterwards, raise the affordance scale for reach as well (`AFF_SCALE` 30 instead of 10).
3. **Check both changes before training.** Use the hand-picked sequences (place vs. hold vs. drop) and `check_rl_env.py`.
4. **Train for 300k steps from scratch** (about 4–5 h with 8 envs), e.g. `--steps 300000 --envs 8 --name sac_1cube_potential_v2_s0`. The goal is `eval/success_rate` > 80 %, stable over several evaluations.
5. **Repeat with other seeds** (`--seed 1`, `--seed 2`) once a run reaches the goal, so the result isn't luck.

What not to change:

- **Don't raise `PLACE_TOL`.** It would make the task easier than for the FSM and break the comparison in plan step 8.
- **Leave the finger-contact case (cause 4) for now.** It affects about 5 % of episodes. Revisit it if it limits success after the changes above.

## Open decisions for later steps

- **Fixed observation size.** `obs_dim = 9 · n_cubes + 7` changes with the number of cubes, so a 1-cube model can't continue training on 2 or 4 cubes. With fixed 4 cube slots plus an "active" flag, the curriculum in plan step 7 could reuse models. This must be decided before models are meant to carry over.
- **MAPLE version (plan step 6).** SB3 SAC has one entropy term for all 7 action dimensions. If the primitive choice stops exploring too early again, for example on harder tasks, the separate entropy for the primitive choice in MAPLE's `PAMDPPolicy` / `SACHybridTrainer` is the structural fix.
- **Buffer slots for `occupied` / `swap`.** The reach affordance currently rewards only the held cube's goal. Moves to a free buffer slot need extra reach keypoints while the goal is blocked.

## Tools

| Script | Purpose |
|---|---|
| `youbot_pnp/scripts/train_sac.py` | training; logs `eval/success_rate`, `primitives/*` (choice, success, aff) to TensorBoard |
| `youbot_pnp/scripts/watch_policy.py` | replays a saved model in the viewer on the eval tasks and prints each primitive |
| `youbot_pnp/scripts/check_timing.py` | wall time per primitive (plan step 1) |
| `tensorboard --logdir runs` | learning curves of all runs |

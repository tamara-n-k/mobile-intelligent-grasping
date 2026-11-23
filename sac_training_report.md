# SAC training on 1 cube: results and next steps

Status as of 2026-11-21. Plan step 5 ("training, quick version") in [maple_adaptation_plan.md](maple_adaptation_plan.md).

## Summary

- **Result (Runs 5 and 6).** After fixing two problems in the environment, SB3 SAC reached **100% evaluation success (20/20) after 100k steps**, both with 4 parallel environments (Run 5) and with 8 (Run 6). In both runs the mean episode reward is 11.00 and the mean episode length is 3.00 steps, and the training rollouts also showed 100% success over the last logged iterations. Before the fixes, the best results were 70% with 4 environments and 55% with 8 (Runs 3 and 2).
- **Why the environment mattered.** A scripted oracle (`check_oracle.py`: perfect parameters, no learning) solved only 89/100 random tasks and 19/20 of the evaluation tasks before the fixes. That is an upper bound for any learned policy. Of the 11 failed oracle episodes, 10 were correct placements that the success predicate did not count because a finger still touched the cube, and 1 was a cube that spawned out of the arm's reach.
- **Number of parallel environments.** The same progression was run with 4 and with 8 environments: 0% at the start, 70% (4) and 55% (8) after the affordance score was changed, and 100% in both after the environment fixes. With 8 environments the same 100k steps took about 30% less wall time (8,584 s instead of 12,233 s). See [Effect of the number of parallel environments](#effect-of-the-number-of-parallel-environments).
- **How we got here.** After Run 3 (70%), the hypothesis was that primitive-level changes would close the gap to full-task success. Changing primitives and `env.py` together dropped success to 35% (Run 4). Each change was then tested on its own (B: Reach, C: Open, D: both) and all stayed at exactly the baseline's 70%, so the primitive-level checks were not what mattered. The next question was whether the environment capped what any policy can achieve, which led to the oracle and to the two environment fixes. See [How we got here](#how-we-got-here-from-primitive-changes-to-the-oracle).
- **4 cubes with fixed goals (Runs 7 and 8).** A new variant, `fixed_goals`, gives every cube its own fixed destination and random start positions. The oracle solves it (100/100), but SAC did not learn it: 0% success after 100k steps (generator change only) and after 500k steps (generator + primitive fix). In both runs the policy never chose `reach` and alternated `grasp` and `open`. The primitive fix worked (grasp success 0.64–0.79 → 1.0), so the failure is one of exploration, not of the primitives. Making `reach` snap to the goal is the next experiment. See [Four cubes with fixed goals](#four-cubes-with-fixed-goals-runs-7-and-8).
- **Primitive modifications (B, C, D, Run 4).** Reach verification (B), the Open held-object check (C) and their combination (D) all finished at 70%, and a larger combined implementation (Run 4) reached only 35%. These runs used the old environment, so their negative result applies to it. The original `primitives.py`, `env.py` and `check_primitives.py` were restored from commit `5cd8bda`.
- **Caveats.** One training seed and 20 evaluation episodes. With 20/20 successes the 95% lower confidence bound is about 84%. Confirm with more evaluation seeds and additional training seeds before claiming a perfect result (see [Next steps](#next-steps)).

## Setup

| Item | Value |
|---|---|
| Script | `youbot_pnp/scripts/train_sac.py`; results in `runs/<name>/` |
| Task | `YouBotPnPEnv(n_cubes=1, variant="random")`, up to 5 primitives per episode † (Runs 7 and 8: `n_cubes=4, variant="fixed_goals"`, up to 20 primitives) |
| Algorithm | SB3 SAC, `MlpPolicy` [256, 256], flat 7-dim action (argmax of the first 3 values picks the primitive) |
| Training | 100k primitive steps, 4 or 8 parallel envs depending on the run (`SubprocVecEnv`; see [the table by number of environments](#effect-of-the-number-of-parallel-environments)), 5k random warm-up steps, 1 gradient step per transition, `ent_coef="auto"` † |
| Evaluation | Every 5k steps on 20 fixed tasks (seeds 10000–10019, never used for training), deterministic policy † |
| Wall time | About 2.4–3.4 hours per 100k steps (Run 3: 10,893 s; Run 5, 4 envs: 12,233 s at about 8 fps; Run 6, 8 envs: 8,584 s at about 11 fps) |

† Taken from the documentation of the earlier runs; confirm it is unchanged for Run 5.

## Run 1: state-based reward → reward hacking

Run directory: `runs/sac_1cube_random_s0_1117_1910`

The reward paid for the current state at every step: +1 per placed cube, and `0.25 + 1 − tanh(10 d)` while a cube is held.

| Metric (100k steps) | Value |
|---|---:|
| `eval/success_rate` | **0%** |
| `eval/mean_reward` | 5.13 |
| Episode length | always 5 |
| `chosen_grasp / reach / open` | 0.2 / 0.8 / **0** |

The policy learned to grasp the cube, carry it to the goal and then keep holding it with 4 reaches, never opening. Hand-picked actions on 10 seeds confirmed the cause:

| Sequence | Return | Solved |
|---|---:|---:|
| grasp, reach ×4 (hold at the goal) | 5.31 | 0/10 |
| grasp, reach, open (place) | 11.54 | 9/10 |

Placing paid twice as much, but the policy never discovered it. A random reach lands within 1 cm of the goal only about 0.75% of the time, so the +10 for placing never appeared during exploration. Holding, meanwhile, paid a safe ~1.25 per step. Once exploration dropped (`ent_coef` 0.035), the policy stayed there.

## Run 2: potential-based reward → 55%

Run directory: `runs/sac_1cube_potential_s0`

The reward now pays only for **progress**, `Φ(after) − Φ(before)`:

- Φ = 1 per placed cube, plus up to `0.25 + 0.5 · (1 − tanh(10 d))` for a held cube at distance `d` from its goal (at most 0.75 < 1).
- +10 when all cubes are placed.
- −0.1 for a failed primitive.
- −5 for a cube that falls off the table.
- −λ · (1 − aff) affordance penalty.

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
| 40k–60k | 0–10% | 5.0 | around 0 |
| 65k–80k | 15–25% | 4.7 → 4.5 | 1.5–2.9 |
| 85k–95k | 35–45% | 4.3 → 4.1 | 3.9–5.0 |
| **100k** | **55%** | **3.9** | **6.0** |

There was no success for 35k steps, until the policy discovered "open at the goal". From then on, success rose steadily and had not levelled off when the run ended.

Final training statistics: `aff` 0.97, `success_grasp` 0.91, `success_reach` 1.0, `chosen_grasp / reach / open` 0.41 / 0.36 / 0.23, `ent_coef` 0.024.

### Failure analysis (final model, 20 eval tasks)

"Reach error" is the distance of the reach target to the goal.

| Seed | Result | Final distance | Primitive sequence |
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

> **TODO: check the counts.** This table lists 12 solved and 8 failed tasks (60%). The evaluation reported 55% (11/20), and the text below mentions 11 solved tasks and 9 failures. Either one row is mislabelled or the table comes from a different checkpoint than the 55% evaluation.

Four failure causes:

1. **Reach is barely precise enough (main cause).** A cube counts as placed within `PLACE_TOL = 10 mm`. All 11 solved tasks had a reach error of 4–9.5 mm, and 4 of the 9 failures missed by 10–13 mm. The reward barely distinguishes 4 mm from 13 mm (affordance 1.0 vs 0.92, Φ differs by 0.04), so the policy learns precision only from the sparse +10.
2. **No room for a retry.** With `max_steps = 5`, a failed attempt (3 steps) leaves only 2 steps, and a retry needs 3. Seeds 10002, 10010 and 10019 retried correctly but ran out of steps. For the same reason, training never rewards recovery.
3. **Some goals not learned yet (3 tasks).** In 10008, 10012 and 10015 the reach goes about 22 cm off, and in 10004 the policy opens without reaching. This is likely to improve with more training.
4. **Finger still touching after open (1 task).** In 10013 the reach was at 2 mm, but a finger still touched the cube after `open`, so `_is_placed` was false and the policy grasped again and pushed the cube away. Run 5 shows this was a general problem of the success predicate, not a single case (see below).

## Run 3: SAC training with the original primitives → 70%

Run directory: `runs/sac_1cube_random_s0_1119_0124` · Script: `youbot_pnp/scripts/train_sac.py` · Training steps: 100,000

This run improved the evaluation result from 55% to **70%**. Success in the training logs varied between 70% and 76% over the final iterations rather than improving at every checkpoint.

> **Note.** The logs alone do not show whether the reward function or the episode configuration changed relative to Run 2. Record the reward configuration and environment parameters before making a direct causal comparison.

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
| `train/actor_loss` | −8.89 |
| `train/critic_loss` | 0.517 |
| `train/learning_rate` | 0.0003 |
| `train/n_updates` | 94,996 |

### Behaviour near 100k steps

The first five rows come from the training logs; the last row is the final evaluation.

| Total timesteps | Success rate | Episode reward | Episode length | Aff | Failure rate |
|---:|---:|---:|---:|---:|---:|
| 99,660 | 74% | 8.17 | 3.53 | 0.947 | 0.0526 |
| 99,732 | 76% | 8.37 | 3.50 | 0.956 | 0.0556 |
| 99,800 | 73% | 8.05 | 3.56 | 0.972 | 0.0147 |
| 99,880 | 73% | 8.01 | 3.56 | 0.952 | 0.0500 |
| 99,952 | 70% | 7.69 | 3.62 | 0.955 | 0.0694 |
| **100,000 (evaluation)** | **70%** | **7.79** | **3.60** | **0.991** | **0.0208** |

### Interpretation

Compared with Run 2:

- Evaluation success: 55% → **70%** (+15 points).
- Mean evaluation reward: 6.0 → 7.79.
- Mean episode length: 3.9 → 3.60 steps.
- Grasp success: 0.91 → 0.941.
- Affordance: 0.97 → 0.991.

The target of more than 80% was not reached. The most interesting observation was the gap between primitive-level success (reach and open 100%, grasp 94.1%) and full-task success (70%). A primitive can finish successfully without completing the intended task. Run 5 shows that part of this gap came from how the environment decided that a cube was placed.

The three primitives were chosen at similar frequencies (0.354 / 0.333 / 0.313), so there was no sign that the policy had stopped using one of them.

### OMPL / nanobind shutdown warning

At the end of the run the process printed warnings such as `nanobind: leaked 6 instances!`, listing OMPL planners, state spaces, validity checkers and problem definitions. This looks like a reference-management warning from the Python bindings at shutdown. It is not evidence of a training failure and does not explain the 70% success rate. Investigate it separately only if it causes resource accumulation or affects repeated runs.

## Follow-up investigation: primitive modifications (B–D and Run 4)

### Objective

After Run 3 reached 70% evaluation success, the goal was to exceed 80%. The working hypothesis was that changes to the hierarchical primitives could improve task completion by verifying grasp retention and release, and by making placement more robust.

Experiments B, C and D were all based on the original code at commit `5cd8bda`. They used the same SAC configuration and evaluation procedure as the baseline (A): seed 0, 100k training steps, evaluation every 5k steps on the fixed evaluation set. `train_sac.py` and `affordance.py` had no diff against `5cd8bda` between the baseline and the `hrl` implementation.

### Experiment B: Reach contact verification

Change in `youbot_pnp/rl/primitives.py`. The original Reach movement was kept, but its two movement calls were combined into one success result. If either movement failed, Reach returned `False`. If an object was held, Reach also checked that both fingers were still in contact after the placement movement:

```python
success = (
    self.task.execute_move(q_app, held, finger_pos=fp) and
    self.task.execute_move(q_drop, held, linear=True, finger_pos=fp)
)

if not success:
    return False

if held is not None:
    return all(finger_contacts(self.robot, held))

return True
```

The recovery behaviour stayed as `pass`, so a failed Reach does not open the gripper and drop the object.

**Result:** 70% final evaluation success, the same as baseline A. The saved evaluation arrays (`timesteps`, `results`, `ep_lengths`) are exactly identical to A (maximum difference 0), so this check produced no observable change in the evaluated learning trajectory.

### Experiment C: Open held-object check

Change in `youbot_pnp/rl/primitives.py`. Open records whether an object was held before opening, and returns `True` only if one had been:

```python
def _run(self, target=None):
    held = self.held()
    self._open()
    return held is not None
```

This experiment did not include storing the placement approach, a post-release retraction or a release-contact check.

Run directory: `runs/sac_1cube_random_s0_held_verification/`

C changed the evaluation reward trajectory relative to A, with higher mean rewards at several intermediate checkpoints, but it did not improve final task success: both finished at 70%. Its `results` and `ep_lengths` arrays are not identical to A.

### Experiment D: combined Reach and Open checks

D combined exactly B and C: Reach checks that both fingers remain in contact after the placement movement, and Open checks that an object was held before opening. It did not include the shared context, the stored `q_app`, the post-open retraction or the release verification of Run 4.

Run directory: `runs/sac_1cube_random_s0_reach_held_verification/`

**Final evaluation at 100k steps (C and D gave the same values):**

| Metric | Value |
|---|---:|
| Evaluation success | 70% (14/20) |
| Mean evaluation reward | 7.82 ± 4.82 |
| Mean episode length | 3.60 ± 0.92 |
| Affordance (`aff`) | 0.917 |
| Failure rate | 0.125 |
| `chosen_grasp` | 0.450 |
| `chosen_open` | 0.175 |
| `chosen_reach` | 0.375 |
| `success_grasp` | 0.722 |
| `success_open` | 1.000 |
| `success_reach` | 1.000 |

**Checkpoint-by-checkpoint comparison of C and D.** The `evaluations.npz` files were compared at every common checkpoint from 5,000 to 100,000 steps. Mean reward (`results.mean(axis=1)`) and mean episode length (`ep_lengths.mean(axis=1)`) are exactly equal at all 20 checkpoints; the maximum absolute difference is 0.0 for both. The table shows the checkpoints up to 90k.

| Steps | Mean reward (C = D) | Mean length (C = D) |
|---:|---:|---:|
| 5,000 | −3.399 | 5.00 |
| 10,000 | −2.396 | 5.00 |
| 15,000 | −0.638 | 5.00 |
| 20,000 | −0.217 | 5.00 |
| 25,000 | 0.491 | 4.90 |
| 30,000 | −0.271 | 5.00 |
| 35,000 | 1.512 | 4.70 |
| 40,000 | 2.098 | 4.60 |
| 45,000 | 1.590 | 4.70 |
| 50,000 | 3.639 | 4.35 |
| 55,000 | 3.474 | 4.40 |
| 60,000 | 8.240 | 3.50 |
| 65,000 | 7.138 | 3.75 |
| 70,000 | 9.237 | 3.30 |
| 75,000 | 7.868 | 3.60 |
| 80,000 | 7.342 | 3.70 |
| 85,000 | 7.843 | 3.60 |
| 90,000 | 8.883 | 3.40 |

This shows that the aggregated mean reward and episode-length curves are identical. It does not by itself show that every individual evaluation episode in the two files is identical.

**Conclusion from D:** adding Reach verification to the Open check did not change the measured mean evaluation curves or the final success relative to C. The combined isolated checks did not improve the 70% baseline.

### Run 4: extended implementation and regression

Run directory: `runs/sac_1cube_random_s0_1119_0731/`

This implementation went beyond B and C. The runtime changes between `5cd8bda` and branch `hrl` were concentrated in `primitives.py` and the context plumbing in `env.py`.

Changes included:

- `Primitive.__init__` accepted an optional shared context dictionary.
- `env.py` created `self.context`, passed it to `Grasp`, `Reach` and `Open`, and cleared it on reset.
- Reach stored the placement approach configuration as `self.context["place_approach"]`.
- Reach returned failure if either movement failed, and verified finger contacts when carrying an object.
- Open checked that an object was held before opening.
- Open retrieved the stored placement approach and executed a retraction with the gripper open.
- Open checked that finger contact with the object had ended after opening.
- Open's `_recover()` was overridden with `pass`, instead of using the inherited recovery that opens the gripper.

The inspected `env.py` diff was limited to creating, passing and resetting the context. It showed no change to the observation space, action space, reward, affordance or termination logic. The altered physical sequence was implemented in the primitives.

**Final evaluation at 100k steps:**

| Metric | Value |
|---|---:|
| Evaluation success | 35% (7/20) |
| Mean evaluation reward | 3.4683 |
| Mean episode length | 4.30 |
| Affordance (`aff`) | 0.86296 |
| Failure rate | 0.03125 |
| `chosen_grasp` | 0.4375 |
| `chosen_open` | 0.375 |
| `chosen_reach` | 0.1875 |
| `success_grasp` | 1.000 |
| `success_open` | 0.9167 |
| `success_reach` | 1.000 |

The learning curve stayed at 0% success through 40k steps, reached only 10% at 75k and finished at 35%, against 70% for baseline A at 100k steps. The run combined several changes (shared-context handoff, retraction after opening, release verification and a changed Open recovery), so the evidence does not identify which of them caused the regression. Attributing it to one of them would need further ablation.

### Conclusion and decision

| Experiment | Change | Final success | Note |
|---|---|---:|---|
| A | Baseline (original primitives, `5cd8bda`) | 70% | Control |
| B | Reach contact verification | 70% | Evaluation arrays identical to A |
| C | Open held-object check | 70% | Different intermediate trajectory, same final success |
| D | B + C | 70% | Mean reward and length curves identical to C at all 20 checkpoints |
| Run 4 | Extended implementation | 35% | Clear regression; cause not isolated |

The hypothesis that primitive-level verification would raise SAC task success above the 70% baseline was **not supported**. The line of changes was discontinued, and these files were restored from commit `5cd8bda`:

- `youbot_pnp/rl/primitives.py`
- `youbot_pnp/rl/env.py`
- `youbot_pnp/scripts/check_primitives.py`

Keep the experiment run directories and this report so the negative result stays documented and reproducible. The rollback must not discard unrelated documentation, videos or experiment data.

> **Note (added after Run 5).** B, C, D and Run 4 were all trained and evaluated with the original success predicate (`not any(finger_contacts)`) and the original task generator. Run 5 shows that this environment capped any policy at about 89–95% on 1 cube, so these results apply to that environment. They were not repeated with the fixed one.

## How we got here: from primitive changes to the oracle

This is the order of events that led to the environment fixes. The sections above list the experiments by type, not in time. † Confirm the order against the run timestamps.

1. **Starting point (Run 3, 70%).** Primitive-level metrics were close to perfect (reach and open 100%, grasp 94.1%), but full-task success was 70%. A primitive can report success without the task being completed, so the gap had to come from somewhere other than the policy's choice of primitives.
2. **Hypothesis.** Changes to the primitives (checking that the cube is still held after reach and that it was held before open, retracting after open) could close the gap. These changes were implemented **together with changes in `env.py`** (the shared context between primitives). This is Run 4.
3. **Regression (Run 4, 35%).** Success dropped from 70% to 35%. Because several changes had been mixed, the cause could not be identified.
4. **One change at a time (B, C, D).** Each change was then implemented on its own on top of the original code: Reach verification (B), Open held-object check (C) and both together (D). All three finished at **exactly 70%**, the same as the baseline (B's evaluation arrays are identical to the baseline's, and the C and D curves are identical at all 20 checkpoints).
5. **Conclusion.** The primitive-level checks did not explain or improve the result, so the lever was elsewhere. The environment (what it decides, and how) appeared to matter much more for training than expected. Run 4's own drop is not explained by this: its cause was not isolated, because it combined several changes.
6. **New question.** If the primitives with perfect parameters cannot reliably solve the task, no policy can reach 100% however it is trained. The tool to answer this is a perfect scripted policy: the oracle.
7. **Result.** The oracle showed that the environment capped success at about 89–95% with 1 cube and 67% with 4 cubes. It pointed to the exact causes (see below), which were fixed in `env.py` and `task_generator.py` without touching the primitives. SAC then reached 100% with 4 and with 8 environments (Runs 5 and 6).

## The oracle: `check_oracle.py`

### Motivation

After B, C and D, the remaining 30% could be caused by the policy (it chooses poorly) or by the environment (the task cannot always be solved, or a correct solution is not counted). The SAC logs cannot separate the two. The policy only chooses a primitive and a target. The execution of the primitive and the decision that a cube is "placed" (`_is_placed`) belong to the environment. If those fail with perfect parameters, no learned policy can do better. A perfect scripted policy is the standard control for this: an upper bound.

There is a second reason. A 4-cube task needs 12 primitives (3 per cube). If each primitive succeeds with probability p, the task succeeds with about p¹²: 0.97 gives 69% and 0.99 gives 89%. Small per-primitive failures multiply.

### What the script does

`youbot_pnp/scripts/check_oracle.py` plays a perfect policy through `env.step()`:

- It reads the true cube and goal positions from the simulator.
- If no cube is held, it runs `grasp` on the first unplaced cube. If one is held and it is more than 2 cm from its goal, it runs `reach` to the goal. Otherwise it runs `open`.
- It builds the action by inverting `_to_workspace`, so it goes through the same primitives, success predicate and episode rules as SAC.
- It prints one line per seed and a summary with the number of solved tasks and the first failing primitive of each failed episode.

```bash
python youbot_pnp/scripts/check_oracle.py --cubes 1 --seeds 100
python youbot_pnp/scripts/check_oracle.py --cubes 1 --first-seed 10000 --seeds 20   # the evaluation seeds
python youbot_pnp/scripts/check_oracle.py --cubes 4 --seeds 60
```

Reading a line: `g+ r+ o+` is grasp, reach and open, each successful (`-` is a failed primitive). If more primitives follow a successful `g+ r+ o+`, the environment did not count the cube as placed.

### Results, step by step

**1 cube (`random`).**

| Environment | Seeds 0–99 | Evaluation seeds 10000–10019 |
|---|---:|---:|
| Original | 89/100 (89%) | 19/20 (95%) |
| `_is_placed` with `not all` (seeds 0–39 only) | 40/40 ‡ | not measured |
| `not all` + x limit in the generator | 100/100 ‡ | 20/20 ‡ |

**4 cubes (`random`).** The number of seeds differs between rows, so compare them with care.

| Environment | Solved |
|---|---:|
| Original | 10/15 (67%) ‡ |
| + `not all` | 24/30 (80%) ‡ |
| + x limit in the generator | 53/60 (88%) ‡ |
| + vertical Cartesian rise instead of the joint-space `servo_to` in `_free_start` | 60/60 (100%) ‡ |

‡ Measured on a separate copy of the repo (MuJoCo 4.0.1, Python 3.12). **TODO:** repeat locally. The 4-cube fixes (the vertical rise) have **not** been applied to the training environment yet.

### What it revealed

1. **Correct placements that were not counted.** `_is_placed` required that no finger touched the cube. After `open`, the cube was 1–4 mm from the goal, but one finger still touched it. The oracle then tried to grasp again. In the 100 random seeds (original environment):

   | Pattern | Seeds | Meaning |
   |---|---|---|
   | `g+ r+ o+ g- g-` | 7, 12, 48, 65, 87, 91, 97 (and evaluation seed 10006) | Placed correctly, not counted, regrasp fails |
   | `g+ r+ o+ g+ r+` | 30, 53, 69 | Placed correctly, not counted, second cycle runs out of steps |
   | `g+ r+ o+ g-` (solved) | 46, 63 | Same cause, but the failed regrasp opened the gripper and the cube then counted |
   | `g- g- g- g- g-` | 55 | The cube spawned out of reach (next point) |

   Fix: `not all(finger_contacts(...))`, the same rule `_held_cube` uses.

2. **Cubes that spawned out of reach.** The far row (x = 0.93) had ±1 cm of jitter. In a test over positions, the arm reached every position with x ≤ 0.93, but at x ≥ 0.935 the far corners (y ≈ 0.25 and 0.45) failed for every yaw, mostly because the approach pose above the cube had no IK solution. Seed 55 spawned at (0.934, 0.448). Fix: `X_REACH_MAX = 0.93` in the task generator.

3. **A finger dragging a placed cube (4 cubes).** After `open` a finger still touched the cube. `_free_start` saw that as a collision and moved the arm with a direct joint-space `servo_to`, without ignoring the cube. In two seeds the cube moved 20.1 and 20.5 mm (to x ≈ 0.947, out of reach). `execute_move(..., linear=True)` is a straight line in joint space, not a vertical rise. Fix: a vertical Cartesian rise of the TCP.

### How it helped the training

- It showed that the environment capped success at about 89–95% (1 cube), so Run 3's 70% was well below a ceiling that was itself not 100%.
- It pointed to the exact code (`_is_placed`, the task generator) instead of more primitive or SAC changes.
- It is cheap (minutes) compared with a training run (2.4–3.4 hours), so it can be run before every training to check that the ceiling is 100%.
- After the fixes, SAC reached 100% evaluation success with 4 and with 8 environments.

### Limits

- The oracle uses perfect parameters. It measures the ceiling of the environment, not the quality of a policy. For example, it does not show how precise the policy's reach target is.
- That the fixes caused SAC's improvement is very likely but not isolated. The runs have one training seed each, and the fixes were not tested one at a time.
- Only the `random` variant with 1 and 4 cubes has been tested.

## Run 5: environment fixes, 4 environments → 100%

Run directory: `runs/sac_1cube_random_fifth_evaluation` · Script: `youbot_pnp/scripts/train_sac.py` · Training steps: 100,000 · Parallel environments: 4 · Wall time: 12,233 s (about 3.4 h, 8 fps)

Command:

```bash
python -m youbot_pnp.scripts.train_sac --steps 100000 --envs 4 --cubes 1 --variant random \
    --learning-starts 5000 --utd 1 --eval-every 5000 --seed 0 --name sac_1cube_random_fifth_evaluation
```

Run 5 uses the original primitives and the same reward, affordance and SAC settings as Run 3 †. The only change is in the environment (two fixes, described below).

† Confirm these were left unchanged.

### Why: the oracle

The oracle (`check_oracle.py`; see [The oracle](#the-oracle-check_oraclepy)) gives the highest success that any policy could reach in this environment. On the tasks used by these runs:

| Tasks (1 cube, `random`) | Before the fixes | After the fixes |
|---|---:|---:|
| Seeds 0–99 | 89/100 (89%) | 100/100 (100%) ‡ |
| Evaluation seeds 10000–10019 | 19/20 (95%) | 20/20 (100%) ‡ |

‡ Measured on a separate copy of the repo (MuJoCo 4.0.1, Python 3.12). **TODO:** re-run `check_oracle.py` locally after the fixes and record the result.

Before the fixes, the oracle failed in 11 of the 100 random tasks and in 1 of the 20 evaluation tasks. Two causes:

1. **Correct placements that were not counted (10 of the 11 failures, and evaluation seed 10006).** The sequence `grasp → reach → open` succeeded and the cube ended 1–4 mm from the goal, but one finger still touched it, so `_is_placed` returned `False` (it required that *no* finger touched the cube). The oracle then tried to grasp again and either failed or ran out of steps (seeds 7, 12, 30, 48, 53, 65, 69, 87, 91, 97). Seeds 46 and 63 were affected in the same way but ended solved after a failed regrasp.
2. **A cube that spawned out of reach (seed 55).** The cube started at (0.934, 0.448), in a far corner where the arm cannot reach the approach pose above the cube. The first grasp failed on every attempt.

### Environment fixes applied before Runs 5 and 6 †

| File | Change | Reason |
|---|---|---|
| `youbot_pnp/rl/env.py`, `_is_placed` | `not any(finger_contacts(...))` → `not all(finger_contacts(...))` | A cube counts as placed unless *both* fingers still hold it. This is the same rule `_held_cube` uses. |
| `youbot_pnp/tasks/task_generator.py` | New `X_REACH_MAX = 0.93`; the x jitter of a start position is clamped so that x ≤ 0.93 | In a separate test, the arm reached every tested position with x ≤ 0.93, but cubes at x ≥ 0.935 in the far corners (y ≈ 0.25 or 0.45) failed for every yaw. Only the far row (x = 0.93) is affected. |

† Confirm these are the changes that were applied. `primitives.py`, the reward, the affordance and the SAC settings are unchanged.

The evaluation tasks change slightly because of the x clamp, and the success predicate is different. Results of Run 5 are therefore **not directly comparable** with Runs 1–4.

### Final evaluation (100k steps)

| Metric | Value |
|---|---:|
| `eval/success_rate` | **100%** (20/20) |
| `eval/mean_reward` | 11.00 ± 0.00 |
| `eval/mean_ep_length` | 3.00 ± 0.00 |
| `primitives/aff` | 0.997 |
| `primitives/success_grasp` | 1.000 |
| `primitives/success_reach` | 1.000 |
| `primitives/success_open` | 1.000 |
| `primitives/failure_rate` | 0 |
| `primitives/chosen_grasp` | 0.346 |
| `primitives/chosen_reach` | 0.327 |
| `primitives/chosen_open` | 0.327 |
| `train/ent_coef` | 0.0141 |
| `train/actor_loss` | −10.1 |
| `train/critic_loss` | 0.0782 |
| `train/learning_rate` | 0.0003 |
| `train/n_updates` | 94,996 |

> **Note.** All 20 evaluation episodes lasted exactly 3 steps, so if `primitives/*` covered only those episodes, each `chosen_*` value would be 0.333. The reported 0.346 / 0.327 / 0.327 suggests that the `PrimitiveStats` callback averages over recent steps, not only over the evaluation episodes. The 20 evaluation episodes give 60 primitive steps, so every fraction would be a multiple of 1/60, and 0.346 is not. Check this before quoting these values as evaluation metrics.

### Behaviour near 100k steps

The first five rows come from the training logs; the last row is the final evaluation. SB3 prints the episode reward rounded, so the exact value is not known.

| Total timesteps | Success rate | Episode reward | Episode length | Aff | Failure rate | Critic loss |
|---:|---:|---:|---:|---:|---:|---:|
| 99,708 | 100% | 11 | 3 | 0.994 | 0 | 0.0499 |
| 99,768 | 100% | 11 | 3 | 0.995 | 0 | 0.0495 |
| 99,828 | 100% | 11 | 3 | 0.995 | 0 | 0.0439 |
| 99,888 | 100% | 11 | 3 | 0.994 | 0 | 0.0433 |
| 99,948 | 100% | 11 | 3 | 0.996 | 0 | 0.0354 |
| **100,000 (evaluation)** | **100%** | **11.00** | **3.00** | **0.997** | **0** | **0.0782** |

### Learning curve (eval on the 20 fixed tasks)

**TODO:** fill this table with the output of the script below. Only the 100k row is known from the logs.

```bash
python youbot_pnp/scripts/export_learning_curve.py runs/<run_name> --every 5000
```

| Steps | Success | Mean reward | Mean length |
|---:|---:|---:|---:|
| 5,000 | TBD | TBD | TBD |
| 10,000 | TBD | TBD | TBD |
| 15,000 | TBD | TBD | TBD |
| 20,000 | TBD | TBD | TBD |
| 25,000 | TBD | TBD | TBD |
| 30,000 | TBD | TBD | TBD |
| 35,000 | TBD | TBD | TBD |
| 40,000 | TBD | TBD | TBD |
| 45,000 | TBD | TBD | TBD |
| 50,000 | TBD | TBD | TBD |
| 55,000 | TBD | TBD | TBD |
| 60,000 | TBD | TBD | TBD |
| 65,000 | TBD | TBD | TBD |
| 70,000 | TBD | TBD | TBD |
| 75,000 | TBD | TBD | TBD |
| 80,000 | TBD | TBD | TBD |
| 85,000 | TBD | TBD | TBD |
| 90,000 | TBD | TBD | TBD |
| 95,000 | TBD | TBD | TBD |
| **100,000** | **100% (20/20)** | **11.00** | **3.00** |

### Interpretation

- **What the numbers show.** An episode reward of 11.00 corresponds to the +1 progress for the placed cube plus the +10 terminal reward, with an affordance penalty close to zero (aff ≈ 0.997). Together with episodes of exactly 3 steps and `chosen_*` ≈ 1/3 each, this is consistent with the policy running `grasp → reach → open` in every episode. Success is 100% both in the evaluation and in the last training rollouts.
- **Comparison with Run 3.** Success rose from 70% to 100%, the mean reward from 7.79 to 11.00 and the mean length from 3.60 to 3.00. This is not a like-for-like comparison, because the environment changed.
- **Hypotheses (not tested).** The fixes may have helped in two ways: they removed the 89–95% ceiling, and they removed a misleading reward signal. Before, a correct placement was sometimes scored as not placed, and the policy was then pushed to grasp again, which could displace the cube. A single run cannot separate these effects. An ablation would need Run 5 repeated with each fix alone.
- **Limits of the evidence.** One training seed and 20 evaluation episodes. With 20/20 successes the 95% lower confidence bound (Wilson) is about 84%. 100/100 on new seeds would raise it to about 96%.

## Run 6: environment fixes, 8 environments → 100%

Run directory: *to be recorded* · Script: `youbot_pnp/scripts/train_sac.py` · Training steps: 100,000 · Parallel environments: 8 † · Wall time: 8,584 s (about 2.4 h, 11 fps)

† The log below is assumed to belong to the 8-environment run (it has a higher fps and a shorter wall time than Run 5). Confirm, and record the exact command. Environment fixes, primitives, reward, affordance and the other SAC settings are as in Run 5 (4 environments), so the only intended difference is the number of parallel environments.

### Final evaluation (100k steps)

| Metric | Value |
|---|---:|
| `eval/success_rate` | **100%** (20/20) |
| `eval/mean_reward` | 11.00 ± 0.00 |
| `eval/mean_ep_length` | 3.00 ± 0.00 |
| `primitives/aff` | 0.998 |
| `primitives/success_grasp` | 1.000 |
| `primitives/success_reach` | 1.000 |
| `primitives/success_open` | 1.000 |
| `primitives/failure_rate` | 0 |
| `primitives/chosen_grasp` | 0.357 |
| `primitives/chosen_reach` | 0.321 |
| `primitives/chosen_open` | 0.321 |
| `train/ent_coef` | 0.0078 |
| `train/actor_loss` | −10.4 |
| `train/critic_loss` | 0.0113 |
| `train/learning_rate` | 0.0003 |
| `train/n_updates` | 94,992 |

> **Note.** As in Run 5, the `primitives/chosen_*` values cannot come from the 20 evaluation episodes: those have 60 primitive steps, and 0.357 and 0.321 are not multiples of 1/60. They are consistent with a window of recent steps (0.357 = 10/28, 0.321 = 9/28).

### Last training log (99,944 steps)

SB3 prints the episode reward rounded, so the exact value is not known. Only one log block was recorded for this run.

| Metric | Value |
|---|---:|
| `rollout/success_rate` | 100% |
| `rollout/ep_rew_mean` | 11 |
| `rollout/ep_len_mean` | 3 |
| `primitives/aff` | 0.997 |
| `primitives/failure_rate` | 0 |
| `primitives/chosen_grasp / reach / open` | 0.321 / 0.357 / 0.321 |
| `train/ent_coef` | 0.00781 |
| `train/critic_loss` | 0.0135 |
| `train/n_updates` | 94,936 |
| `time/episodes` | 26,700 |
| `time/fps` | 11 |
| `time/time_elapsed` | 8,584 s |

## Effect of the number of parallel environments

The same progression was run with 4 and with 8 parallel environments (`SubprocVecEnv`), with the same number of primitive steps (100k) and, as far as is known, the same number of gradient updates (1 per transition, about 95k updates in both).

| Stage | 4 environments | 8 environments |
|---|---:|---:|
| Initial setup (before any change) | 0% | 0% |
| After changing the affordance score ‡ | 70% | 55% |
| After the environment fixes | **100%** (20/20) | **100%** (20/20) |
| Wall time for 100k steps (after the fixes) | 12,233 s (8 fps) | 8,584 s (11 fps) |

‡ **TODO: clarify what changed between the 0% runs and the 55% / 70% runs.** Run 2 above attributes the jump from 0% to 55% to the potential-based reward, while this table follows the description that the affordance score was changed. If both changed, say so here and in Run 2 so that the improvement is not attributed to a single cause. Also record the directory of the second 0% run (only `runs/sac_1cube_random_s0_1117_1910` is listed in Run 1) and the number of environments of Runs 1–3. In the table above, Run 2 (55%) is taken as the 8-environment run and Run 3 (70%) as the 4-environment run.

Observations:

- **With the fixed environment the number of environments does not change the final result.** Both runs reach 100% (20/20) with a mean reward of 11.00 and episodes of exactly 3 steps. 8 environments are faster: the wall time falls by about 30% (from 12,233 s to 8,584 s), which is consistent with more parallel simulation, although the machine load may also have played a part.
- **The 70% vs 55% gap before the fixes is not conclusive.** It is 14/20 against 11/20 evaluation episodes, with one training seed per setting. The 95% (Wilson) intervals overlap (about 48–85% and 34–74%), so these runs do not show that 4 environments were better.
- **The final entropy coefficient is lower with 8 environments** (0.0078 against 0.0141 with 4). This is only an observation; its effect has not been tested.

## Comparison of the main runs

| Metric | Run 1: state-based | Run 2: potential-based | Run 3: original primitives | Run 5: fixed env, 4 envs | Run 6: fixed env, 8 envs |
|---|---:|---:|---:|---:|---:|
| Environment | original | original | original | fixed | fixed |
| Parallel environments § | 4 and 8 | 8 | 4 | 4 | 8 |
| Training steps | 100k | 100k | 100k | 100k | 100k |
| Evaluation success | 0% | 55% | 70% | **100%** (20/20) | **100%** (20/20) |
| Mean evaluation reward | 5.13 | 6.0 | 7.79 | **11.00** | **11.00** |
| Mean episode length | 5.0 | 3.9 | 3.60 | **3.00** | **3.00** |
| Wall time | not recorded | not recorded | 10,893 s | 12,233 s | 8,584 s |
| Main observation | Reward hacking | Learned grasp → reach → open | Better, below target | Ceiling of the environment removed | Same result, about 30% faster |

§ The environment counts of Runs 1–3 come from the description of the 4-environment and 8-environment results (see the TODO above); confirm them. Run 1 appears with both counts because the initial 0% was obtained with 4 and with 8 environments, but only one of those runs is documented above.

Run 4 (extended primitives, 35%) is documented in the follow-up investigation above. Runs 1–3 used the original environment, so comparing them with Runs 5 and 6 is only indicative.

## Four cubes with fixed goals (Runs 7 and 8)

Status: the oracle solves the task, but SAC has not learned it yet. Both runs ended at 0% success and never chose `reach`.

### Goal and design of the `fixed_goals` variant

The target use is 4 cubes that start in diverse positions and always end in the same place. Each cube has its own fixed destination, and the policy has to place it there wherever it starts. The existing `random` variant does not do this: it also shuffles the goals in every episode (they go to the free slots at random). `fixed_goals` is therefore the variant that matches the target use, and it is the only one used for now.

| Cube | Fixed goal (x, y) |
|---|---|
| red | (0.86, 0.44) |
| green | (0.86, 0.26) |
| blue | (0.86, 0.38) |
| yellow | (0.86, 0.32) |

These are the nominal goals of `main.py` (`NOMINAL_GOAL_SLOTS`), so the results can be compared with the FSM. The start positions are a **random permutation of the four far-row slots** (x = 0.93, y = 0.26, 0.32, 0.38, 0.44), so any colour can start in any corner (24 arrangements), plus the usual ±1 cm jitter (x clamped to ≤ 0.93) and a random yaw.

Design decisions and limits:

- **Starts only in the far row.** The near row holds the goals. A cube that started on another cube's goal would block it, and solving that needs the `occupied` / `swap` behaviour (moving a cube to a free buffer slot first), which is a separate step.
- **Limited diversity.** The start positions are the four slots that were checked for reachability and have 6 cm spacing, so the diversity is the 24 arrangements plus jitter and yaw, not arbitrary positions on the table.
- **`random` (goals also vary) is not used for now.** It could be tested later as a generalisation test: the policy would have to read the goal from the observation instead of learning it per colour.

Code: `fixed_goals` is added to `VARIANTS`, and `generate_task` has a branch that takes `goal_slots = NOMINAL_GOAL_SLOTS[:n]` and `start_slots = rng.permutation(_free_slots(goal_slots))[:n]`.

### Oracle check

| Primitives | Tasks | Solved |
|---|---|---:|
| Local machine † | seeds 0–99 | 100/100 |
| Local machine † | evaluation seeds 10000–10019 | 20/20 |
| Separate copy, with the vertical rise ‡ | seeds 0–59 | 60/60 |
| Separate copy, with the vertical rise ‡ | evaluation seeds 10000–10019 | 20/20 |
| Separate copy, without the vertical rise ‡ | seeds 0–39 | 40/40 |

† Confirm which primitives were active in that run. ‡ Measured on a separate copy of the repo (MuJoCo 4.0.1, Python 3.12).

### Runs 7 and 8

Run directories, parallel environments and commands: *to be recorded* (both runs).

| | Run 7: generator only | Run 8: generator + primitive fix |
|---|---:|---:|
| Training steps | 100k | 500k |
| Wall time | 6,786 s (1.9 h, 14 fps) | 26,079 s (7.2 h, 19 fps) |
| Evaluation success | 0% | 0% |
| Mean evaluation reward | −2.88 ± 2.64 | −0.06 ± 0.09 |
| Mean episode length | 20 (the limit) | 20 (the limit) |
| `chosen_grasp / reach / open` (eval log) | 0.563 / 0 / 0.437 | 0.5 / 0 / 0.5 |
| `success_grasp` | 0.794 (training: 0.64–0.73) | 1.0 |
| `success_open` | 1.0 | 1.0 |
| `failure_rate` | 0.116 (training: 0.16–0.22) | 0 |
| `primitives/aff` | 0.947 | 0.996 |
| Training `rollout/ep_rew_mean` | −1.44 | −0.055 |
| `train/ent_coef` | 0.0023 | 0.0006 |
| `train/n_updates` | 94,992 | 494,992 |

For comparison, the 1-cube runs with 100% success ended with `ent_coef` 0.0141 (4 environments) and 0.0078 (8 environments) at 100k steps.

What the logs show:

- **`reach` is never chosen** in either run, and the policy runs `grasp` and `open` until the 20 steps are used up. In Run 8 that means 10 grasps and 10 opens per episode. A grasp raises the potential (the cube is held) and the following open lowers it again, so the net reward is about zero (−0.055 per episode).
- **The primitive fix worked.** With it, `success_grasp` went from 0.64–0.79 to 1.0 and `failure_rate` from 0.12–0.22 to 0. The two runs also differ in length (100k against 500k steps), so the comparison is not isolated.
- **Exploration has stopped.** `ent_coef` fell to 0.0006 in Run 8, about 20 times below the value of the 1-cube runs.

### Why `reach` is not learned

The per-attempt incentive to try `reach` is the same as in the 1-cube setting that did learn. It was computed with the repo's own affordance and potential formulas, for 400 seeds and 400 random reach targets per cube, assuming that the target is uniform in the action space and that the held cube is at its start position. It is a model, not a run of the simulator.

| | 1 cube, `random` | 4 cubes, `fixed_goals` |
|---|---:|---:|
| Start → goal distance (mean, range) | 10.6 cm (5.1–20.1) | 11.0 cm (6.0–20.2) |
| P(random reach within 1 cm of the goal) | 0.24% | 0.25% |
| Mean affordance of a random reach | 0.32 | 0.31 |
| Mean shaping reward | +0.016 | +0.024 |
| Mean affordance penalty | −0.681 | −0.688 |
| **Mean reward of a random reach** | **−0.664** | **−0.664** |

A random reach costs about −0.66 (almost all of it the affordance penalty), against about −0.05 for the grasp/open loop. The cost is the same as with 1 cube, so it does not explain the difference by itself.

**Hypothesis (not tested):** what changes is the payoff of the first lucky placement. With 1 cube it is worth about +11 (+1 for the placement and +10 at the end). With 4 cubes it is worth about +1, because the +10 is only paid when all four cubes are placed. The signal that could make `reach` attractive is about 11 times weaker against the same cost, and the entropy coefficient decays before a lucky reach is found often enough. Even with 1 cube, success needed tens of thousands of steps to appear (Run 2: none for 35k steps).

### The primitive fix: why `primitives.py` changes

After `open`, one finger still touches the cube that has just been placed. The next primitive calls `_free_start`, which treated that contact as a collision and moved the arm with `servo_to`, a straight line in joint space that does not check collisions and does not ignore the cube. With 1 cube the episode ends right after the placement, so this never mattered. With 4 cubes the arm keeps working next to the cubes that are already placed, and in the tests the finger moved a placed cube by about 20 mm, out of reach and out of tolerance.

The fix replaces that movement (and the one in `_lift`) with `_rise`, a vertical rise of the TCP by `APPROACH_HEIGHT` in five IK waypoints, so the finger slides up along the cube. Nothing else changes in `grasp`, `reach` or `open`, and the policy interface is the same.

A correction to an earlier statement: the rise is **not** required for the oracle in `fixed_goals` (40/40 without it). The claim that it was necessary came from the `random` variant (53/60 without, 60/60 with). It matters for the sequences that a learning policy produces and the oracle never does (opening without a reach, grasping again), which is consistent with the change in `success_grasp` from Run 7 to Run 8. It stays in.

### Next experiment: reach snap (Run 9, planned)

`reach` is made to snap onto the goal of the held cube when the target is within 3 cm of it in xy, in the same way `grasp` already snaps onto the nearest unplaced cube. The policy then only has to aim near the goal. By the area of the 3 cm circle over the action space, the chance that a random reach counts as a hit rises from 0.25% to about 6.8%.

| File | Change |
|---|---|
| `rl/primitives.py` | `REACH_SNAP = 0.03`; `Primitive` takes `goal_of`; at the start of `Reach._run`, if a cube is held and the target is within `REACH_SNAP` (xy) of its goal, the target becomes the goal |
| `rl/env.py` | the three primitives receive `goal_of=lambda c: self.cfg.goals[c]`; `_affordance` uses `tol=0.03` for `reach` |
| `rl/affordance.py` | `reach` is measured in xy (`n = 2`), because the snap fixes z |

Test on a separate copy (4 cubes, `fixed_goals`, seeds 10000–10005, one cube per seed):

| Reach aimed at | Cube placed | Mean final error |
|---|---:|---:|
| 0 to 2.8 cm from the goal | 6/6 (every offset tested) | 1.0 mm |
| 4.5 cm from the goal (outside the snap) | 0/6 | 43.4 mm |

The oracle still solves 8/8 tasks.

Design implication: the policy no longer learns fine placement, only which goal to go to. With fixed goals this is what is wanted, but it is a design decision and should be reported as such. `PLACE_TOL` is not changed.

Protocol: a 100k-step screening run (about 2 h at 14 fps), not 500k. Criterion: if `chosen_reach` is still 0 at 100k steps, stop. If the snap is not enough, the next changes are tested one at a time, in this order:

1. A lower affordance weight λ (0.1–0.2), so that a random reach costs about −0.07 to −0.13 instead of −0.66.
2. A curriculum 1 → 2 → 4 cubes (needs the fixed observation size, see below).
3. Oracle demonstrations in the replay buffer (changes the claim from "learned from scratch").

## Next steps

The 1-cube target has been reached in one run. The priority is to confirm it and record it properly before moving on.

1. **Record the runs.** Write down the Run 6 directory and the directory of the second 0% run, confirm which of the environment counts belongs to each run, confirm that the environment fixes and the SAC configuration are as described, re-run `check_oracle.py` locally (expected 100/100 and 20/20), and fill the learning-curve tables of Runs 5 and 6 with `export_learning_curve.py`.
2. **Confirm beyond 20 episodes.** Evaluate `final.zip` (and `best_model.zip`) deterministically on at least 100 seeds that were not used for training. Target: 100/100, which gives a 95% lower bound of about 96%.
3. **Repeat with other training seeds.** Train with `--seed 1` and `--seed 2` under the same configuration and report mean ± standard deviation. Keep the number of environments fixed within a comparison; 8 environments are about 30% faster.
4. **Optional ablation.** Run the oracle with each fix alone (cheap). Retrain with each fix alone only if the report needs to say which one mattered.
5. **4 cubes (`fixed_goals`).** The oracle reaches 100% (100/100 and 20/20 on the local machine †), but Runs 7 and 8 did not learn (0% success, `reach` never chosen). Next: Run 9 with the reach snap as a 100k-step screening run, stopping if `chosen_reach` is still 0. If it does not work, change one factor at a time (lower λ, curriculum, oracle demonstrations). See [Four cubes with fixed goals](#four-cubes-with-fixed-goals-runs-7-and-8). Keep `max_steps` at 20 until the failures are understood (the oracle needs 12 steps).

**What not to change:**

- **Do not raise `PLACE_TOL`.** It would make the task easier than for the FSM and break the comparison in plan step 8.
- **Do not change several reward coefficients and SAC hyperparameters at once.** It would be impossible to tell which change improved learning.
- **Do not use training reward as a substitute for task success.** The target remains the independently evaluated success rate.

## Open decisions for later steps

- **Fixed observation size.** `obs_dim = 9 · n_cubes + 7` changes with the number of cubes, so a 1-cube model cannot keep training on 2 or 4 cubes. With 4 fixed cube slots plus an "active" flag, the curriculum in plan step 7 could reuse models. It is now the main candidate for the 4-cube problem (a curriculum 1 → 2 → 4 cubes), so decide it before models are meant to carry over.
- **MAPLE version (plan step 6).** SB3 SAC has one entropy term for all 7 action dimensions. If the primitive choice stops exploring too early again (for example on harder tasks), the separate entropy for the primitive choice in MAPLE's `PAMDPPolicy` / `SACHybridTrainer` is the structural fix.
- **Buffer slots for `occupied` / `swap`.** The reach affordance currently rewards only the held cube's goal. Moves to a free buffer slot need extra reach keypoints while the goal is blocked.

## Tools

| Script | Purpose |
|---|---|
| `youbot_pnp/scripts/train_sac.py` | Training; logs `eval/success_rate` and `primitives/*` (choice, success, aff) to TensorBoard |
| `youbot_pnp/scripts/watch_policy.py` | Replays a saved model in the viewer on the eval tasks and prints each primitive |
| `youbot_pnp/scripts/check_oracle.py` | Scripted perfect policy through `env.step()`; upper bound for any learned policy and failure diagnosis (`--cubes`, `--variant`, `--seeds`, `--first-seed`) |
| `youbot_pnp/scripts/export_learning_curve.py` | Prints the learning curve of a run as a markdown table from `evaluations.npz` |
| `youbot_pnp/scripts/check_timing.py` | Wall time per primitive (plan step 1) |
| `tensorboard --logdir runs` | Learning curves of all runs |
import numpy as np
from youbot_pnp.rl.affordance import aff_score

cube = np.array([0.90, 0.30, 0.139])
cubes = [cube, np.array([0.86, 0.44, 0.139])]

assert aff_score("grasp", cube, cubes) == 1.0
assert aff_score("grasp", cube + [0.005, 0, 0], cubes) == 1.0           # dentro de la tolerancia
assert abs(aff_score("grasp", cube + [0.025, 0, 0], cubes) - 0.80) < 0.01
assert aff_score("grasp", cube + [0.10, 0, 0], cubes) < 0.35
assert aff_score("grasp", cube, []) == 0.0                              # nada que agarrar
assert aff_score("reach", cube, []) == 0.0                              # no hay cubo en la mano
assert aff_score("reach", cube, [cube]) == 1.0
assert aff_score("open", None, []) == 1.0
print("affordance tests passed")
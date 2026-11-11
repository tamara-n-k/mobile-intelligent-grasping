import numpy as np

AFF_SCALE = 10.0   # 1/m: how much does affordance decrease as one moves away from the keypoint
AFF_TOL = 0.005    # m: within this distance, aff = 1

def aff_score(primitive, target, keypoints, tol=AFF_TOL):
    """
    How useful the chosen parameter is, on a scale of [0, 1]
      grasp: keypoints = unplaced cubes -> distance in xy (z and yaw are ignored by the grasp action)
      reach: keypoints = [goal position of the cube held in the hand] -> distance in xyz
      open:  no parameters -> always 1
    """
    if primitive == "open":
        return 1.0
    if len(keypoints) == 0:
        return 0.0                                  # the primitive has no meaning in this state
    n = 2 if primitive == "grasp" else 3
    kp = np.asarray(keypoints, dtype=float)[:, :n]
    d = np.linalg.norm(kp - np.asarray(target, dtype=float)[:n], axis=1).min()
    return float(1.0 - np.tanh(AFF_SCALE * max(0.0, d - tol)))
import numpy as np


def simplify_polyline(points, eps=2e-3):
    """Ramer-Douglas-Peucker: keeps only the vertices of the path (OMPL interpolates it into N collinear points)"""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-12:
        dists = np.linalg.norm(points - a, axis=1)
    else:
        dists = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / norm
    i = int(np.argmax(dists))
    if dists[i] <= eps:
        return np.array([a, b])
    left = simplify_polyline(points[:i + 1], eps)
    right = simplify_polyline(points[i:], eps)
    return np.vstack([left[:-1], right])

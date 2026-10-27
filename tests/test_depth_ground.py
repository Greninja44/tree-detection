import numpy as np

from src.depth_ground import classify, fit_ground_plane
from src.traversability import OBSTACLE, TRAVERSABLE


def _scene(H=120, W=160):
    v = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    u = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    d = 0.5 + 0.45 * v + 0.05 * u + np.zeros((H, W), np.float32)   # tilted ground plane, closer toward the bottom
    d[40:90, 60:100] += 0.35                                         # box standing on the floor, closer than the plane
    return np.clip(d, 0, 1)


def test_plane_fit_separates_obstacle_from_ground():
    d = _scene()
    seed = np.zeros(d.shape, bool)
    seed[70:118, 30:130] = True
    coef, inl, tau = fit_ground_plane(d, seed, np.random.default_rng(0))
    assert inl > 0.5
    classes, p, valid, r = classify(d, coef, tau)
    assert (classes[100:115, 20:40] == TRAVERSABLE).mean() > 0.9
    assert (classes[50:80, 70:90] == OBSTACLE).mean() > 0.9
    assert p[60, 80] < 0.05 < 0.9 < p[110, 30]

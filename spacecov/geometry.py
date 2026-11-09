"""Small vectorised geometry helpers."""
from __future__ import annotations

import numpy as np


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise vectors along `axis` (zero vectors stay zero)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


# Astrodynamics convention: R1, R3 are *frame* (passive) rotations, so that
# Q = R3(-Omega) R1(-i) R3(-omega) maps perifocal -> inertial (eq. 4.7).
def R1(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def R3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def Rz(phi: float) -> np.ndarray:
    """Active rotation about z by phi (eq. 4.8)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sphere_point(phi: np.ndarray, theta: np.ndarray, r: float | np.ndarray = 1.0) -> np.ndarray:
    """Longitude phi, latitude theta -> Cartesian.  Shape (..., 3)."""
    phi, theta = np.asarray(phi, float), np.asarray(theta, float)
    return np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], axis=-1
    ) * np.asarray(r, float)[..., None]


def frame_from_direction(d: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    """Right-handed rotation matrix whose 3rd column is unit(d).

    Used to orient spacecraft models and FOV cones (local +z = boresight).
    """
    z = unit(np.asarray(d, float))
    for hint in (up_hint, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
        if hint is None:
            continue
        x = np.cross(hint, z)
        if np.linalg.norm(x) > 1e-6:
            x = unit(x)
            return np.column_stack([x, np.cross(z, x), z])
    raise ValueError("degenerate direction")

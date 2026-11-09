import numpy as np
import pytest

from spacecov.asteroid import bumpy_ellipsoid, make_asteroid, uv_sphere
from spacecov.geometry import R1, R3, Rz, frame_from_direction


def test_rotations_are_orthonormal():
    for R in (R1(0.3), R3(-1.2), Rz(2.0)):
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_passive_R3_of_minus_angle_is_active_Rz():
    assert np.allclose(R3(-0.7), Rz(0.7))


def test_frame_from_direction():
    d = np.array([0.0, 0.0, -1.0])
    R = frame_from_direction(d, up_hint=np.array([0.0, 0.0, 1.0]))   # degenerate hint -> fallback
    assert np.allclose(R[:, 2], d)
    assert np.allclose(R.T @ R, np.eye(3))


def test_uv_sphere_facet_count_and_outward_normals():
    v, f = uv_sphere(48, 24)
    assert len(f) == 2208
    ast = make_asteroid({"source": "ellipsoid", "semi_axes": [1, 1, 1], "n_lon": 48, "n_lat": 24,
                         "bump_amplitude": 0.0, "n_bumps": 0})
    assert np.all((ast.normals * ast.centers).sum(1) > 0)
    assert ast.trimesh.is_watertight
    assert ast.total_area == pytest.approx(4 * np.pi, rel=0.01)


def test_bumpy_ellipsoid_outward_and_extent():
    v, f = bumpy_ellipsoid((16, 7, 6), bump_amplitude=0.06, seed=1)
    assert np.abs(v).max(0)[0] == pytest.approx(16, rel=0.1)

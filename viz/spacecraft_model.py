"""True-scale spacecraft and FOV-cone meshes.  Local frame: +z = scanner boresight, units km.

The spacecraft is drawn at its real size (SPACECRAFT_SPAN_M across the solar panels), i.e.
about 1/6000 of the length of Eros.  From a distance it is far smaller than a pixel, so the
scene also draws a fixed-size screen marker on top of it (see scene.py) - the marker is
not to scale, the model is.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from spacecov.config import SPACECRAFT_BUS_M, SPACECRAFT_SPAN_M

M = 1e-3            # metres -> km


def spacecraft_parts() -> dict[str, pv.PolyData]:
    b = SPACECRAFT_BUS_M * M
    wing = (SPACECRAFT_SPAN_M - SPACECRAFT_BUS_M) / 2 * M                     # one panel length
    bus = pv.Cube(center=(0, 0, 0), x_length=b, y_length=b, z_length=1.2 * b)
    scanner = pv.Cylinder(center=(0, 0, 0.75 * b), direction=(0, 0, 1), radius=0.22 * b, height=0.35 * b,
                          resolution=24)
    lens = pv.Disc(center=(0, 0, 0.93 * b), inner=0.0, outer=0.17 * b, normal=(0, 0, 1), c_res=24)
    dish = pv.Cone(center=(0, 0, -0.8 * b), direction=(0, 0, 1), height=0.3 * b, radius=0.45 * b,
                   resolution=24)
    panels = (pv.Cube(center=(b / 2 + wing / 2, 0, 0), x_length=wing, y_length=0.75 * b, z_length=0.02 * b)
              + pv.Cube(center=(-b / 2 - wing / 2, 0, 0), x_length=wing, y_length=0.75 * b,
                        z_length=0.02 * b))
    return {"bus": bus + dish, "scanner": scanner, "lens": lens, "panels": panels}


CONE_RAYS = 48


def cone_mesh(n_rays: int = CONE_RAYS) -> pv.PolyData:
    """Topology of a scanner cone: apex (point 0) + n_rays rim points, as a triangle fan.

    The rim points are set every frame to where the cone's edge rays hit the asteroid (or stop
    at the scanner's usable range), so the cone ends exactly on the imaged surface."""
    pts = np.zeros((n_rays + 1, 3))
    faces = np.hstack([[3, 0, 1 + j, 1 + (j + 1) % n_rays] for j in range(n_rays)])
    return pv.PolyData(pts, faces=faces)


def cone_lines(n_rays: int = CONE_RAYS) -> pv.PolyData:
    """Rim loop + 4 edge lines from the apex."""
    pts = np.zeros((n_rays + 1, 3))
    lines = [n_rays + 1] + list(range(1, n_rays + 1)) + [1]
    for j in range(0, n_rays, n_rays // 4):
        lines += [2, 0, j + 1]
    return pv.PolyData(pts, lines=np.array(lines))


def cone_directions(fov_rad: float, rotation: np.ndarray, n_rays: int = CONE_RAYS) -> np.ndarray:
    """Unit directions of the cone's edge rays; rotation columns x, y, z (= boresight)."""
    a = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    h = fov_rad / 2.0
    local = np.column_stack([np.sin(h) * np.cos(a), np.sin(h) * np.sin(a), np.full_like(a, np.cos(h))])
    return local @ rotation.T


def pose_matrix(position: np.ndarray, rotation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation * scale
    T[:3, 3] = position
    return T

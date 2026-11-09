"""Asteroid surface model: triangle mesh + per-facet quantities (A_f, c_f, n_f).

Builders
--------
* ``bumpy_ellipsoid``  procedural ellipsoid (fallback when the Eros shape file is missing)
* ``load_shape``       PDS vertex/plate ``.tab`` (Gaskell Eros) or any ``.obj``/``.ply``/``.stl``
* ``make_asteroid``    build from a JSON-able spec dict (used by scenarios and the viewer)

The mesh lives in the body frame B (z = spin axis), units km.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import sphere_point, unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EROS_RAW = REPO_ROOT / "data" / "eros" / "raw" / "ver64q.tab"
# 216 Kleopatra, radar shape model (Ostro et al. 2000), NASA PDS EAR-A-5-DDR-RADARSHAPE-MODELS-V2.0
KLEOPATRA_RAW = REPO_ROOT / "data" / "kleopatra" / "raw" / "216kleopatra.tab"
CACHE_DIR = REPO_ROOT / "data" / "eros" / "cache"


# --------------------------------------------------------------------------
# Asteroid
# --------------------------------------------------------------------------
@dataclass
class Asteroid:
    name: str
    vertices: np.ndarray                     # (V, 3) km, body frame
    faces: np.ndarray                        # (F, 3) int, counter-clockwise seen from outside
    spec: dict[str, Any] | None = None       # spec used to build it (for saving scenarios)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=float)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        tri = self.vertices[self.faces]                          # (F, 3, 3)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.centers = tri.mean(axis=1)                          # c_f
        self.areas = 0.5 * np.linalg.norm(cross, axis=1)         # A_f
        self.normals = unit(cross)                               # n_f (outward)

    # -- derived --------------------------------------------------------------
    @property
    def n_facets(self) -> int:
        return len(self.faces)

    @property
    def r_max(self) -> float:
        """Farthest surface point from the centre (R_max)."""
        return float(np.linalg.norm(self.vertices, axis=1).max())

    @property
    def total_area(self) -> float:
        return float(self.areas.sum())

    @cached_property
    def trimesh(self):
        import trimesh
        return trimesh.Trimesh(self.vertices, self.faces, process=False)

    @cached_property
    def ray(self):
        """Ray intersector (Embree if ``embreex`` is installed, else rtree/pure numpy)."""
        return self.trimesh.ray


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def uv_sphere(n_lon: int, n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit UV sphere with 2*n_lon*(n_lat-1) triangles (48 x 24 -> 2208, as in the report)."""
    lat = np.linspace(np.pi / 2, -np.pi / 2, n_lat + 1)[1:-1]            # rings, north -> south
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    ring_pts = sphere_point(LON.ravel(), LAT.ravel())
    verts = np.vstack([[0, 0, 1], ring_pts, [0, 0, -1]])
    north, south = 0, len(verts) - 1

    def idx(ring, j):
        return 1 + ring * n_lon + (j % n_lon)

    faces = []
    for j in range(n_lon):                                               # north cap
        faces.append([north, idx(0, j), idx(0, j + 1)])
    for r in range(n_lat - 2):                                           # bands
        for j in range(n_lon):
            a, b = idx(r, j), idx(r, j + 1)
            c, d = idx(r + 1, j), idx(r + 1, j + 1)
            faces += [[a, c, b], [b, c, d]]
    for j in range(n_lon):                                               # south cap
        faces.append([south, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)])
    return verts, _orient_outward(verts, np.array(faces))


def bumpy_ellipsoid(semi_axes=(16.0, 7.0, 6.0), n_lon: int = 48, n_lat: int = 24,
                    bump_amplitude: float = 0.06, n_bumps: int = 40, seed: int = 21):
    """Ellipsoid with random Gaussian bumps/craters (relative radial amplitude)."""
    verts, faces = uv_sphere(n_lon, n_lat)
    rng = np.random.default_rng(seed)
    if bump_amplitude > 0 and n_bumps > 0:
        centres = unit(rng.normal(size=(n_bumps, 3)))
        widths = rng.uniform(0.15, 0.5, n_bumps)                         # radians
        heights = rng.uniform(-1.0, 1.0, n_bumps) * bump_amplitude
        ang = np.arccos(np.clip(verts @ centres.T, -1, 1))               # (V, n_bumps)
        radial = 1.0 + (heights * np.exp(-0.5 * (ang / widths) ** 2)).sum(axis=1)
        verts = verts * radial[:, None]
    verts = verts * np.asarray(semi_axes, float)
    return verts, _orient_outward(verts, faces)


def read_pds_vertex_tab(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a PDS3 vertex-facet table (Gaskell shape models).

    Layout: ``n_vert n_plate`` / n_vert rows ``id x y z`` (km) / n_plate rows ``id v1 v2 v3`` (1-based).
    Only numbers are read; nothing in the file is executed.
    """
    path = Path(path)
    with path.open() as fh:
        nv, nf = (int(x) for x in fh.readline().split()[:2])
    v = np.loadtxt(path, skiprows=1, max_rows=nv, usecols=(1, 2, 3), dtype=float)
    f = np.loadtxt(path, skiprows=1 + nv, max_rows=nf, usecols=(1, 2, 3), dtype=np.int64) - 1
    if v.shape != (nv, 3) or f.shape != (nf, 3) or f.min() < 0 or f.max() >= nv:
        raise ValueError(f"{path}: malformed vertex-facet table")
    return v, f


def read_vf_tab(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a PDS radar shape model: rows ``v x y z`` (km) then ``f i j k`` (1-based), like an OBJ file."""
    v, f = [], []
    with Path(path).open() as fh:
        for line in fh:
            p = line.split()
            if len(p) == 4 and p[0] == "v":
                v.append([float(x) for x in p[1:]])
            elif len(p) == 4 and p[0] == "f":
                f.append([int(x) - 1 for x in p[1:]])
    v, f = np.array(v, float), np.array(f, np.int64)
    if v.ndim != 2 or f.ndim != 2 or f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"{path}: malformed v/f shape table")
    return v, f


def center_on_mass(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Shift the vertices so the centre of mass (uniform density) is the origin: orbits are about it."""
    tri = v[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0     # signed tetrahedra
    com = (vol[:, None] * tri.sum(axis=1) / 4.0).sum(axis=0) / vol.sum()
    return v - com


def load_shape(path: str | Path, target_faces: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".tab":
        with path.open() as fh:
            first = fh.readline().split()
        v, f = read_vf_tab(path) if first and first[0] == "v" else read_pds_vertex_tab(path)
    else:
        import trimesh
        m = trimesh.load(path, force="mesh", process=True)
        v, f = np.asarray(m.vertices), np.asarray(m.faces)
    if target_faces and len(f) > target_faces:
        v, f = decimate(v, f, target_faces)
    return v, _orient_outward(v, f)


def decimate(vertices: np.ndarray, faces: np.ndarray, target_faces: int):
    """Quadric decimation (VTK) to roughly ``target_faces`` triangles."""
    import pyvista as pv

    mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    out = mesh.decimate(1.0 - target_faces / len(faces), volume_preservation=True)
    out = out.clean().triangulate()
    f = out.faces.reshape(-1, 4)[:, 1:]
    v = np.asarray(out.points, float)
    # drop degenerate triangles
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return v, f[area > 1e-12]


def _orient_outward(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Flip winding if the signed volume is negative (normals must point outward)."""
    tri = v[f]
    signed_vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    return f[:, ::-1].copy() if signed_vol < 0 else f


# --------------------------------------------------------------------------
# Spec-based construction (scenarios, viewer)
# --------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    "eros_ellipsoid": {     # Table 4.4 fallback: 17.2 x 5.6 x 5.6 km semi-axes
        "source": "ellipsoid", "semi_axes": [17.2, 5.6, 5.6], "n_lon": 64, "n_lat": 32,
        "bump_amplitude": 0.0, "n_bumps": 0, "seed": 0},
    "eros": {"source": "eros", "faces": 1500},
    "kleopatra": {"source": "kleopatra"},
}

PRESET_LABELS = {
    "eros_ellipsoid": "Eros ellipsoid 17.2x5.6x5.6 km (fallback)",
    "eros": "433 Eros - Gaskell shape model",
    "kleopatra": "216 Kleopatra - radar shape model",
}


def display_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Spec of the mesh to DRAW: the full-resolution Gaskell model for Eros, else the same mesh.

    The optimiser works on a decimated copy (1500-6000 facets) for speed; the viewer draws the
    real 49,152-plate shape and maps results onto it."""
    return {"source": "eros", "faces": 0} if spec.get("source") == "eros" else dict(spec)


def make_asteroid(spec: dict[str, Any] | str) -> Asteroid:
    """Build an asteroid from a spec dict or preset name.

    Spec keys: ``source`` (ellipsoid | eros | kleopatra | file) + builder args.
    """
    if isinstance(spec, str):
        spec = PRESETS[spec]
    spec = {k: v for k, v in spec.items() if k != "priority"}     # old static-layer key, ignored
    v, f, name = _geometry(json.dumps(spec, sort_keys=True))
    return Asteroid(name, v.copy(), f.copy(), spec)


@lru_cache(maxsize=16)
def _geometry(key: str):
    spec = json.loads(key)
    src = spec.get("source", "ellipsoid")
    if src == "ellipsoid":
        args = {k: spec[k] for k in ("semi_axes", "n_lon", "n_lat", "bump_amplitude", "n_bumps", "seed")
                if k in spec}
        v, f = bumpy_ellipsoid(**args)
        a = args.get("semi_axes", (16, 7, 6))
        return v, f, f"Ellipsoid {2*a[0]:g}x{2*a[1]:g}x{2*a[2]:g} km"
    if src == "eros":
        target = int(spec.get("faces") or 0)            # 0 / None -> full 49,152-plate model
        if not EROS_RAW.exists():
            warnings.warn(f"{EROS_RAW} not found - using the 17.2x5.6x5.6 km ellipsoid fallback")
            v, f, _ = _geometry(json.dumps(PRESETS["eros_ellipsoid"], sort_keys=True))
            return v, f, "Eros (ellipsoid fallback)"
        cache = CACHE_DIR / (f"eros_{target}.npz" if target else "eros_full.npz")
        if cache.exists():
            d = np.load(cache)
            return d["v"], d["f"], "433 Eros"
        v, f = load_shape(EROS_RAW, target)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, v=v, f=f)
        return v, f, "433 Eros"
    if src == "kleopatra":                              # 2048 vertices / 4092 facets, used as is
        v, f = load_shape(KLEOPATRA_RAW, spec.get("faces"))
        return center_on_mass(v, f), f, "216 Kleopatra"
    if src == "file":
        v, f = load_shape(spec["path"], spec.get("faces"))
        return v, f, Path(spec["path"]).stem
    raise ValueError(f"unknown asteroid source {src!r}")

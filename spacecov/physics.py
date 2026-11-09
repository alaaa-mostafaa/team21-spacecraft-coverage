"""Physics of the mission (no objectives, no optimisation): what one spacecraft sees.

  1. Kepler propagation (inertial frame I)
  2. rotation into the asteroid body frame B (uniform spin about z)
  3. per-step facet visibility: cone (a), emission (b), occlusion (c), resolution (d),
     sunlight incl. cast shadows (e); eclipses of the spacecraft and its battery charge

``track()`` runs stages 1-3 for ONE spacecraft.  Everything here depends only on that
spacecraft's own orbit, so the engine caches tracks per spacecraft (``engine.Engine``).

The functions are duck-typed on the config object: they read physical constants
(mu, omega_ast, dt, scanners, lighting, battery ...), never objective weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asteroid import Asteroid
from .geometry import R1, R3, Rz

TYPE_UNUSED = 0
# element vector of OrbitalSolution.to_vector / from_vector (general orbit + camera)
N_VARS = 7
VAR_NAMES = ("a", "e", "inc", "raan", "argp", "M0", "type")


@dataclass
class OrbitalSpacecraft:
    a: float            # semi-major axis [km]
    e: float = 0.0      # eccentricity
    inc: float = 0.0    # inclination [deg]
    raan: float = 0.0   # longitude of ascending node [deg]
    argp: float = 0.0   # argument of periapsis [deg]
    M0: float = 0.0     # mean anomaly at t = 0 [deg]
    type: int = 1       # camera type 1 (narrow) / 2 (wide); 0 = slot unused

    @property
    def active(self) -> bool:
        return int(self.type) != TYPE_UNUSED

    @property
    def periapsis(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def apoapsis(self) -> float:
        return self.a * (1.0 + self.e)

    def rotation(self) -> np.ndarray:
        """Q = R3(-RAAN) R1(-inc) R3(-argp): perifocal -> inertial."""
        return R3(-np.deg2rad(self.raan)) @ R1(-np.deg2rad(self.inc)) @ R3(-np.deg2rad(self.argp))

    def period(self, mu: float) -> float:
        return 2.0 * np.pi / mean_motion(self.a, mu)

    def speed(self, r: float, mu: float) -> float:
        """Vis-viva speed [km/s] at distance r from the asteroid centre."""
        return float(np.sqrt(mu * (2.0 / r - 1.0 / self.a)))

    def key(self) -> tuple:
        return (self.a, self.e, self.inc, self.raan, self.argp, self.M0, int(self.type))

    def to_dict(self):
        return {"type": int(self.type), "a": self.a, "e": self.e, "inc": self.inc,
                "raan": self.raan, "argp": self.argp, "M0": self.M0}


@dataclass
class OrbitalSolution:
    spacecraft: list[OrbitalSpacecraft]          # all slots, including unused ones

    @property
    def n(self) -> int:
        return len(self.spacecraft)

    @property
    def n_active(self) -> int:
        return sum(s.active for s in self.spacecraft)

    def active(self) -> "OrbitalSolution":
        return OrbitalSolution([s for s in self.spacecraft if s.active])

    @property
    def types(self) -> np.ndarray:
        return np.array([s.type for s in self.spacecraft], dtype=int)

    def to_vector(self) -> np.ndarray:
        return np.array([[s.a, s.e, s.inc, s.raan, s.argp, s.M0, s.type]
                         for s in self.spacecraft], float).ravel()

    @classmethod
    def from_vector(cls, X) -> "OrbitalSolution":
        X = np.asarray(X, float).reshape(-1, N_VARS)
        return cls([OrbitalSpacecraft(*(float(v) for v in row[:6]), type=int(np.clip(np.rint(row[6]), 0, 2)))
                    for row in X])


@dataclass
class Track:
    """Stage 1-3 result for one spacecraft."""
    r_inertial: np.ndarray           # (K, 3) km
    r_body: np.ndarray               # (K, 3) km
    first_seen: np.ndarray           # (F,) int, first step the facet was imaged (K if never)
    imaging: np.ndarray              # (K,) bool, a frame with at least one valid facet was taken
    eclipse: np.ndarray              # (K,) bool, spacecraft in the asteroid's shadow
    soc: np.ndarray                  # (K,) battery state of charge [0, 1]
    dv: float                        # km/s, Delta-V to reach the orbit from the arrival orbit
    e_elec: float                    # J, electrical energy used over the mission
    visible: np.ndarray | None       # (K, F) bool V_ifk, only kept when asked for

    @property
    def n_img(self) -> int:
        return int(self.imaging.sum())


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------
# Stage 1-2: propagation
# --------------------------------------------------------------------------
def mean_motion(a, mu):
    """n = sqrt(mu / a^3)."""
    return np.sqrt(mu / np.asarray(a, float) ** 3)


def solve_kepler(M, e, tol: float = 1e-12, max_iter: int = 50):
    """Solve M = E - e sin E for E with Newton's method from E = M."""
    M = np.asarray(M, float)
    e = np.broadcast_to(np.asarray(e, float), M.shape)
    E = M.copy()
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE), initial=0.0) < tol:
            break
    return E


def propagate_one(s: OrbitalSpacecraft, times: np.ndarray, mu: float) -> np.ndarray:
    """Inertial positions r^I(t) of one spacecraft, shape (K, 3)."""
    times = np.atleast_1d(np.asarray(times, float))
    e = float(np.clip(s.e, 0.0, 0.99))
    M = np.deg2rad(s.M0) + mean_motion(s.a, mu) * times
    E = solve_kepler(np.mod(M, 2 * np.pi), e)
    pf = np.stack([s.a * (np.cos(E) - e), s.a * np.sqrt(1 - e * e) * np.sin(E),
                   np.zeros_like(E)], axis=-1)
    return pf @ s.rotation().T


def propagate(sol: OrbitalSolution, times: np.ndarray, mu: float) -> np.ndarray:
    """Inertial positions r_i^I(t), shape (N, K, 3)."""
    times = np.atleast_1d(np.asarray(times, float))
    out = np.empty((sol.n, len(times), 3))
    for i, s in enumerate(sol.spacecraft):
        out[i] = propagate_one(s, times, mu)
    return out


def to_body(r_inertial: np.ndarray, times: np.ndarray, omega_ast: float) -> np.ndarray:
    """r^B = Rz(w t)^T r^I.  Works on (..., K, 3)."""
    ang = omega_ast * np.asarray(times, float)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = r_inertial[..., 0], r_inertial[..., 1], r_inertial[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def orbit_curve(sc: OrbitalSpacecraft, n: int = 361) -> np.ndarray:
    """Closed inertial ellipse (n, 3), sampled uniformly in eccentric anomaly."""
    E = np.linspace(0, 2 * np.pi, n)
    e = float(np.clip(sc.e, 0.0, 0.99))
    pf = np.stack([sc.a * (np.cos(E) - e), sc.a * np.sqrt(1 - e * e) * np.sin(E), np.zeros_like(E)], -1)
    return pf @ sc.rotation().T


def body_rotation(t: float, omega_ast: float) -> np.ndarray:
    """Rotation that takes body-frame coordinates to inertial at time t."""
    return Rz(omega_ast * t)


# --------------------------------------------------------------------------
# Sunlight, eclipses, propulsion, battery
# --------------------------------------------------------------------------
def sun_body(times: np.ndarray, cfg) -> np.ndarray:
    """Unit Sun direction in the body frame at each time, (K, 3)."""
    return to_body(np.broadcast_to(cfg.sun_inertial, (len(times), 3)), times, cfg.omega_ast)


def lit_mask(ast: Asteroid, cfg, times: np.ndarray | None = None) -> np.ndarray | None:
    """(e) L[k, f]: facet f is sunlit at step k, or None when lighting is off.

    Lit = the Sun is at least 90 - incidence_max above the facet's horizon
          AND (with cfg.shadows) a ray from the facet towards the Sun hits no other facet.
    Independent of the decision variables, so it is computed once per mission."""
    if not cfg.lighting:
        return None
    times = cfg.times if times is None else times
    s = sun_body(times, cfg)                                              # (K, 3)
    lit = s @ ast.normals.T >= np.cos(np.deg2rad(cfg.incidence_max_deg))
    if cfg.shadows and lit.any():
        kk, ff = np.nonzero(lit)
        origins = ast.centers + 1e-4 * ast.r_max * ast.normals
        shadowed = np.zeros(len(kk), bool)
        for j in range(0, len(kk), 200_000):
            sl = slice(j, j + 200_000)
            shadowed[sl] = ast.ray.intersects_any(origins[ff[sl]], s[kk[sl]])
        lit[kk[shadowed], ff[shadowed]] = False
    return lit


def illuminable(ast: Asteroid, lit: np.ndarray | None) -> np.ndarray:
    """Facets that are lit at least once during the mission (they make up A*)."""
    return np.ones(ast.n_facets, bool) if lit is None else lit.any(axis=0)


def eclipse_track(r_inertial: np.ndarray, cfg) -> np.ndarray:
    """Spacecraft in the shadow of Eros (a sphere of its mean radius): r.s < 0 and |r - (r.s)s| < R."""
    s = cfg.sun_inertial
    proj = r_inertial @ s
    perp = np.linalg.norm(r_inertial - proj[:, None] * s, axis=1)
    return (proj < 0) & (perp < cfg.eclipse_radius_km)


def delta_v(s: OrbitalSpacecraft, cfg) -> float:
    """Delta-V [km/s] from the circular equatorial arrival orbit r0 to the target orbit.

    plane change at r0:          2 v0 sin(inc / 2)
    burn 1 at r0 onto a transfer ellipse that touches the target periapsis rp
    burn 2 at rp to move the far apse from r0 to the target apoapsis ra
    (phasing in RAAN, argp, M0 assumed free)."""
    mu, r0, v0 = cfg.mu, cfg.r0, cfg.v0
    rp, a = s.periapsis, s.a
    at = 0.5 * (r0 + rp)
    dv_plane = 2.0 * v0 * np.sin(np.deg2rad(s.inc) / 2.0)
    dv1 = abs(np.sqrt(mu * (2.0 / r0 - 1.0 / at)) - v0)
    dv2 = abs(np.sqrt(mu * (2.0 / rp - 1.0 / a)) - np.sqrt(mu * (2.0 / rp - 1.0 / at)))
    return float(dv_plane + dv1 + dv2)


def battery_track(imaging: np.ndarray, eclipse: np.ndarray, cfg) -> np.ndarray:
    """State of charge after every step, starting full.

    generation: p_solar in sunlight, 0 in eclipse
    use:        p_base + (frame + downlink energy) / dt in steps where a frame is taken"""
    dt_h = cfg.dt / 3600.0
    use = cfg.p_base_w + imaging * ((cfg.e_img_j + cfg.e_tx_j) / cfg.dt)
    net_wh = (np.where(eclipse, 0.0, cfg.p_solar_w) - use) * dt_h
    soc = np.empty(len(net_wh))
    q = cfg.battery_wh
    for k, d in enumerate(net_wh):
        q = min(cfg.battery_wh, q + d)
        soc[k] = q / cfg.battery_wh
    return soc


# --------------------------------------------------------------------------
# Stage 3: visibility
# --------------------------------------------------------------------------
def visibility_track(r_body: np.ndarray, scanner_type: int, ast: Asteroid, cfg,
                     lit: np.ndarray | None = None, chunk: int = 64) -> np.ndarray:
    """V_ifk for one spacecraft along its body-frame track r_body (K, 3) -> (K, F) bool.

    With q = r - c_f (facet -> spacecraft), l = |q| and the nadir boresight b = -r/|r|,
    every test needs only the products r.c_f and r.n_f, so each chunk of time steps is two
    matrix products instead of a (K, F, 3) array:
        l^2       = |r|^2 - 2 r.c_f + |c_f|^2
        b.(-q/l)  = (|r| - r.c_f/|r|) / l           (a) cone
        n_f.q/l   = (r.n_f - n_f.c_f) / l           (b) emission
    """
    K, F = len(r_body), ast.n_facets
    cos_half = np.cos(cfg.fov(scanner_type) / 2.0)
    cos_eta = np.cos(np.deg2rad(cfg.eta_max_deg))
    l_max = cfg.max_range(scanner_type)                                   # (d) GSD <= GSD_max
    V = np.zeros((K, F), dtype=bool)
    c, n = ast.centers, ast.normals
    cc = np.einsum("ij,ij->i", c, c)
    nc = np.einsum("ij,ij->i", n, c)

    for k0 in range(0, K, chunk):
        r = r_body[k0:k0 + chunk]                                         # (k, 3)
        rn = np.linalg.norm(r, axis=1)[:, None]                           # (k, 1)
        rc = r @ c.T                                                      # (k, F)
        ell = np.sqrt(np.maximum(rn * rn - 2.0 * rc + cc, 1e-12))         # range l
        cand = ell <= l_max                                               # (d) resolution
        cand &= (rn - rc / rn) >= cos_half * ell                          # (a) cone
        cand &= (r @ n.T - nc) >= cos_eta * ell                           # (b) emission
        if lit is not None:
            cand &= lit[k0:k0 + chunk]                                    # (e) sunlight
        if cfg.occlusion and cand.any():                                  # (c) occlusion
            kk, ff = np.nonzero(cand)
            qhat = (r[kk] - c[ff]) / ell[kk, ff][:, None]
            origins = c[ff] + 1e-4 * ast.r_max * n[ff]
            blocked = ast.ray.intersects_any(origins, qhat)
            cand[kk[blocked], ff[blocked]] = False
        V[k0:k0 + chunk] = cand
    return V


def track(s: OrbitalSpacecraft, ast: Asteroid, cfg, times: np.ndarray,
          lit: np.ndarray | None, keep_visible: bool = False) -> Track:
    """Stages 1-3 for one (active) spacecraft."""
    r_I = propagate_one(s, times, cfg.mu)
    r_B = to_body(r_I, times, cfg.omega_ast)
    V = visibility_track(r_B, s.type, ast, cfg, lit)
    seen = V.any(axis=0)
    first = np.where(seen, V.argmax(axis=0), len(times))
    imaging = V.any(axis=1)
    ecl = eclipse_track(r_I, cfg)
    soc = battery_track(imaging, ecl, cfg)
    e_elec = cfg.p_base_w * cfg.duration + imaging.sum() * (cfg.e_img_j + cfg.e_tx_j)
    return Track(r_I, r_B, first, imaging, ecl, soc, delta_v(s, cfg), float(e_elec),
                 V if keep_visible else None)


def min_separation(r_I: np.ndarray) -> float:
    N = r_I.shape[0]
    best = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            best = min(best, float(np.linalg.norm(r_I[i] - r_I[j], axis=-1).min()))
    return best

"""Decision variables: what an optimisation algorithm is allowed to change.

    x = [ a_1, e_1, inc_1, raan_1, argp_1, M0_1, t_1,  ...,  a_Nmax, ..., t_Nmax ]     (7 N_max numbers)

    a        semi-major axis [km]                 e      eccentricity
    inc      inclination [deg], 0 .. 180           raan   longitude of the ascending node [deg]
    argp     argument of periapsis [deg]           M0     mean anomaly at t = 0 [deg]
    t        slot switch: 0 = slot unused, 1 = spacecraft with the 6 deg camera (integer)

So the number of spacecraft N = #{t_i != 0} is a decision too.  The vector has a fixed length so
every algorithm (SA, GA, PSO, ...) can treat it as one point in a box.  ``DesignSpace`` gives the
box, the kind of every variable (integer / angle that wraps) and ``repair``, which maps any vector
onto a valid design (angles wrap, a and inc reflect at their bounds, e is clipped so periapsis and
apoapsis stay in the allowed band, at least one spacecraft flies).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OrbitalConfig
from .geometry import R1, R3
from .physics import mean_motion

N_VARS = 7
VAR_NAMES = ("a", "e", "inc", "raan", "argp", "M0", "type")
TYPE_UNUSED = 0
ANGLE_COLS = (3, 4, 5)          # RAAN, argp, M0 wrap around 360 deg
TYPE_COL = 6


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


class DesignSpace:
    """Layout, bounds, variable kinds and repair of the decision vector for ``n_max`` slots.

    With ``fixed_n=True`` every slot must fly (the fleet size is fixed to n_max)."""

    def __init__(self, n_max: int, r_min: float, cfg: OrbitalConfig, fixed_n: bool = False):
        self.n = int(n_max)
        self.r_min = float(r_min)
        self.cfg = cfg
        self.fixed_n = fixed_n

    @property
    def dim(self) -> int:
        return N_VARS * self.n

    @property
    def names(self) -> list[str]:
        return [f"{v}_{i + 1}" for i in range(self.n) for v in VAR_NAMES]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Type is searched on [-0.5, max_type + 0.5) and rounded, so every allowed type is equally likely."""
        lo = np.tile([self.r_min, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 if self.fixed_n else -0.5], self.n)
        hi = np.tile([self.cfg.r_max, self.cfg.e_max, 180.0, 360.0, 360.0, 360.0, self.cfg.max_type + 0.4999], self.n)
        return lo, hi

    @property
    def integer_mask(self) -> np.ndarray:
        """Variables that only take whole values (the slot switch)."""
        m = np.zeros((self.n, N_VARS), bool)
        m[:, TYPE_COL] = True
        return m.ravel()

    @property
    def periodic_mask(self) -> np.ndarray:
        """Angles that wrap around 360 deg."""
        m = np.zeros((self.n, N_VARS), bool)
        m[:, list(ANGLE_COLS)] = True
        return m.ravel()

    def active_mask(self, x) -> np.ndarray:
        """Variables that currently matter: the orbits of flying spacecraft and every slot switch."""
        y = np.asarray(x, float).reshape(self.n, N_VARS)
        m = np.zeros((self.n, N_VARS), bool)
        m[np.rint(y[:, TYPE_COL]) > 0, :] = True
        m[:, TYPE_COL] = True
        return m.ravel()

    def e_limit(self, a: np.ndarray) -> np.ndarray:
        """Largest eccentricity that keeps periapsis >= r_min and apoapsis <= r_max (g1, g2 <= 0)."""
        return np.clip(np.minimum.reduce([np.full_like(a, self.cfg.e_max), 1.0 - self.r_min / a,
                                          self.cfg.r_max / a - 1.0]), 0.0, None)

    def repair(self, X: np.ndarray) -> np.ndarray:
        """Map any vector into the box, onto orbits that respect g1 and g2, with >= 1 active slot."""
        lo, hi = self.bounds()
        Y = np.asarray(X, float).reshape(self.n, N_VARS).copy()
        lo, hi = lo.reshape(self.n, N_VARS), hi.reshape(self.n, N_VARS)
        Y[:, ANGLE_COLS] %= 360.0
        Y[:, 2] = _reflect(Y[:, 2], 0.0, 180.0)                       # inclination
        Y[:, 0] = _reflect(Y[:, 0], lo[:, 0], hi[:, 0])               # a
        Y[:, 1] = np.clip(np.abs(Y[:, 1]), 0.0, self.e_limit(Y[:, 0]))
        raw = Y[:, 6].copy()
        Y[:, 6] = np.clip(np.rint(Y[:, 6]), 1 if self.fixed_n else 0, self.cfg.max_type)
        if not np.any(Y[:, 6] > 0):                                    # never send zero spacecraft
            Y[int(np.argmax(raw)), 6] = 1
        return Y.ravel()

    def random(self, rng: np.random.Generator | None = None, repair: bool = True) -> np.ndarray:
        rng = rng or np.random.default_rng()
        lo, hi = self.bounds()
        X = rng.uniform(lo, hi)
        return self.repair(X) if repair else X

    @staticmethod
    def decode(X) -> OrbitalSolution:
        return OrbitalSolution.from_vector(X)


def _reflect(x, lo, hi):
    """Reflect values back into [lo, hi] (keeps a random walk unbiased near the bounds)."""
    x = np.asarray(x, float)
    span = np.asarray(hi, float) - np.asarray(lo, float)
    y = np.mod(x - lo, 2 * span)
    return lo + np.where(y > span, 2 * span - y, y)

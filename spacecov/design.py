"""Decision variables: what an optimisation algorithm is allowed to change.

    x = [ T,  a_1, inc_1, raan_1, M0_1, t_1,  ...,  a_Nmax, inc_Nmax, raan_Nmax, M0_Nmax, t_Nmax ]

    T        mission duration [days], in [T_min, T_max]
    a_i      orbit radius [km]                      (circular orbits: a is the radius)
    inc_i    inclination [deg], 0 .. 180            (tilt of the orbit to the asteroid's equator)
    raan_i   longitude of the ascending node [deg]  (where the orbit crosses the equator)
    M0_i     mean anomaly at t = 0 [deg]            (where the spacecraft starts)
    t_i      on/off switch: 1 = slot i flies a spacecraft, 0 = slot unused (integer)

The number of spacecraft is N = t_1 + ... + t_Nmax: switching slots on and off is how an
algorithm decides how many spacecraft to send.  The vector has a fixed length 1 + 5 N_max so
every algorithm (SA, GA, PSO, ...) can treat it as one point in a box; the orbit numbers of a
switched-off slot are carried along unused.

Every orbit is circular (e = 0, so the argument of periapsis has no meaning and is 0) and every
spacecraft carries the same camera (MissionConfig.camera).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MissionConfig
from .physics import OrbitalSolution, OrbitalSpacecraft

HEAD_VARS = ("T",)
SLOT_VARS = ("a", "inc", "raan", "M0", "on")
N_HEAD, N_SLOT = len(HEAD_VARS), len(SLOT_VARS)
ON = SLOT_VARS.index("on")
UNITS = {"T": " d", "a": " km", "inc": "°", "raan": "°", "M0": "°", "on": ""}


@dataclass
class CircularOrbit:
    a: float            # radius [km]
    inc: float = 0.0    # inclination [deg]
    raan: float = 0.0   # longitude of the ascending node [deg]
    M0: float = 0.0     # mean anomaly (= argument of latitude) at t = 0 [deg]

    def key(self) -> tuple:
        return (float(self.a), float(self.inc), float(self.raan), float(self.M0))

    def spacecraft(self, camera: int) -> OrbitalSpacecraft:
        """The general orbit the physics module propagates (e = 0, argp = 0)."""
        return OrbitalSpacecraft(float(self.a), 0.0, float(self.inc), float(self.raan), 0.0, float(self.M0),
                                 type=int(camera))

    def to_dict(self) -> dict:
        return {"a": self.a, "inc": self.inc, "raan": self.raan, "M0": self.M0}


@dataclass
class Design:
    """A decoded decision vector: the mission the engine simulates (flying spacecraft only)."""
    T_days: float
    orbits: list[CircularOrbit] = field(default_factory=list)     # the N flying spacecraft

    @property
    def N(self) -> int:
        return len(self.orbits)

    @property
    def T(self) -> float:
        """Mission duration [s]."""
        return self.T_days * 86400.0

    def fleet(self, camera: int) -> OrbitalSolution:
        return OrbitalSolution([o.spacecraft(camera) for o in self.orbits])

    def to_dict(self) -> dict:
        return {"N": self.N, "T_days": self.T_days, "spacecraft": [o.to_dict() for o in self.orbits]}

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        orbits = [CircularOrbit(**s) for s in d.get("spacecraft", [])]
        if "N" in d and int(d["N"]) != len(orbits):
            raise ValueError(f"design says N = {d['N']} but lists {len(orbits)} spacecraft")
        return cls(float(d["T_days"]), orbits)

    def summary(self) -> str:
        return f"N = {self.N}, T = {self.T_days:.2f} d"


class DesignSpace:
    """Layout, bounds and variable kinds of the decision vector for a given N_max."""

    def __init__(self, n_max: int, r_min: float, r_max: float, t_min_days: float, t_max_days: float,
                 fixed_n: bool = False):
        if n_max < 1:
            raise ValueError("n_max must be >= 1")
        self.n_max = int(n_max)
        self.fixed_n = bool(fixed_n)          # every switch held at 1: N = n_max
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.t_min_days, self.t_max_days = float(t_min_days), float(t_max_days)

    @classmethod
    def from_config(cls, cfg: MissionConfig, asteroid_r_max: float) -> "DesignSpace":
        return cls(cfg.n_max, cfg.resolve_r_min(asteroid_r_max), cfg.r_max, cfg.t_min_days, cfg.t_max_days,
                   cfg.fixed_n)

    # -- layout --------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return N_HEAD + N_SLOT * self.n_max

    @property
    def names(self) -> list[str]:
        return list(HEAD_VARS) + [f"{v}_{i + 1}" for i in range(self.n_max) for v in SLOT_VARS]

    @staticmethod
    def slot_index(i: int, var: str) -> int:
        return N_HEAD + N_SLOT * i + SLOT_VARS.index(var)

    def slots(self, x) -> np.ndarray:
        """(n_max, 5) view of the slot part of x: columns a, inc, raan, M0, on."""
        return np.asarray(x, float)[N_HEAD:].reshape(self.n_max, N_SLOT)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        on_lo = 1.0 if self.fixed_n else 0.0
        lo = np.concatenate([[self.t_min_days], np.tile([self.r_min, 0.0, 0.0, 0.0, on_lo], self.n_max)])
        hi = np.concatenate([[self.t_max_days], np.tile([self.r_max, 180.0, 360.0, 360.0, 1.0], self.n_max)])
        return lo, hi

    @property
    def integer_mask(self) -> np.ndarray:
        """The on/off switches."""
        m = np.zeros(self.dim, bool)
        for i in range(self.n_max):
            m[self.slot_index(i, "on")] = True
        return m

    @property
    def periodic_mask(self) -> np.ndarray:
        """Variables that wrap around (RAAN and M0: 360 deg = 0 deg)."""
        m = np.zeros(self.dim, bool)
        for i in range(self.n_max):
            m[self.slot_index(i, "raan")] = m[self.slot_index(i, "M0")] = True
        return m

    def slot_groups(self) -> list[np.ndarray]:
        """Indices of the five genes (a, inc, raan, M0, on) of each slot: one group per spacecraft."""
        return [np.arange(N_HEAD + N_SLOT * i, N_HEAD + N_SLOT * (i + 1)) for i in range(self.n_max)]

    def on_slots(self, x) -> np.ndarray:
        """Indices of the slots that fly (switch rounds to 1)."""
        return np.flatnonzero(np.rint(self.slots(x)[:, ON]) >= 1)

    def n_of(self, x) -> int:
        """Number of flying spacecraft N encoded in x."""
        return len(self.on_slots(x))

    def active_mask(self, x) -> np.ndarray:
        """Coordinates that currently matter: T, every switch, and the orbits of flying spacecraft."""
        m = np.zeros((self.n_max, N_SLOT), bool)
        m[self.on_slots(x), :] = True
        m[:, ON] = not self.fixed_n                  # fixed fleet size: the switches cannot move
        return np.concatenate([[True], m.ravel()])

    # -- vector <-> design ---------------------------------------------------------------
    def decode(self, x) -> Design:
        """Read a vector as a design (no clipping of the values: feasibility judges them)."""
        x = np.asarray(x, float)
        if x.shape != (self.dim,):
            raise ValueError(f"decision vector must have length {self.dim}, got {x.shape}")
        slots = self.slots(x)
        return Design(float(x[0]), [CircularOrbit(*map(float, slots[i, :ON])) for i in self.on_slots(x)])

    def encode(self, design: Design, fill: np.ndarray | None = None) -> np.ndarray:
        """Vector of a design: its spacecraft in the first slots (switched on), the rest switched
        off with their orbit numbers copied from ``fill`` (or set mid-range)."""
        if design.N > self.n_max:
            raise ValueError(f"design has {design.N} spacecraft but N_max = {self.n_max}")
        lo, hi = self.bounds()
        x = np.asarray(fill, float).copy() if fill is not None else 0.5 * (lo + hi)
        x[0] = design.T_days
        for i in range(self.n_max):
            x[self.slot_index(i, "on")] = 0.0
        for i, o in enumerate(design.orbits):
            j = self.slot_index(i, "a")
            x[j:j + N_SLOT] = [o.a, o.inc, o.raan, o.M0, 1.0]
        return x

    def random(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Uniform random vector.  Every switch is a coin flip, or, with ``n``, exactly n random
        slots are switched on (stratified start of the GA).  A fixed fleet size ignores n."""
        lo, hi = self.bounds()
        x = rng.uniform(lo, hi)
        if n is not None and not self.fixed_n:
            on = rng.choice(self.n_max, size=int(np.clip(n, 1, self.n_max)), replace=False)
            for i in range(self.n_max):
                x[self.slot_index(i, "on")] = 1.0 if i in on else 0.0
            return x
        for i in range(self.n_max):
            x[self.slot_index(i, "on")] = 1.0 if self.fixed_n else float(rng.integers(0, 2))
        return x

    def describe(self, x) -> str:
        d = self.decode(x)
        parts = [f"N={d.N}", f"T={d.T_days:.2f}d"]
        parts += [f"[{o.a:.1f} km, {o.inc:.0f}°, {o.raan:.0f}°, {o.M0:.0f}°]" for o in d.orbits]
        return " ".join(parts)

    def describe_change(self, x_prev, y) -> tuple[int, str]:
        """(index of the changed spacecraft in the drawn fleet, or -1; readable description) of x_prev -> y."""
        a, b = np.asarray(x_prev, float), np.asarray(y, float)
        diff = np.flatnonzero(np.abs(a - b) > 1e-9)
        if len(diff) == 0:
            return -1, "no change"
        parts, shown = [], -1
        for j in diff:
            if j == 0:
                parts.append(f"T {a[0]:.2f} -> {b[0]:.2f} d")
                continue
            i, var = divmod(j - N_HEAD, N_SLOT)
            name = SLOT_VARS[var]
            if name == "on":
                on = np.rint(b[j]) >= 1
                parts.append(f"slot {i + 1} switched {'on (adds' if on else 'off (removes'} a spacecraft)")
                flying = self.on_slots(b if on else a)
            else:
                dec = 1 if name == "a" else 0
                parts.append(f"slot {i + 1} {name} {a[j]:.{dec}f} -> {b[j]:.{dec}f}{UNITS[name]}")
                flying = self.on_slots(b)
            if shown < 0 and i in flying:
                shown = int(np.flatnonzero(flying == i)[0])
        return shown, ", ".join(parts)

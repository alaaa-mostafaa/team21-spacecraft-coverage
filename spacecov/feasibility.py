"""Feasibility: does a decision vector describe a mission that is allowed to fly?

Sits between the algorithm and the engine.  ``check`` runs every test that does not
need the (expensive) imaging simulation; a design that fails it never reaches the engine.
``check_result`` adds the one constraint that needs the simulation (the battery).

Constraints, each normalised so that a value <= 0 is satisfied and > 0 is the relative
size of the violation (Milestone 2 Ch. 4.6 / Milestone 3, for circular orbits):

    N          at least one slot switched on; every switch is 0 or 1
    T          T_min <= T <= T_max
    g1 r_min   a_i >= r_min = kappa R_max   for every flying spacecraft  (periapsis safety;
               circular orbit: periapsis = apoapsis = a)
    g2 r_max   a_i <= r_max                                              (apoapsis limit)
    inc        0 <= inc_i <= 180 deg
    g3 sep     |r_i(t_k) - r_j(t_k)| >= d_sep for every pair and every step t_k <= T
    g4 battery state of charge >= soc_min at every step t_k <= T          (after the engine)

RAAN and M0 are angles: every real value is a valid orbit (they wrap), so they are
not constraints.  ``repair`` maps any vector back into the box (reflects a, inc and T at
their bounds, wraps the angles, rounds the switches and keeps at least one on); algorithms
may call it on the candidates they generate.  It cannot repair separation or battery violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MissionConfig
from .design import N_HEAD, ON, DesignSpace
from .physics import min_separation, propagate

PRE_CHECKS = ("N", "T", "g1 r_min", "g2 r_max", "inc", "g3 separation")
POST_CHECKS = ("g4 battery",)
LABELS = {"N": "at least one spacecraft", "T": "T in [T_min, T_max]", "g1 r_min": "g1 radius ≥ r_min",
          "g2 r_max": "g2 radius ≤ r_max", "inc": "inclination 0-180°", "g3 separation": "g3 separation",
          "g4 battery": "g4 battery"}


@dataclass
class FeasibilityReport:
    violations: dict[str, float] = field(default_factory=dict)   # name -> normalised value (> 0 violated)
    min_separation: float = float("inf")                          # km, over the mission
    min_soc: float | None = None                                  # lowest battery charge (after the engine)

    @property
    def feasible(self) -> bool:
        return all(v <= 0.0 for v in self.violations.values())

    @property
    def violated(self) -> dict[str, float]:
        return {k: v for k, v in self.violations.items() if v > 0.0}

    @property
    def penalty(self) -> float:
        """Sum of squared violations (0 when feasible)."""
        return float(sum(max(0.0, v) ** 2 for v in self.violations.values()))

    def summary(self) -> str:
        if self.feasible:
            return "feasible"
        return "INFEASIBLE: " + ", ".join(f"{k} {v:+.3f}" for k, v in self.violated.items())


class Feasibility:
    def __init__(self, space: DesignSpace, cfg: MissionConfig):
        self.space, self.cfg = space, cfg

    # ------------------------------------------------------------------ before the engine
    def check(self, x) -> FeasibilityReport:
        x = np.asarray(x, float)
        sp, cfg = self.space, self.cfg
        rep = FeasibilityReport()
        v = rep.violations

        switches = sp.slots(x)[:, ON]
        not_binary = np.any(np.abs(switches - np.rint(switches)) > 1e-9) or np.any((switches < 0) | (switches > 1))
        n = sp.n_of(x)
        v["N"] = 1.0 if not_binary else (1.0 - n if n < 1 else -1.0)
        if sp.fixed_n and n != sp.n_max:                    # forced fleet size: every slot flies
            v["N"] = max(v["N"], (sp.n_max - n) / sp.n_max)

        T = float(x[0])
        v["T"] = max(sp.t_min_days - T, T - sp.t_max_days) / sp.t_max_days

        flying = sp.slots(x)[sp.on_slots(x)]
        a, inc = flying[:, 0], flying[:, 1]
        v["g1 r_min"] = float(np.max((sp.r_min - a) / sp.r_min)) if n else -1.0
        v["g2 r_max"] = float(np.max((a - sp.r_max) / sp.r_max)) if n else -1.0
        v["inc"] = float(np.max(np.maximum(-inc, inc - 180.0) / 180.0)) if n else -1.0

        # separation needs positions, so only for designs whose orbits are otherwise valid
        if n >= 2 and v["g1 r_min"] <= 0 and v["g2 r_max"] <= 0 and v["T"] <= 0:
            rep.min_separation = self.min_separation(x)
            v["g3 separation"] = (cfg.d_sep - rep.min_separation) / cfg.d_sep
        else:
            v["g3 separation"] = -1.0 if n < 2 else 0.0
        return rep

    def min_separation(self, x) -> float:
        """Closest approach [km] of any two flying spacecraft over the steps t_k <= T."""
        d = self.space.decode(x)
        times = self.cfg.times[:self.cfg.steps_within(d.T)]
        return min_separation(propagate(d.fleet(self.cfg.camera), times, self.cfg.mu))

    # ------------------------------------------------------------------ after the engine
    def check_result(self, rep: FeasibilityReport, result) -> FeasibilityReport:
        """Add the constraints that need the simulation (battery)."""
        rep.min_soc = float(result.soc.min()) if result.soc.size else 1.0
        rep.violations["g4 battery"] = (self.cfg.soc_min - rep.min_soc) / self.cfg.soc_min
        return rep

    # ------------------------------------------------------------------ repair
    def repair(self, x) -> np.ndarray:
        """Map any vector into the box of DesignSpace.bounds(), with at least one slot switched on."""
        sp = self.space
        y = np.asarray(x, float).copy()
        y[0] = _reflect(y[0], sp.t_min_days, sp.t_max_days)
        slots = y[N_HEAD:].reshape(sp.n_max, -1)
        raw = slots[:, ON].copy()
        slots[:, 0] = _reflect(slots[:, 0], sp.r_min, sp.r_max)
        slots[:, 1] = _reflect(slots[:, 1], 0.0, 180.0)
        slots[:, 2:4] %= 360.0
        slots[:, ON] = 1.0 if sp.fixed_n else np.clip(np.rint(slots[:, ON]), 0, 1)
        if not np.any(slots[:, ON] > 0):                        # never send zero spacecraft
            slots[int(np.argmax(raw)), ON] = 1.0
        y[N_HEAD:] = slots.ravel()
        return y


def _reflect(x, lo, hi):
    """Reflect values back into [lo, hi] (keeps a random walk unbiased near the bounds)."""
    x = np.asarray(x, float)
    span = np.asarray(hi, float) - np.asarray(lo, float)
    if np.all(span <= 0):
        return np.full_like(x, lo)
    y = np.mod(x - lo, 2 * span)
    return lo + np.where(y > span, 2 * span - y, y)

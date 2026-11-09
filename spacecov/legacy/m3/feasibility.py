"""Constraints g1-g4, each normalised so that g <= 0 is satisfied and g > 0 is the relative violation.

    g1 periapsis   a (1 - e) >= r_min = 1.5 R_max      (never violated: ``DesignSpace.repair`` clips e)
    g2 apoapsis    a (1 + e) <= r_max                  (never violated: ``DesignSpace.repair`` clips e)
    g3 separation  every pair of spacecraft >= d_sep apart at every step
    g4 battery     state of charge >= soc_min at every step

g1-g3 only need the orbits; g4 needs the imaging schedule of the simulation.  Infeasible designs are
still simulated: they enter the fitness with a quadratic penalty (``Engine``), so the search can rank
them by coverage as well and cross briefly through infeasible regions; the best design reported is
always a feasible one when one was found.
"""
from __future__ import annotations

import numpy as np

from .config import OrbitalConfig
from .physics import min_separation

FEASIBLE_TOL = 1e-9     # repair puts periapsis exactly on r_min, which can give g1 ~ 1e-16 from round-off


def constraints(a: np.ndarray, e: np.ndarray, r_inertial: np.ndarray, soc: np.ndarray, r_min: float,
                cfg: OrbitalConfig) -> tuple[np.ndarray, float]:
    """(g = [g1, g2, g3, g4], closest approach [km]) of the flying spacecraft."""
    g1 = float(np.max((r_min - a * (1 - e)) / r_min))                     # periapsis
    g2 = float(np.max((a * (1 + e) - cfg.r_max) / cfg.r_max))            # apoapsis
    min_sep = min_separation(r_inertial)
    g3 = (cfg.d_sep - min_sep) / cfg.d_sep if np.isfinite(min_sep) else -1.0   # separation
    g4 = float((cfg.soc_min - soc.min()) / cfg.soc_min)                   # battery
    return np.array([g1, g2, g3, g4]), min_sep


def penalty(g: np.ndarray, cfg: OrbitalConfig) -> float:
    """rho * sum max(0, g_j)^2: zero for a feasible design, grows quadratically with the violation."""
    return cfg.rho * float(np.sum(np.maximum(0.0, g) ** 2))


def is_feasible(g: np.ndarray) -> bool:
    return bool(np.all(g <= FEASIBLE_TOL))

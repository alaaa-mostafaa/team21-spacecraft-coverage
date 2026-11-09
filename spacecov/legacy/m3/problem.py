"""The fleet design problem as an optimisation algorithm sees it.

    Algorithm --x--> DesignSpace.repair --> Engine (simulation, objectives, constraints) --J--> Algorithm

``OrbitalProblem`` is the only object an algorithm needs.  Its interface (shared with the benchmark
problems in ``spacecov.algorithms.benchmarks``) is:

    dim, names, bounds(), integer_mask, periodic_mask, active_mask(x)    the search space
    repair(x), random_solution(rng)                                       valid designs
    objective(x) -> (J, info)                                             the fitness to minimise
    evaluate(x) -> OrbitalResult                                          everything about a design
    move(p_type), default_chain_length                                    the local move used by SA
    multiobjective(x) -> ([-C, Z, E], g)                                  for NSGA-II later
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .asteroid import Asteroid
from .config import OrbitalConfig
from .design import N_VARS, DesignSpace, OrbitalSolution
from .engine import Engine, OrbitalResult


class OrbitalProblem:
    """Search space, bounds, repair and fitness for any optimiser (SA here; GA/PSO/NSGA-II later).

    ``n_max`` slots of 7 variables; a slot whose type rounds to 0 is unused.  With
    ``fixed_n=True`` every slot must carry a camera (fleet size fixed to n_max) - used to measure
    coverage as a function of the number of spacecraft."""

    def __init__(self, asteroid: Asteroid, n_max: int, cfg: OrbitalConfig | None = None, fixed_n: bool = False):
        self.asteroid = asteroid
        self.n = int(n_max)
        self.fixed_n = fixed_n
        self.cfg = replace(cfg or OrbitalConfig(), n_max=self.n)
        self.r_min = self.cfg.resolve_r_min(asteroid.r_max)
        self.space = DesignSpace(self.n, self.r_min, self.cfg, fixed_n)
        self.engine = Engine(asteroid, self.cfg)

    # ------------------------------------------------------------------ search space (DesignSpace)
    @property
    def dim(self) -> int:
        return self.space.dim

    @property
    def names(self) -> list[str]:
        return self.space.names

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.space.bounds()

    @property
    def integer_mask(self) -> np.ndarray:
        return self.space.integer_mask

    @property
    def periodic_mask(self) -> np.ndarray:
        return self.space.periodic_mask

    def active_mask(self, x) -> np.ndarray:
        return self.space.active_mask(x)

    def e_limit(self, a: np.ndarray) -> np.ndarray:
        return self.space.e_limit(a)

    def repair(self, X: np.ndarray) -> np.ndarray:
        return self.space.repair(X)

    def random_solution(self, rng: np.random.Generator | None = None, repair: bool = True) -> np.ndarray:
        return self.space.random(rng, repair)

    def decode(self, X) -> OrbitalSolution:
        return self.space.decode(X)

    # ------------------------------------------------------------------ evaluation (Engine)
    def evaluate(self, X) -> OrbitalResult:
        return self.engine.evaluate(self.decode(X))

    def fitness(self, X) -> float:
        """Scalar J to minimise (GA, PSO, SA)."""
        return self.evaluate(X).J

    def multiobjective(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(objectives [-C, Z, E], constraints g) for NSGA-II."""
        r = self.evaluate(X)
        return r.objectives, r.g

    def objective(self, X) -> tuple[float, dict]:
        """(J, info) for the algorithms: info holds the objectives and ``feasible`` (1 / 0)."""
        r = self.evaluate(X)
        return r.J, {"C": r.C, "Z": r.Z, "E": r.E, "S": r.S, "N": float(r.n_active), "feasible": float(r.feasible)}

    # ------------------------------------------------------------------ local move (SA)
    @property
    def default_chain_length(self) -> int:
        return 10 * self.n

    def move(self, p_type: float = 0.15) -> "SlotMove":
        """The neighbourhood move that changes ONE slot (used by SA)."""
        return SlotMove(self, p_type)

    @staticmethod
    def describe_move(x_prev, y) -> tuple[int, str]:
        return describe_move(x_prev, y)


class SlotMove:
    """Neighbourhood move: change ONE slot of the fleet.

    * with probability p_type: switch a random slot (type in {0, ..., max_type}, 0 = unused), i.e. add or
      remove a spacecraft.  The last flying spacecraft is never removed.
    * otherwise: perturb 1-2 orbital elements of a random FLYING spacecraft (Gaussian,
      sigma = step * range); unused slots are never moved, so no evaluation is wasted.
    The result is repaired (angles wrap, a and inc reflect, e is clipped so g1 and g2 always hold)."""

    def __init__(self, problem: OrbitalProblem, p_type: float = 0.15):
        self.problem = problem
        self.p_type = p_type
        lo, hi = problem.bounds()
        self.span = (hi - lo)[:N_VARS - 1]                        # ranges of a, e, inc, RAAN, argp, M0

    def __call__(self, x: np.ndarray, step: float, rng: np.random.Generator) -> np.ndarray:
        y = np.asarray(x, float).reshape(self.problem.n, N_VARS).copy()
        types = np.rint(y[:, 6]).astype(int)
        active = np.flatnonzero(types > 0)
        options = []
        if rng.random() < self.p_type or len(active) == 0:
            i = rng.integers(self.problem.n)
            options = [t for t in range(self.problem.cfg.max_type + 1)
                       if t != types[i] and not (t == 0 and len(active) == 1 and types[i] > 0)]
            if getattr(self.problem, "fixed_n", False):
                options = [t for t in options if t != 0]          # fleet size fixed: only swap cameras
        if options:
            y[i, 6] = rng.choice(options)
        else:                                                     # (one camera type and N fixed: no type move)
            i = rng.choice(active)
            cols = rng.choice(N_VARS - 1, size=rng.integers(1, 3), replace=False)
            y[i, cols] += rng.normal(0.0, step * self.span[cols])
        return self.problem.repair(y.ravel())


MOVE_UNITS = {"a": ("a", " km", 1), "e": ("e", "", 3), "inc": ("inc", "°", 0), "raan": ("RAAN", "°", 0),
              "argp": ("argp", "°", 0), "M0": ("M0", "°", 0)}
TYPE_NAMES = {0: "unused", 1: "narrow camera", 2: "wide camera"}


def describe_move(x_prev: np.ndarray, y: np.ndarray) -> tuple[int, str]:
    """(slot index, human-readable description) of the change between two fleet vectors."""
    a = np.asarray(x_prev, float).reshape(-1, N_VARS)
    b = np.asarray(y, float).reshape(-1, N_VARS)
    diff = np.flatnonzero(np.any(np.abs(a - b) > 1e-9, axis=1))
    if len(diff) == 0:
        return -1, "no change (the step was clipped back at a bound)"
    i = int(diff[0])
    ta, tb = int(np.rint(a[i, 6])), int(np.rint(b[i, 6]))
    if ta != tb:
        what = {0: "removes a spacecraft", }.get(tb, "adds a spacecraft" if ta == 0 else "swaps the camera")
        return i, f"slot {i + 1}: {TYPE_NAMES[ta]} -> {TYPE_NAMES[tb]} ({what})"
    parts = []
    for j, name in enumerate(("a", "e", "inc", "raan", "argp", "M0")):
        if abs(a[i, j] - b[i, j]) > 1e-9:
            label, unit, dec = MOVE_UNITS[name]
            parts.append(f"{label} {a[i, j]:.{dec}f} -> {b[i, j]:.{dec}f}{unit}")
    return i, f"slot {i + 1}: " + ", ".join(parts)

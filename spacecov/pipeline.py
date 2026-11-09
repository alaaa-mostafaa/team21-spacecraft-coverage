"""Pipeline: the optimisation problem as an algorithm sees it.

    Algorithm --x--> Feasibility --design--> Engine --J--> cost --> Algorithm

``Pipeline.evaluate(x)`` is the only call an algorithm needs:

  1. Feasibility.check(x)          bounds, orbit radius limits, separation (cheap)
       fails -> cost = infeasible_cost + rho * sum(violation^2), engine NOT run
  2. Engine.simulate(design)       imaging simulation over [0, T], cost terms
  3. Feasibility.check_result      battery (needs the simulation)
       fails -> cost = infeasible_cost + rho * sum(violation^2)
  4. cost = J (the fitness)

infeasible_cost is larger than any feasible J (<= 1), so a feasible design always beats an
infeasible one; the violation term still ranks infeasible designs among themselves
(useful for population methods such as GA / PSO).

Besides ``evaluate`` the pipeline exposes what generic algorithms need to move in the
search space: ``dim``, ``bounds()``, ``integer_mask``, ``periodic_mask``,
``active_mask(x)``, ``repair(x)`` and ``random_solution(rng)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asteroid import Asteroid
from .config import MissionConfig
from .design import Design, DesignSpace
from .engine import Engine, EngineResult
from .feasibility import Feasibility, FeasibilityReport


@dataclass
class Evaluation:
    x: np.ndarray
    design: Design
    report: FeasibilityReport
    result: EngineResult | None          # None when feasibility rejected the design before the engine
    cost: float

    @property
    def feasible(self) -> bool:
        return self.report.feasible

    def info(self) -> dict[str, float]:
        """Flat numbers for histories, logs and live charts."""
        d = {"cost": self.cost, "feasible": float(self.feasible), "N": float(self.design.N),
             "T_days": self.design.T_days, "violation": self.report.penalty}
        r = self.result
        if r is not None:
            d.update({"S_a": r.S_a, **r.cost.as_dict()})
        else:
            d.update({k: np.nan for k in ("S_a", "C", "Z", "E", "E_dv", "E_elec", "f_time", "J")})
        return d

    def summary(self) -> str:
        head = f"cost = {self.cost:.4f}   {self.report.summary()}"
        return head + ("\n" + self.result.summary() if self.result is not None else
                       f"\n{self.design.summary()} (not simulated)")


class Pipeline:
    def __init__(self, asteroid: Asteroid, cfg: MissionConfig | None = None):
        self.asteroid = asteroid
        self.cfg = cfg or MissionConfig()
        self.space = DesignSpace.from_config(self.cfg, asteroid.r_max)
        self.feasibility = Feasibility(self.space, self.cfg)
        self.engine = Engine(asteroid, self.cfg)
        self.n_evaluations = 0          # calls to evaluate()
        self.n_rejected = 0             # stopped by feasibility before the engine

    # ------------------------------------------------------------------ the loop
    def evaluate(self, x, keep_visible: bool = False, progress=None) -> Evaluation:
        x = np.asarray(x, float)
        self.n_evaluations += 1
        design = self.space.decode(x)
        report = self.feasibility.check(x)
        if not report.feasible:
            self.n_rejected += 1
            return Evaluation(x, design, report, None, self.penalised(report))
        result = self.engine.simulate(design, keep_visible, progress)
        report = self.feasibility.check_result(report, result)
        cost = result.J if report.feasible else self.penalised(report)
        return Evaluation(x, design, report, result, float(cost))

    def __call__(self, x) -> float:
        return self.evaluate(x).cost

    def penalised(self, report: FeasibilityReport) -> float:
        return float(self.cfg.infeasible_cost + self.cfg.rho * report.penalty)

    # ------------------------------------------------------------------ search space
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

    def repair(self, x) -> np.ndarray:
        return self.feasibility.repair(x)

    def slot_groups(self) -> list[np.ndarray]:
        """Gene indices of each spacecraft slot (used by the GA's slot crossover)."""
        return self.space.slot_groups()

    def random_solution(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Random design; with ``n``, exactly n spacecraft fly."""
        return self.repair(self.space.random(rng, n))

    def random_feasible(self, rng: np.random.Generator, tries: int = 200, n: int | None = None) -> np.ndarray:
        """A random design that passes the cheap feasibility checks (separation can fail);
        with ``n``, exactly n spacecraft fly."""
        for _ in range(tries):
            x = self.random_solution(rng, n)
            if self.feasibility.check(x).feasible:
                return x
        raise RuntimeError("no feasible random design found - are the constraints satisfiable?")

    def describe(self, x) -> str:
        return self.space.describe(x)

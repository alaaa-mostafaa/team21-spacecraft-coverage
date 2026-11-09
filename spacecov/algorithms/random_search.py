"""Random search: the baseline every optimiser must beat with the same evaluation budget.

Draws designs uniformly in the box (repaired, like any candidate), evaluates each through the
problem, and keeps the best (a feasible design beats an infeasible one, then lower cost).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Algorithm, OptimResult, Problem, Recorder, Step


@dataclass
class RandomSearchParams:
    max_evals: int = 1500             # evaluation budget
    seed: int = 0


class RandomSearch(Algorithm):
    name = "random"
    Params = RandomSearchParams

    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[Step], bool | None] | None = None) -> OptimResult:
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.params.seed)
        rec = Recorder(problem, self.params.max_evals, callback)
        while not rec.exhausted:
            x = problem.random_solution(rng)
            before = rec.best
            ev = rec.evaluate(x)
            rec.report(x, ev, rec.best is ev and ev is not before)
        return self._result(rec, t_start, "budget used")


def random_search(problem: Problem, max_evals: int, seed: int = 0,
                  callback: Callable[[Step], bool | None] | None = None) -> OptimResult:
    """Shortcut: RandomSearch(max_evals=..., seed=...).run(problem)."""
    return RandomSearch(max_evals=max_evals, seed=seed).run(problem, callback=callback)

"""What every optimisation algorithm can rely on, and what it must provide.

An algorithm never sees orbits or the asteroid.  It only talks to a ``Problem``:

    dim                          number of decision variables
    bounds()                     (lower, upper) arrays of the box
    integer_mask, periodic_mask  which variables are whole numbers / angles that wrap
    active_mask(x)               which variables currently matter (e.g. orbits of flying spacecraft)
    repair(x)                    map any vector onto a valid one
    random_solution(rng)         a random valid vector
    objective(x) -> (J, info)    fitness to MINIMISE; info["feasible"] is 1.0 / 0.0
    move(p_type), default_chain_length      optional: a problem-specific local move (used by SA)

``spacecov.problem.OrbitalProblem`` (the fleet) and ``spacecov.algorithms.benchmarks.BenchmarkProblem``
(test functions with a known optimum) both implement it, so the same algorithm code is validated on
the benchmarks and then run on the fleet.

To add an algorithm (GA, PSO, ...): subclass ``Algorithm`` with a ``Params`` dataclass and a
``run(problem, x0=None, callback=None)``, then register it in ``spacecov/algorithms/__init__.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class Problem(Protocol):
    dim: int

    def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    def repair(self, x: np.ndarray) -> np.ndarray: ...

    def random_solution(self, rng: np.random.Generator) -> np.ndarray: ...

    def objective(self, x: np.ndarray) -> tuple[float, dict]: ...


def better(J_a: float, feasible_a: float, J_b: float, feasible_b: float) -> bool:
    """Is design a better than design b?  A feasible design always beats an infeasible one, then lower J."""
    return feasible_a > feasible_b or (feasible_a == feasible_b and J_a < J_b)


def gaussian_move(problem: Problem) -> Callable[[np.ndarray, float, np.random.Generator], np.ndarray]:
    """Default local move for problems without their own: perturb one variable by N(0, (step * range)^2)."""
    lo, hi = problem.bounds()
    span = hi - lo

    def move(x, step, rng):
        y = np.asarray(x, float).copy()
        j = rng.integers(len(y))
        y[j] += rng.normal(0.0, step * span[j])
        return problem.repair(y)
    return move


class Algorithm:
    """Base class: ``name`` (registry key), ``Params`` (a dataclass of settings) and ``run``."""
    name = "base"
    Params: type = object

    def __init__(self, params: Any = None):
        self.params = params if params is not None else self.Params()

    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[Any], bool | None] | None = None) -> Any:
        raise NotImplementedError

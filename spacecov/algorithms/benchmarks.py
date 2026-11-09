"""Test functions with a known optimum, wrapped as ``Problem`` objects.

Because algorithms only see the Problem interface, running them here checks the
algorithm itself (does it find the known minimum?) independently of the spacecraft model.
All functions have their global minimum 0 at x = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
                 + 20 + np.e)


FUNCTIONS = {   # name: (function, lower bound, upper bound)
    "sphere": (sphere, -5.12, 5.12),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
}


@dataclass
class BenchmarkEvaluation:
    x: np.ndarray
    cost: float
    feasible: bool = True

    def info(self) -> dict:
        return {"cost": self.cost, "feasible": 1.0}


class BenchmarkProblem:
    """Box-bounded function minimisation with the same interface as spacecov.pipeline.Pipeline."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.f, lo, hi = FUNCTIONS[name]
        self.dim = int(dim)
        self._lo, self._hi = np.full(self.dim, lo), np.full(self.dim, hi)
        self.integer_mask = np.zeros(self.dim, bool)
        self.periodic_mask = np.zeros(self.dim, bool)

    def bounds(self):
        return self._lo.copy(), self._hi.copy()

    def evaluate(self, x) -> BenchmarkEvaluation:
        x = np.asarray(x, float)
        return BenchmarkEvaluation(x, self.f(x))

    def repair(self, x):
        return np.clip(np.asarray(x, float), self._lo, self._hi)

    def random_solution(self, rng):
        return rng.uniform(self._lo, self._hi)

    def random_feasible(self, rng):
        return self.random_solution(rng)

    def active_mask(self, x):
        return np.ones(self.dim, bool)

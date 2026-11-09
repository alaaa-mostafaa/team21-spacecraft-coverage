"""Test functions with a known optimum (0 at x = 0), wrapped as problems with the same interface as the fleet.

Running an algorithm here checks the algorithm itself, independently of the spacecraft model.
"""
from __future__ import annotations

import numpy as np

from .sa import SAParams, SAResult, SimulatedAnnealing


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
                 + 20 + np.e)


BENCHMARKS = {   # name: (function, lower bound, upper bound); global minimum 0 at x = 0
    "sphere": (sphere, -5.12, 5.12),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
}


class BenchmarkProblem:
    """Box-bounded minimisation of a test function."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.f, self.lo, self.hi = BENCHMARKS[name]
        self.dim = int(dim)
        self.integer_mask = np.zeros(self.dim, bool)
        self.periodic_mask = np.zeros(self.dim, bool)

    def bounds(self):
        return np.full(self.dim, self.lo), np.full(self.dim, self.hi)

    def active_mask(self, x):
        return np.ones(self.dim, bool)

    def repair(self, x):
        return np.clip(np.asarray(x, float), self.lo, self.hi)

    def random_solution(self, rng: np.random.Generator):
        return rng.uniform(self.lo, self.hi, self.dim)

    def objective(self, x):
        return self.f(x), {}

    @property
    def default_chain_length(self) -> int:
        return 10 * self.dim

    def move(self, p_type: float = 0.0):
        """Change one coordinate by N(0, (step * range)^2), clipped to the box."""
        span = self.hi - self.lo

        def neighbour(x, step, rng):
            y = x.copy()
            j = rng.integers(len(x))
            y[j] = np.clip(y[j] + rng.normal(0.0, step * span), self.lo, self.hi)
            return y
        return neighbour


def solve_benchmark(name: str, dim: int, params: SAParams) -> SAResult:
    return SimulatedAnnealing(params).run(BenchmarkProblem(name, dim))

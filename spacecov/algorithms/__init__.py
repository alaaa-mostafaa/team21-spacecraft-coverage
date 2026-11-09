"""Optimisation algorithms.  Every one implements ``Algorithm.run(problem, x0, callback)``.

    from spacecov.algorithms import make
    algo = make("sa", max_evals=2000, seed=1)
    result = algo.run(pipeline)

Add a new algorithm by subclassing base.Algorithm and registering it in ALGORITHMS.
"""
from __future__ import annotations

from .base import Algorithm, OptimResult, Problem, Step, better


def _registry() -> dict[str, type[Algorithm]]:
    from .ga import GeneticAlgorithm
    from .random_search import RandomSearch
    from .sa import SimulatedAnnealing
    return {"sa": SimulatedAnnealing, "ga": GeneticAlgorithm, "random": RandomSearch}


def available() -> list[str]:
    return list(_registry())


def get(name: str) -> type[Algorithm]:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown algorithm {name!r}; available: {', '.join(reg)}")
    return reg[name]


def make(name: str, **params) -> Algorithm:
    return get(name)(**params)


__all__ = ["Algorithm", "OptimResult", "Problem", "Step", "better", "available", "get", "make"]

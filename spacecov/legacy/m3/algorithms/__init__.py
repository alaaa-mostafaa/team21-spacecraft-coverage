"""Optimisation algorithms.  Each works on any problem with the interface of ``base.Problem``.

    ALGORITHMS["sa"]       Simulated Annealing (Milestone 3)
    ALGORITHMS["random"]   random search, the baseline

Add a new one (GA, PSO, ...) in its own module and register it here.
"""
from .base import Algorithm, Problem, better
from .random_search import RandomSearch, RandomSearchParams, random_search
from .sa import SAParams, SAResult, SAStep, SimulatedAnnealing, anneal, solve

ALGORITHMS: dict[str, type[Algorithm]] = {cls.name: cls for cls in (SimulatedAnnealing, RandomSearch)}


def get_algorithm(name: str, params=None) -> Algorithm:
    try:
        cls = ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}") from None
    return cls(params)


__all__ = ["ALGORITHMS", "Algorithm", "Problem", "RandomSearch", "RandomSearchParams", "SAParams", "SAResult",
           "SAStep", "SimulatedAnnealing", "anneal", "better", "get_algorithm", "random_search", "solve"]

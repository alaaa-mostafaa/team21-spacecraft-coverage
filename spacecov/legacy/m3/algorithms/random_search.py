"""Random search: the baseline every optimiser must beat.  Same budget, random valid designs, keep the best."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Algorithm, Problem


@dataclass
class RandomSearchParams:
    max_evals: int = 1500
    seed: int = 0


class RandomSearch(Algorithm):
    name = "random"
    Params = RandomSearchParams

    def run(self, problem: Problem, x0=None, callback=None) -> dict[str, np.ndarray]:
        """{"J_best", "best_C" (per evaluation), "x_best"}; ``x0`` and ``callback`` are not used."""
        rng = np.random.default_rng(self.params.seed)
        best, best_C, hist, hist_C = np.inf, np.nan, [], []
        x_best = None
        for _ in range(self.params.max_evals):
            x = problem.random_solution(rng)
            J, info = problem.objective(x)
            if J < best:
                best, best_C, x_best = J, info.get("C", np.nan), x
            hist.append(best)
            hist_C.append(best_C)
        return {"J_best": np.array(hist), "best_C": np.array(hist_C), "x_best": x_best}


def random_search(problem: Problem, max_evals: int, seed: int = 0) -> dict[str, np.ndarray]:
    return RandomSearch(RandomSearchParams(max_evals, seed)).run(problem)

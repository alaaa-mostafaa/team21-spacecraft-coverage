"""Simulated Annealing (Kirkpatrick et al. 1983), written from scratch with NumPy only.

``anneal``               the SA loop (Metropolis acceptance + geometric cooling); knows nothing about
                         spacecraft: it gets an objective, a start and a local move.
``SimulatedAnnealing``   the algorithm on any ``Problem`` (the fleet, or a benchmark function).
``solve``                SA on the fleet; returns the SA record and the best design's full result.

Algorithm (minimisation of J)
-----------------------------
    x <- best of 20 random valid designs (warm start),  J(x)
    T0 <- -median(uphill dJ) / ln(p0)        (uphill moves accepted with prob. p0 at start)
    repeat for each temperature level T = T0, a T0, a^2 T0, ...:
        repeat L times (Markov chain length):
            y <- move(x, step(T))             step shrinks as the system cools
            dJ = J(y) - J(x)
            accept y if dJ <= 0, else with probability exp(-dJ / T)
            remember the best design ever seen (feasible designs first)
    until the evaluation budget is used or T < T_min
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .base import Algorithm, Problem, better, gaussian_move


# --------------------------------------------------------------------------
# Parameters and results
# --------------------------------------------------------------------------
@dataclass
class SAParams:
    max_evals: int = 1500             # evaluation budget (including the T0 estimate)
    chain_length: int | None = None   # L, moves per temperature; None -> 10 * N (orbital) / 10 * dim
    alpha: float | None = None        # geometric cooling factor; None -> reach T_min exactly at the budget
    p0: float = 0.5                   # initial acceptance probability of an average uphill move
    t_min_ratio: float = 1e-3         # stop when T < t_min_ratio * T0
    step0: float = 0.25               # initial move size, fraction of each variable's range
    step_min: float = 0.01            # smallest move size
    p_type: float = 0.15              # probability that a move changes the scanner type (orbital)
    n_t0_samples: int = 20            # moves used to estimate T0
    n_init: int = 20                  # warm start: best of n_init random designs (1 = a single random design)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SAStep:
    """What the optimiser reports after every evaluation (for live plots)."""
    iteration: int
    T: float
    J: float                          # current design
    J_best: float
    candidate_J: float
    accepted: bool
    improved: bool                    # a new best was found at this step
    x_best: np.ndarray
    info: dict                        # extra values of the candidate (e.g. coverage)
    info_best: dict
    # the move itself, so the search can be visualised step by step
    x_prev: np.ndarray | None = None  # current design before the move
    candidate: np.ndarray | None = None   # proposed design y
    x: np.ndarray | None = None       # current design after the decision
    dJ: float = 0.0                   # J(y) - J(x_prev)
    p_accept: float = 1.0             # Metropolis probability used for this move
    info_current: dict | None = None  # extra values of the current design after the decision


@dataclass
class SAResult:
    x_best: np.ndarray
    J_best: float
    info_best: dict
    history: dict[str, np.ndarray]    # per evaluation: J, J_best, candidate_J, T, accepted, + info keys
    levels: dict[str, np.ndarray]     # per temperature level: T, acceptance ratio, J_best
    T0: float
    alpha: float
    chain_length: int
    n_evals: int
    seconds: float
    stop_reason: str
    params: SAParams = field(default_factory=SAParams)


Objective = Callable[[np.ndarray], tuple[float, dict]]
Neighbour = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


# --------------------------------------------------------------------------
# Generic SA loop
# --------------------------------------------------------------------------
def estimate_t0(objective: Objective, x0: np.ndarray, J0: float, neighbour: Neighbour, step: float,
                n: int, p0: float, rng: np.random.Generator) -> tuple[float, int]:
    """T0 = -median(dJ+) / ln(p0): a typical uphill move is accepted with probability p0.

    The median (not the mean) is used so that a few moves that hit the large separation
    penalty do not make the start temperature huge."""
    ups = []
    for _ in range(n):
        dJ = objective(neighbour(x0, step, rng))[0] - J0
        if dJ > 0:
            ups.append(dJ)
    if not ups:
        return 1e-3, n
    return float(-np.median(ups) / np.log(p0)), n


def anneal(objective: Objective, x0: np.ndarray, neighbour: Neighbour, params: SAParams,
           chain_length: int, rng: np.random.Generator | None = None,
           callback: Callable[[SAStep], bool | None] | None = None,
           start: list[tuple[float, dict]] | None = None) -> SAResult:
    """Minimise ``objective`` from ``x0``.  ``callback(step)`` may return True to stop early.

    ``start``: the (J, info) of every design evaluated to pick ``x0`` (warm start); they count in the
    budget and open the history, and the last best of them is ``x0``'s value."""
    t_start = time.perf_counter()
    rng = rng or np.random.default_rng(params.seed)
    L = max(1, int(chain_length))

    x = np.asarray(x0, float).copy()
    if start:
        J, info = min(start, key=lambda s: (-s[1].get("feasible", 1.0), s[0]))   # feasible first, then J
    else:
        J, info = objective(x)
    T0, used = estimate_t0(objective, x, J, neighbour, params.step0, params.n_t0_samples, params.p0, rng)
    n_evals = (len(start) if start else 1) + used
    budget = max(params.max_evals - n_evals, 1)
    # number of temperature levels the budget allows, and the cooling that ends at T_min there
    n_levels = max(1, -(-budget // L))                     # ceil: the last level may be partial
    alpha = params.alpha if params.alpha else params.t_min_ratio ** (1.0 / max(n_levels - 1, 1))
    T_min = params.t_min_ratio * T0

    x_best, J_best, info_best = x.copy(), J, dict(info)
    hist = {k: [] for k in ("J", "J_best", "candidate_J", "T", "accepted")}
    hist.update({f"best_{k}": [] for k in info})
    hist.update({k: [] for k in info})
    lv_T, lv_acc, lv_best = [], [], []
    if start:                                   # warm-start samples open the history (best so far)
        bJ, binfo = None, None
        for Jc, ic in start:
            if bJ is None or ic.get("feasible", 1.0) > binfo.get("feasible", 1.0) or                     (ic.get("feasible", 1.0) == binfo.get("feasible", 1.0) and Jc < bJ):
                bJ, binfo = Jc, ic
            hist["J"].append(bJ)
            hist["J_best"].append(bJ)
            hist["candidate_J"].append(Jc)
            hist["T"].append(np.nan)
            hist["accepted"].append(False)
            for k in info:
                hist[k].append(binfo.get(k, np.nan))
                hist[f"best_{k}"].append(binfo.get(k, np.nan))

    T, it, stop = T0, 0, "budget"
    while True:
        accepted_here = 0
        for _ in range(L):
            step = max(params.step_min, params.step0 * np.sqrt(T / T0))
            y = neighbour(x, step, rng)
            Jy, info_y = objective(y)
            n_evals += 1
            it += 1
            x_prev = x
            dJ = Jy - J
            p_acc = 1.0 if dJ <= 0 else float(np.exp(-dJ / T))
            accept = dJ <= 0 or rng.random() < p_acc                     # Metropolis criterion
            if accept:
                x, J, info = y, Jy, info_y
                accepted_here += 1
            # incumbent: a feasible design always beats an infeasible one, then lower J wins
            # (acceptance above still uses the penalised J; this only decides what is reported)
            fy, fb = info_y.get("feasible", 1.0), info_best.get("feasible", 1.0)
            improved = fy > fb or (fy == fb and Jy < J_best)
            if improved:
                x_best, J_best, info_best = y.copy(), Jy, dict(info_y)

            hist["J"].append(J)
            hist["J_best"].append(J_best)
            hist["candidate_J"].append(Jy)
            hist["T"].append(T)
            hist["accepted"].append(accept)
            for k in info:
                hist[k].append(info[k])
                hist[f"best_{k}"].append(info_best.get(k, np.nan))

            if callback is not None and callback(SAStep(it, T, J, J_best, Jy, accept, improved, x_best,
                                                        info_y, info_best, x_prev, y, x, dJ, p_acc,
                                                        info)):
                stop = "stopped by user"
                break
            if n_evals >= params.max_evals:
                break
        lv_T.append(T)
        lv_acc.append(accepted_here / L)
        lv_best.append(J_best)
        if stop != "budget" or n_evals >= params.max_evals:
            break
        T *= alpha
        if params.alpha and T < T_min:          # a fixed cooling factor may run out of temperature first
            stop = "T < T_min"
            break

    return SAResult(x_best, float(J_best), info_best,
                    {k: np.asarray(v, float) for k, v in hist.items()},
                    {"T": np.array(lv_T), "acceptance": np.array(lv_acc), "J_best": np.array(lv_best)},
                    T0, float(alpha), L, n_evals, time.perf_counter() - t_start, stop, params)


class SimulatedAnnealing(Algorithm):
    name = "sa"
    Params = SAParams

    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[SAStep], bool | None] | None = None) -> SAResult:
        p = self.params
        rng = np.random.default_rng(p.seed)
        start = None
        if x0 is not None:                              # start from a given design
            x0 = problem.repair(x0)
        elif p.n_init > 1:                              # warm start: best of n_init random designs
            start, best = [], None
            for _ in range(p.n_init):
                x = problem.random_solution(rng)
                J, info = problem.objective(x)
                start.append((J, info))
                if best is None or better(J, info.get("feasible", 1.0), best[1], best[2].get("feasible", 1.0)):
                    best = (x, J, info)
            x0 = best[0]
        else:
            x0 = problem.random_solution(rng)
        L = p.chain_length or getattr(problem, "default_chain_length", 10 * problem.dim)
        move = problem.move(p.p_type) if hasattr(problem, "move") else gaussian_move(problem)
        return anneal(problem.objective, x0, move, p, L, rng, callback, start)


def solve(problem, params: SAParams | None = None, x0: np.ndarray | None = None,
          callback: Callable[[SAStep], bool | None] | None = None):
    """Run SA on the fleet design problem; returns (SA record, OrbitalResult of the best design)."""
    res = SimulatedAnnealing(params or SAParams()).run(problem, x0, callback)
    return res, problem.evaluate(res.x_best)

"""Simulated Annealing
Generic: works on any ``Problem`` (the fleet pipeline, or a benchmark function).

    x  <- best of n_init random designs that pass feasibility (warm start), J(x) <- cost
    T0 <- -median(uphill dJ of n_t0 trial moves) / ln(p0)    a typical worse move is
                                                              accepted with probability p0
    alpha <- (T_min / T0)^(1 / (levels - 1))                  cooling that reaches
                                                              T_min = t_min_ratio * T0 exactly
                                                              when the budget runs out
    for T = T0, alpha T0, alpha^2 T0, ...:
        repeat L times (Markov chain at temperature T):
            step <- max(step_min, step0 * sqrt(T / T0))       smaller moves as it cools
            y    <- repair(neighbour(x, step))
            dJ   <- J(y) - J(x)                                Algorithm -> Feasibility -> Engine
            accept y if dJ <= 0, else with probability exp(-dJ / T)       (Metropolis)
            best <- y if y is better (feasible first, then lower J)
    until the evaluation budget is used (or T < T_min with a fixed alpha)

Neighbour move: pick 1..max_vars_per_move of the coordinates that currently matter
(``problem.active_mask(x)``: for the fleet that is N, T and the four variables of each
FLYING spacecraft, so no evaluation is wasted on unused slots) and add Gaussian noise
sigma = step * (upper - lower bound).  An integer coordinate (N) moves by at least 1.
``problem.repair`` maps the result back into the box (angles wrap, a / inc / T reflect);
a move that repair maps back onto x, or widens (e.g. switching off the last spacecraft makes
repair switch another one on), is redrawn, so every move stays local.

Why the warm start: with the narrow camera (usable to 58.6 km) a large part of the box
(high orbits, short missions) images almost nothing, so A*/S_a is huge and flat there.
T0 estimated from such a start is ~100x too hot for the region of good designs and the
budget is spent on a random walk.  Starting from the best of a few random designs puts
T0 on the scale of the moves that matter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Algorithm, OptimResult, Problem, Recorder, Step, better, local_move


@dataclass
class SAParams:
    max_evals: int = 1500              # evaluation budget (including the start samples and T0 moves)
    n_init: int = 20                   # random feasible designs sampled to pick the start (0: one)
    chain_length: int | None = None    # L, moves per temperature; None -> max(10, dim)
    alpha: float | None = None         # geometric cooling factor; None -> fitted to the budget
    p0: float = 0.5                    # initial acceptance probability of a typical uphill move
                                       # (0.5 beat 0.8 on 1- and 10-day missions, as in the M3 study)
    t_min_ratio: float = 1e-3          # final temperature T_min = t_min_ratio * T0
    step0: float = 0.25                # initial move size, fraction of each variable's range
    step_min: float = 0.01             # smallest move size
    max_vars_per_move: int = 2         # a move changes 1 .. this many coordinates
    n_t0_samples: int = 20             # trial moves used to estimate T0
    seed: int = 0


class SimulatedAnnealing(Algorithm):
    name = "sa"
    Params = SAParams

    # ------------------------------------------------------------------ the move
    def neighbour(self, problem: Problem, x: np.ndarray, step: float, rng: np.random.Generator) -> np.ndarray:
        return local_move(problem, x, step, rng, self.params.max_vars_per_move)

    # ------------------------------------------------------------------ T0
    def initial_temperature(self, problem: Problem, rec: Recorder, x0, ev0, rng) -> float:
        """T0 = -median(dJ+) / ln(p0) over trial moves from x0 (feasible candidates only, so a
        rejected design's penalty does not blow the start temperature up)."""
        ups = []
        for _ in range(self.params.n_t0_samples):
            if rec.exhausted:
                break
            ev = rec.evaluate(self.neighbour(problem, x0, self.params.step0, rng))
            dJ = ev.cost - ev0.cost
            if dJ > 0 and ev.feasible:
                ups.append(dJ)
        if not ups:
            return 1e-3 * max(abs(ev0.cost), 1e-3)
        return float(-np.median(ups) / np.log(self.params.p0))

    # ------------------------------------------------------------------ run
    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[Step], bool | None] | None = None) -> OptimResult:
        t_start = time.perf_counter()
        p = self.params
        rng = np.random.default_rng(p.seed)
        rec = Recorder(problem, p.max_evals, callback)

        if x0 is not None:
            x = problem.repair(np.asarray(x0, float))
            ev = rec.evaluate(x)
        else:                                        # warm start: best of n_init random designs
            draw = getattr(problem, "random_feasible", problem.random_solution)
            x, ev = None, None
            for _ in range(max(1, p.n_init)):
                xi = draw(rng)
                ei = rec.evaluate(xi)
                if ev is None or better(ei.cost, ei.feasible, ev.cost, ev.feasible):
                    x, ev = xi, ei
        rec_start = rec.n_evals
        T0 = self.initial_temperature(problem, rec, x, ev, rng)
        L = max(1, int(p.chain_length or max(10, problem.dim)))
        budget = max(p.max_evals - rec.n_evals, 1)
        n_levels = max(1, -(-budget // L))                          # ceil: the last level may be partial
        alpha = p.alpha if p.alpha else p.t_min_ratio ** (1.0 / max(n_levels - 1, 1))
        T_min = p.t_min_ratio * T0

        lv_T, lv_acc, lv_best = [], [], []
        T, stop = T0, "budget used"
        while not rec.exhausted:
            accepted_here = moves_here = 0
            for _ in range(L):
                if rec.exhausted:
                    break
                step = max(p.step_min, p.step0 * np.sqrt(T / T0))
                y = self.neighbour(problem, x, step, rng)
                best_before = rec.best
                ev_y = rec.evaluate(y)
                improved = rec.best is ev_y and ev_y is not best_before
                dJ = ev_y.cost - ev.cost
                p_acc = 1.0 if dJ <= 0 else float(np.exp(-dJ / T))
                accept = dJ <= 0 or rng.random() < p_acc               # Metropolis criterion
                x_prev = x
                if accept:
                    x, ev = y, ev_y
                    accepted_here += 1
                moves_here += 1
                info_cur = ev.info()
                extra = {"T": T, "step": step, "dJ": dJ, "p_accept": p_acc, "current_cost": ev.cost,
                         **{f"current_{k}": v for k, v in info_cur.items() if k != "cost"},
                         "x_prev": x_prev, "x_current": x, "evaluation_current": ev}
                rec.report(y, ev_y, improved, accept, extra)
            lv_T.append(T)
            lv_acc.append(accepted_here / max(moves_here, 1))
            lv_best.append(rec.best.cost)
            T *= alpha
            if p.alpha and T < T_min:                  # a fixed alpha may run out of temperature first
                stop = "T < T_min"
                break

        return self._result(rec, t_start, stop, {
            "T0": T0, "n_start": rec_start, "alpha": float(alpha), "chain_length": L, "T_final": T,
            "levels": {"T": np.array(lv_T), "acceptance": np.array(lv_acc), "best_cost": np.array(lv_best)}})


__all__ = ["SAParams", "SimulatedAnnealing"]

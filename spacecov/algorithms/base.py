"""What every optimisation algorithm shares.

An algorithm only talks to a ``Problem`` (spacecov.pipeline.Pipeline, or a benchmark
function for validation).  It proposes decision vectors, reads back their cost, and
keeps the best one.  It knows nothing about orbits or asteroids, so the same code can be
checked on functions whose optimum is known.

Rules shared by all algorithms
  * minimise ``cost``
  * the best design: a feasible design always beats an infeasible one, then lower cost
  * ``callback(step)`` is called after every evaluation; returning True stops the run
  * ``max_evals`` is the budget, so different algorithms can be compared fairly
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Protocol

import numpy as np


class Problem(Protocol):
    dim: int

    def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...
    def evaluate(self, x) -> Any: ...                     # -> object with .cost, .feasible, .info()
    def repair(self, x) -> np.ndarray: ...                # map into the box
    def random_solution(self, rng: np.random.Generator) -> np.ndarray: ...
    def active_mask(self, x) -> np.ndarray: ...           # coordinates that matter for x
    integer_mask: np.ndarray
    periodic_mask: np.ndarray


def better(cost_a: float, feasible_a: bool, cost_b: float, feasible_b: bool) -> bool:
    """Is a strictly better than b?  Feasible first, then lower cost."""
    if feasible_a != feasible_b:
        return feasible_a
    return cost_a < cost_b


def local_move(problem: Problem, x, step: float, rng: np.random.Generator, max_vars: int = 2) -> np.ndarray:
    """A small random change of 1..max_vars of the coordinates that currently matter.

    Used as SA's neighbour move and as the GA's mutation.  Each chosen coordinate gets Gaussian
    noise sigma = step * (upper - lower bound); an integer coordinate (a fleet switch) moves by at
    least 1, i.e. it is flipped.  ``problem.repair`` maps the result back into the box (angles
    wrap, a / inc / T reflect).  A move that repair maps back onto x, or widens (switching off the
    last spacecraft makes repair switch another one on), is redrawn, so every move stays local.
    """
    lo, hi = problem.bounds()
    span = hi - lo
    x = np.asarray(x, float)
    candidates = np.flatnonzero(problem.active_mask(x))
    for _ in range(20):                          # redraw if repair maps the move back onto x
        y = x.copy()
        k = int(rng.integers(1, min(max_vars, len(candidates)) + 1))
        for j in rng.choice(candidates, size=k, replace=False):
            d = rng.normal(0.0, step * span[j])
            if problem.integer_mask[j]:
                d = np.rint(d)
                if d == 0:
                    d = rng.choice((-1.0, 1.0))
            y[j] += d
        y = problem.repair(y)
        changed = int(np.sum(np.abs(y - x) > 1e-12))
        if 1 <= changed <= max_vars:             # repair may not widen the move
            return y
    return y


@dataclass
class Step:
    """Reported after every evaluation (for logs and live charts)."""
    iteration: int                    # evaluations used so far
    x: np.ndarray                     # the candidate just evaluated
    evaluation: Any                   # what problem.evaluate returned for it
    best_x: np.ndarray
    best: Any                         # evaluation of the best design so far
    improved: bool                    # this candidate is the new best
    accepted: bool | None = None      # e.g. SA's Metropolis decision (None if not applicable)
    extra: dict = field(default_factory=dict)    # algorithm-specific numbers (SA: T, dJ, p_accept, ...)

    @property
    def cost(self) -> float:
        return self.evaluation.cost

    @property
    def best_cost(self) -> float:
        return self.best.cost


@dataclass
class OptimResult:
    algorithm: str
    params: dict
    x_best: np.ndarray
    best: Any                         # evaluation of x_best
    history: dict[str, np.ndarray]    # one entry per evaluation
    n_evals: int
    seconds: float
    stop_reason: str
    extra: dict = field(default_factory=dict)    # algorithm-specific summary (SA: T0, alpha, levels)

    @property
    def cost_best(self) -> float:
        return self.best.cost

    def summary(self) -> str:
        return (f"{self.algorithm}: {self.n_evals} evaluations in {self.seconds:.1f} s ({self.stop_reason}), "
                f"best cost {self.cost_best:.4f} ({'feasible' if self.best.feasible else 'INFEASIBLE'})")


class Recorder:
    """Evaluates through the problem, keeps the best design, the history and the budget."""

    def __init__(self, problem: Problem, max_evals: int, callback: Callable[[Step], bool | None] | None = None):
        self.problem = problem
        self.max_evals = int(max_evals)
        self.callback = callback
        self.n_evals = 0
        self.best_x: np.ndarray | None = None
        self.best = None
        self.stopped_by_user = False
        self._hist: dict[str, list] = {}

    @property
    def exhausted(self) -> bool:
        return self.n_evals >= self.max_evals or self.stopped_by_user

    def evaluate(self, x) -> Any:
        """Evaluate without reporting (e.g. the samples SA uses to pick T0)."""
        ev = self.problem.evaluate(x)
        self.n_evals += 1
        if self.best is None or better(ev.cost, ev.feasible, self.best.cost, self.best.feasible):
            self.best_x, self.best = np.asarray(x, float).copy(), ev
        return ev

    def report(self, x, ev, improved: bool, accepted: bool | None = None, extra: dict | None = None) -> bool:
        """Log one evaluation and call the callback; returns True if the run must stop."""
        extra = extra or {}
        row = {"iteration": self.n_evals, "cost": ev.cost, "best_cost": self.best.cost,
               "feasible": float(ev.feasible), "accepted": np.nan if accepted is None else float(accepted)}
        row.update({k: v for k, v in ev.info().items() if k not in row})
        row.update({f"best_{k}": v for k, v in self.best.info().items() if k != "cost"})
        row.update({k: v for k, v in extra.items() if np.isscalar(v)})
        for k, v in row.items():
            self._hist.setdefault(k, []).append(float(v))
        if self.callback is not None and self.callback(
                Step(self.n_evals, np.asarray(x, float), ev, self.best_x, self.best, improved, accepted, extra)):
            self.stopped_by_user = True
        return self.exhausted

    def history(self) -> dict[str, np.ndarray]:
        n = max((len(v) for v in self._hist.values()), default=0)
        return {k: np.asarray(v + [np.nan] * (n - len(v)), float) for k, v in self._hist.items()}


class Algorithm(ABC):
    """Base class.  Subclasses define ``name``, a ``Params`` dataclass and ``run``."""
    name: str = ""
    Params: type = object

    def __init__(self, params=None, **overrides):
        p = params if params is not None else self.Params()
        self.params = replace(p, **overrides) if overrides else p

    @abstractmethod
    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[Step], bool | None] | None = None) -> OptimResult: ...

    def _result(self, rec: Recorder, t_start: float, stop_reason: str, extra: dict | None = None) -> OptimResult:
        if rec.stopped_by_user:
            stop_reason = "stopped by user"
        return OptimResult(self.name, asdict(self.params), rec.best_x, rec.best, rec.history(), rec.n_evals,
                           time.perf_counter() - t_start, stop_reason, extra or {})

    @classmethod
    def param_names(cls) -> list[str]:
        return [f.name for f in fields(cls.Params)]

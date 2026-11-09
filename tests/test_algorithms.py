"""Algorithms: SA validated on functions with a known optimum, and on the fleet pipeline."""
import numpy as np
import pytest

from spacecov.algorithms import available, get, make
from spacecov.algorithms.base import better
from spacecov.algorithms.benchmarks import BenchmarkProblem
from spacecov.algorithms.random_search import RandomSearch, random_search
from spacecov.algorithms.sa import SimulatedAnnealing
from spacecov.asteroid import make_asteroid
from spacecov.config import MissionConfig
from spacecov.pipeline import Pipeline


def test_registry_and_best_rule():
    assert get("sa") is SimulatedAnnealing and get("random") is RandomSearch and set(available()) >= {"sa", "random"}
    assert make("sa", max_evals=10).params.max_evals == 10
    assert better(5.0, True, 1.0, False)          # feasible beats infeasible, whatever the cost
    assert better(1.0, True, 2.0, True) and not better(2.0, True, 1.0, True)


@pytest.mark.parametrize("name, dim, tol", [("sphere", 5, 2e-2), ("rastrigin", 2, 0.1), ("ackley", 2, 0.1)])
def test_sa_reaches_known_optimum(name, dim, tol):
    best = min(make("sa", max_evals=6000, seed=s).run(BenchmarkProblem(name, dim)).cost_best for s in range(3))
    assert best < tol                                     # global minimum is 0 at x = 0


def test_sa_schedule_and_bookkeeping():
    r = make("sa", max_evals=1200, seed=0).run(BenchmarkProblem("rastrigin", 2))
    h = r.history
    assert r.n_evals == 1200 and len(h["cost"]) == 1200 - 20 - 20        # start samples + T0 moves not reported
    assert np.all(np.diff(h["best_cost"]) <= 0)                           # best-so-far never gets worse
    assert np.all(np.diff(h["T"]) <= 0)                                   # temperature only falls
    assert r.extra["levels"]["T"][-1] == pytest.approx(r.extra["T0"] * 1e-3, rel=1e-6)
    acc = r.extra["levels"]["acceptance"]
    assert acc[: len(acc) // 4].mean() > acc[-len(acc) // 4:].mean()     # uphill moves get rarer


def test_fixed_alpha_stops_at_t_min_and_callback_stops():
    r = make("sa", max_evals=100000, alpha=0.5, chain_length=10, seed=0).run(BenchmarkProblem("sphere", 2))
    assert r.stop_reason == "T < T_min" and r.n_evals < 1000
    r = make("sa", max_evals=5000, seed=0).run(BenchmarkProblem("sphere", 2), callback=lambda s: s.iteration >= 100)
    assert r.stop_reason == "stopped by user" and r.n_evals == 100


@pytest.fixture(scope="module")
def small_pipe():
    cfg = MissionConfig(n_max=3, t_max_days=1.0, t_min_days=0.25, dt_min=20)
    return Pipeline(make_asteroid({"source": "eros", "faces": 1500}), cfg)


def test_neighbour_changes_only_active_coordinates(small_pipe):
    sa = SimulatedAnnealing()
    rng = np.random.default_rng(0)
    lo, hi = small_pipe.bounds()
    x = small_pipe.random_feasible(rng)
    changed_n = 0
    for _ in range(300):
        y = sa.neighbour(small_pipe, x, 0.3, rng)
        diff = np.abs(y - x) > 1e-12
        assert 1 <= diff.sum() <= 2 and not np.any(diff & ~small_pipe.active_mask(x))
        assert np.all(y >= lo - 1e-9) and np.all(y <= hi + 1e-9) and small_pipe.space.n_of(y) >= 1
        changed_n += small_pipe.space.n_of(y) != small_pipe.space.n_of(x)
        x = y
    assert changed_n > 0                                                  # N really is searched (switches)


def test_sa_on_the_fleet_beats_random_search(small_pipe):
    r = make("sa", max_evals=250, seed=2).run(small_pipe)
    assert r.best.feasible and r.cost_best <= r.history["cost"][0]
    assert r.cost_best == pytest.approx(small_pipe.evaluate(r.x_best).cost)
    sa_J = [make("sa", max_evals=400, seed=k).run(small_pipe).cost_best for k in range(3)]
    rs_J = [random_search(small_pipe, 400, seed=k).cost_best for k in range(3)]
    assert np.mean(sa_J) < np.mean(rs_J)

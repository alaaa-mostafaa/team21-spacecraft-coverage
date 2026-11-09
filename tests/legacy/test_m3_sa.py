"""Milestone 3 (frozen, spacecov.legacy.m3) - Simulated annealing: validation on benchmarks with a known optimum, and behaviour on the orbital problem."""
import numpy as np
import pytest

from spacecov.legacy.m3 import algorithms as sa
from spacecov.legacy.m3.algorithms.benchmarks import solve_benchmark
from spacecov.asteroid import make_asteroid
from spacecov.legacy.m3.config import OrbitalConfig
from spacecov.legacy.m3.problem import OrbitalProblem


@pytest.mark.parametrize("name, dim, tol", [("sphere", 5, 1e-2), ("rastrigin", 2, 0.1), ("ackley", 2, 0.1)])
def test_benchmarks_reach_known_optimum(name, dim, tol):
    best = min(solve_benchmark(name, dim, sa.SAParams(max_evals=6000, seed=s)).J_best for s in range(3))
    assert best < tol                                     # global minimum is 0 at x = 0


def test_metropolis_and_schedule_bookkeeping():
    res = solve_benchmark("rastrigin", 2, sa.SAParams(max_evals=1200, seed=0))
    h = res.history
    # budget = 20 warm-start designs + 20 trial moves for T0 + annealing; the history records all but the T0 moves
    assert res.n_evals == 1200 and len(h["J"]) == 1200 - 20
    assert np.all(np.isnan(h["T"][:20])) and not np.any(np.isnan(h["T"][20:]))   # warm start, then annealing
    assert np.all(np.diff(h["J_best"]) <= 0)              # best-so-far never gets worse
    assert np.all(h["J_best"] <= h["J"] + 1e-12)
    assert np.all(np.diff(h["T"][20:]) <= 0)              # temperature only falls
    assert res.levels["T"][-1] == pytest.approx(res.T0 * 1e-3, rel=1e-6)   # auto alpha ends at T_min
    # uphill moves get rarer as the system cools
    acc = res.levels["acceptance"]
    assert acc[: len(acc) // 4].mean() > acc[-len(acc) // 4:].mean()


def test_fixed_alpha_stops_at_t_min():
    res = solve_benchmark("sphere", 2, sa.SAParams(max_evals=100000, alpha=0.5, chain_length=10, seed=0))
    assert res.stop_reason == "T < T_min" and res.n_evals < 1000


@pytest.fixture(scope="module")
def small_problem():
    ast = make_asteroid({"source": "eros", "faces": 1500})
    return OrbitalProblem(ast, 3, OrbitalConfig(duration_days=1.0, dt_min=20))


def test_neighbour_changes_one_spacecraft_and_stays_feasible(small_problem):
    prob = small_problem
    nb = prob.move(p_type=0.3)
    rng = np.random.default_rng(0)
    x = prob.random_solution(rng)
    for _ in range(200):
        y = nb(x, 0.3, rng)
        changed = np.any(np.abs(y.reshape(3, 7) - x.reshape(3, 7)) > 1e-12, axis=1)
        assert changed.sum() <= 1
        sol = prob.decode(y)
        assert all(s.periapsis >= prob.r_min - 1e-9 and s.apoapsis <= prob.cfg.r_max + 1e-9
                   for s in sol.spacecraft)
        assert set(sol.types) <= {0, 1, 2} and sol.n_active >= 1           # slots: unused / narrow / wide
        x = y
    # over many moves, the fleet size really changes
    sizes = set()
    for _ in range(300):
        x = nb(x, 0.3, rng)
        sizes.add(prob.decode(x).n_active)
    assert len(sizes) >= 2


def test_sa_improves_on_its_start_and_beats_random_search(small_problem):
    prob = small_problem
    res, best = sa.solve(prob, sa.SAParams(max_evals=250, seed=2))
    assert res.J_best <= res.history["J"][0]
    assert best.J == pytest.approx(res.J_best)
    # on average over seeds (a single short run can be unlucky), SA beats random search
    sa_J = [sa.solve(prob, sa.SAParams(max_evals=600, seed=k))[0].J_best for k in range(3)]
    rs_J = [sa.random_search(prob, 600, seed=k)["J_best"][-1] for k in range(3)]
    assert np.mean(sa_J) < np.mean(rs_J)

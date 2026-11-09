"""Milestone 3 (frozen, spacecov.legacy.m3) - The optimiser-facing structure: problem interface, design space, feasibility, algorithm registry."""
import numpy as np
import pytest

from spacecov.legacy.m3.algorithms import ALGORITHMS, SAParams, get_algorithm
from spacecov.legacy.m3.algorithms.benchmarks import BenchmarkProblem
from spacecov.legacy.m3.algorithms.random_search import RandomSearchParams
from spacecov.asteroid import make_asteroid
from spacecov.legacy.m3.config import OrbitalConfig
from spacecov.legacy.m3.design import N_VARS
from spacecov.legacy.m3.feasibility import constraints, is_feasible, penalty
from spacecov.legacy.m3.problem import OrbitalProblem


@pytest.fixture(scope="module")
def fleet():
    return OrbitalProblem(make_asteroid({"source": "eros", "faces": 1500}), 4, OrbitalConfig(duration_days=0.25))


def test_both_problems_have_the_same_interface(fleet):
    for p in (fleet, BenchmarkProblem("sphere", 3)):
        lo, hi = p.bounds()
        assert lo.shape == hi.shape == (p.dim,)
        assert p.integer_mask.shape == p.periodic_mask.shape == (p.dim,)
        x = p.random_solution(np.random.default_rng(0))
        J, info = p.objective(x)
        assert np.isfinite(J) and isinstance(info, dict)


def test_design_space_masks(fleet):
    sp = fleet.space
    assert sp.dim == N_VARS * 4 and len(sp.names) == sp.dim
    assert sp.integer_mask.sum() == 4                       # one slot switch per slot
    assert sp.periodic_mask.sum() == 3 * 4                  # RAAN, argp, M0
    x = fleet.random_solution(np.random.default_rng(1)).reshape(4, N_VARS)
    x[:, 6] = [1, 0, 1, 0]
    act = sp.active_mask(x.ravel()).reshape(4, N_VARS)
    assert act[0].all() and act[2].all()                    # flying spacecraft: every variable matters
    assert not act[1, :6].any() and act[1, 6]               # unused slot: only its switch


def test_repair_always_gives_a_valid_design(fleet):
    rng = np.random.default_rng(2)
    lo, hi = fleet.bounds()
    for _ in range(50):
        y = fleet.repair(rng.normal(0, 400, fleet.dim)).reshape(4, N_VARS)
        a, e = y[:, 0], y[:, 1]
        assert np.all(a * (1 - e) >= fleet.r_min - 1e-9) and np.all(a * (1 + e) <= fleet.cfg.r_max + 1e-9)
        assert np.all((y[:, 3:6] >= 0) & (y[:, 3:6] < 360)) and np.all((y[:, 2] >= 0) & (y[:, 2] <= 180))
        assert (y[:, 6] > 0).any()                          # at least one spacecraft flies


def test_feasibility_signs():
    cfg = OrbitalConfig()
    r = np.zeros((2, 3, 3))
    r[0, :, 0], r[1, :, 0] = 40.0, 45.0                     # 5 km apart all the time
    g, d = constraints(np.array([40.0, 45.0]), np.zeros(2), r, np.full((2, 3), 0.9), 26.4, cfg)
    assert d == pytest.approx(5.0) and is_feasible(g) and penalty(g, cfg) == 0.0
    r[1, :, 0] = 41.0                                       # 1 km apart: separation violated
    g, _ = constraints(np.array([40.0, 41.0]), np.zeros(2), r, np.full((2, 3), 0.9), 26.4, cfg)
    assert g[2] > 0 and not is_feasible(g) and penalty(g, cfg) > 0


def test_registry_runs_every_algorithm_on_a_benchmark():
    assert {"sa", "random"} <= set(ALGORITHMS)
    sa = get_algorithm("sa", SAParams(max_evals=3000, seed=0)).run(BenchmarkProblem("sphere", 2))
    rs = get_algorithm("random", RandomSearchParams(max_evals=200, seed=0)).run(BenchmarkProblem("sphere", 2))
    assert sa.J_best < 1e-2 and rs["J_best"][-1] < 5.0
    with pytest.raises(ValueError):
        get_algorithm("nope")


def test_sa_works_on_a_problem_without_its_own_move():
    """Only the minimal interface (no move, no chain length): SA falls back to a Gaussian move."""
    class Plain:
        dim = 2

        def bounds(self):
            return np.full(2, -5.0), np.full(2, 5.0)

        def repair(self, x):
            return np.clip(x, -5.0, 5.0)

        def random_solution(self, rng):
            return rng.uniform(-5.0, 5.0, 2)

        def objective(self, x):
            return float(np.sum(x ** 2)), {}

    res = get_algorithm("sa", SAParams(max_evals=3000, seed=1)).run(Plain())
    assert res.J_best < 1e-2

"""Genetic Algorithm: settings rules, every operator on its own, known optima, and the fleet pipeline."""
import numpy as np
import pytest

from spacecov.algorithms import available, get, make
from spacecov.algorithms.benchmarks import BenchmarkProblem
from spacecov.algorithms.ga import (OP_NAMES, GeneticAlgorithm, _Genome, auto_pop_size, n_generations,
                                    split)
from spacecov.algorithms.random_search import random_search
from spacecov.asteroid import make_asteroid
from spacecov.config import MissionConfig
from spacecov.pipeline import Pipeline


@pytest.fixture(scope="module")
def small_pipe():
    cfg = MissionConfig(n_max=3, t_max_days=1.0, t_min_days=0.25, dt_min=20)
    return Pipeline(make_asteroid({"source": "eros", "faces": 1500}), cfg)


def fleet(pipe, T, orbits):
    """Vector with the given (a, inc, raan, M0) orbits switched on in the first slots."""
    x = pipe.random_solution(np.random.default_rng(0))
    x[0] = T
    for i in range(pipe.space.n_max):
        on = i < len(orbits)
        j = pipe.space.slot_index(i, "a")
        if on:
            x[j:j + 4] = orbits[i]
        x[pipe.space.slot_index(i, "on")] = 1.0 if on else 0.0
    return x


# ---------------------------------------------------------------- settings (report Tables popsize / split)
def test_registry_and_settings_rules():
    assert get("ga") is GeneticAlgorithm and "ga" in available()
    assert auto_pop_size(26) == 40 and auto_pop_size(51) == 60 and auto_pop_size(2) == 30
    assert split(40, 0.05, 0.8) == (2, 30, 8) and split(60, 0.05, 0.8) == (3, 46, 11)
    assert split(40, 0.0, 0.8) == (0, 32, 8) and split(40, 0.2, 0.8) == (8, 26, 6)
    assert n_generations(2000, 40, 2) == 53 and n_generations(2000, 60, 3) == 36
    assert split(10, 1.0, 1.0)[0] == 9                        # always at least one child per generation


# ---------------------------------------------------------------- operators
def test_alignment_undoes_a_slot_permutation(small_pipe):
    g = _Genome(small_pipe)
    a = fleet(small_pipe, 0.5, [(30, 20, 10, 0), (60, 90, 120, 90), (45, 150, 250, 200)])
    perm = [2, 0, 1]
    b = a.copy()
    for i, j in enumerate(perm):                              # same fleet, other slot order
        b[g.slots[i]] = a[g.slots[j]]
    assert not np.allclose(b, a)
    assert np.allclose(g.align(a, b), a)


def test_alignment_pairs_flying_slots_first(small_pipe):
    g = _Genome(small_pipe)
    a = fleet(small_pipe, 0.5, [(30, 20, 10, 0)])                              # slot 1 flies
    b = fleet(small_pipe, 0.5, [(80, 160, 200, 180), (31, 21, 12, 3)])        # slots 1, 2 fly
    y = g.align(a, b)
    assert g.flying(y, 0) and y[g.slots[0]][0] == pytest.approx(31)          # A's flyer meets B's nearest flyer


def test_blend_takes_the_shorter_arc_for_angles(small_pipe):
    ga = GeneticAlgorithm(p_blend=1.0, blend_weight=0.5)
    g = _Genome(small_pipe)
    a = fleet(small_pipe, 0.5, [(40, 30, 350, 100)])
    b = fleet(small_pipe, 0.5, [(50, 50, 10, 300)])
    c1, c2, blended = ga.crossover(small_pipe, g, a, b, np.random.default_rng(0))
    s = small_pipe.space
    for c in (c1, c2):
        assert blended
        assert c[s.slot_index(0, "a")] == pytest.approx(45) and c[s.slot_index(0, "inc")] == pytest.approx(40)
        assert min(c[s.slot_index(0, "raan")], 360 - c[s.slot_index(0, "raan")]) == pytest.approx(0, abs=1e-9)
        assert c[s.slot_index(0, "M0")] == pytest.approx(20)                 # 100 and 300: midpoint via 0, not 200
        assert c[0] == pytest.approx(0.5)


def test_swap_children_are_made_of_parent_slots(small_pipe):
    ga = GeneticAlgorithm(p_blend=0.0)
    g = _Genome(small_pipe)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = small_pipe.random_solution(rng), small_pipe.random_solution(rng)
        b_al = g.align(a, b)
        c1, c2, blended = ga.crossover(small_pipe, g, a, b, rng)
        assert not blended
        for i, sl in enumerate(g.slots):                      # every slot comes whole from one parent ...
            assert np.allclose(c1[sl], a[sl]) or np.allclose(c1[sl], b_al[sl])
            other = b_al[sl] if np.allclose(c1[sl], a[sl]) else a[sl]
            assert np.allclose(c2[sl], other)                 # ... and the second child gets the other one


def test_children_stay_in_the_box(small_pipe):
    ga = GeneticAlgorithm(p_blend=0.5)
    g = _Genome(small_pipe)
    lo, hi = small_pipe.bounds()
    rng = np.random.default_rng(2)
    for _ in range(100):
        c1, c2, _ = ga.crossover(small_pipe, g, small_pipe.random_solution(rng), small_pipe.random_solution(rng), rng)
        for c in (c1, c2):
            assert np.all(c >= lo - 1e-9) and np.all(c <= hi + 1e-9) and small_pipe.space.n_of(c) >= 1


def test_tournament_prefers_feasible_then_lower_cost():
    ga = GeneticAlgorithm(tournament_k=40)                    # k = P: (almost) always the best
    costs = np.array([0.5, 0.1, 1000.0, 0.3] * 10)
    feas = np.array([True, True, False, True] * 10)
    rng = np.random.default_rng(0)
    wins = [ga.tournament(costs, feas, rng) for _ in range(50)]
    assert np.mean([costs[w] == 0.1 for w in wins]) > 0.9
    two = GeneticAlgorithm(tournament_k=2)                    # second crossover parent: another individual
    assert all(two.tournament(np.array([0.1, 0.5]), np.array([True, True]), rng, exclude=0) == 1
               for _ in range(20))


def test_stratified_start_has_every_fleet_size_equally(small_pipe):
    ga = GeneticAlgorithm()
    xs = ga._initial(small_pipe, _Genome(small_pipe), 30, None, np.random.default_rng(0))
    counts = np.bincount([small_pipe.space.n_of(x) for x in xs], minlength=4)[1:]
    assert list(counts) == [10, 10, 10]
    uni = GeneticAlgorithm(stratified=False)._initial(small_pipe, _Genome(small_pipe), 30, None,
                                                      np.random.default_rng(0))
    assert len(uni) == 30


# ---------------------------------------------------------------- whole algorithm
@pytest.mark.parametrize("name, dim, tol", [("sphere", 5, 1e-6), ("rastrigin", 2, 1e-3), ("rastrigin", 5, 1e-2),
                                            ("ackley", 2, 1e-2), ("ackley", 5, 1e-2)])
def test_ga_reaches_known_optimum(name, dim, tol):
    best = min(make("ga", max_evals=6000, seed=s).run(BenchmarkProblem(name, dim)).cost_best for s in range(3))
    assert best < tol                                         # global minimum is 0 at x = 0


def test_ga_bookkeeping():
    r = make("ga", max_evals=1000, seed=0).run(BenchmarkProblem("rastrigin", 2))
    h, e = r.history, r.extra
    assert r.n_evals == 1000 and len(h["cost"]) == 1000                   # every evaluation is reported
    assert np.all(np.diff(h["best_cost"]) <= 0)                           # best-so-far never gets worse
    assert set(np.unique(h["operator"])) <= set(OP_NAMES)
    assert e["pop_size"] == 30 and (e["n_elite"], e["n_crossover"], e["n_mutation"]) == (2, 22, 6)
    gs = e["generations"]
    assert np.all(np.diff(gs["best"]) <= 0) and gs["diversity"][-1] < gs["diversity"][0]
    assert sum(e["improvements_by_operator"].values()) >= 1


def test_ga_on_the_fleet_beats_random_search(small_pipe):
    r = make("ga", max_evals=300, seed=2).run(small_pipe)
    assert r.best.feasible and r.cost_best == pytest.approx(small_pipe.evaluate(r.x_best).cost)
    ga_J = [make("ga", max_evals=400, seed=k).run(small_pipe).cost_best for k in range(3)]
    rs_J = [random_search(small_pipe, 400, seed=k).cost_best for k in range(3)]
    assert np.mean(ga_J) < np.mean(rs_J)


def test_ga_starts_from_a_given_design_and_respects_a_fixed_fleet(small_pipe):
    x0 = fleet(small_pipe, 0.5, [(40, 60, 0, 0), (50, 120, 90, 180)])
    r = make("ga", max_evals=60, seed=0).run(small_pipe, x0)
    assert r.history["cost"][0] == pytest.approx(small_pipe.evaluate(small_pipe.repair(x0)).cost)
    cfg = MissionConfig(n_max=2, t_max_days=0.5, t_min_days=0.25, dt_min=20, fixed_n=True)
    pipe = Pipeline(make_asteroid({"source": "eros", "faces": 1500}), cfg)
    r = make("ga", max_evals=80, seed=0).run(pipe)
    assert pipe.space.n_of(r.x_best) == 2

"""Design space, feasibility, engine (cost F) and the pipeline that chains them."""
import numpy as np
import pytest

from spacecov.asteroid import make_asteroid
from spacecov.config import MissionConfig
from spacecov.design import CircularOrbit, Design, DesignSpace
from spacecov.pipeline import Pipeline
from spacecov.scenario import Scenario


@pytest.fixture(scope="module")
def pipe():
    cfg = MissionConfig(n_max=4, t_max_days=2.0, t_min_days=0.25, dt_min=20)
    return Pipeline(make_asteroid({"source": "eros", "faces": 1500}), cfg)


def design(T=1.0, *orbits):
    return Design(T, [CircularOrbit(*o) for o in orbits])


# ---------------------------------------------------------------- design space
def test_layout_bounds_and_round_trip():
    sp = DesignSpace(n_max=3, r_min=26.4, r_max=100.0, t_min_days=0.25, t_max_days=10.0)
    assert sp.dim == 1 + 5 * 3
    assert sp.names[:7] == ["T", "a_1", "inc_1", "raan_1", "M0_1", "on_1", "a_2"]
    lo, hi = sp.bounds()
    assert lo[:6].tolist() == [0.25, 26.4, 0, 0, 0, 0] and hi[:6].tolist() == [10, 100, 180, 360, 360, 1]
    assert np.flatnonzero(sp.integer_mask).tolist() == [5, 10, 15]     # the on/off switches
    assert sp.periodic_mask.sum() == 2 * 3                             # RAAN and M0 of every slot
    d = design(2.5, (40, 60, 10, 20), (55, 120, 200, 300))
    x = sp.encode(d)
    back = sp.decode(x)
    assert back.N == 2 and back.T_days == 2.5 and back.orbits[1].key() == (55, 120, 200, 300)
    assert sp.n_of(x) == 2 and x[15] == 0                               # slot 3 switched off
    assert sp.active_mask(x).sum() == 1 + 5 * 2 + 1                    # T, two flying slots, slot 3's switch
    y = x.copy()
    y[5] = 0                                                           # switch slot 1 off: slot 2 flies alone
    assert sp.decode(y).N == 1 and sp.decode(y).orbits[0].key() == (55, 120, 200, 300)


def test_orbit_is_circular_with_fixed_camera():
    sc = CircularOrbit(40, 30, 10, 5).spacecraft(camera=1)
    assert (sc.e, sc.argp, sc.type) == (0.0, 0.0, 1)


# ---------------------------------------------------------------- feasibility
def test_feasibility_checks(pipe):
    sp, fe = pipe.space, pipe.feasibility
    good = sp.encode(design(1.0, (40, 30, 0, 0), (60, 100, 90, 180)))
    assert fe.check(good).feasible
    cases = {"T": (0, 5.0), "g1 r_min": (1, 20.0), "g2 r_max": (1, 120.0), "inc": (2, 200.0)}
    for name, (j, val) in cases.items():
        x = good.copy()
        x[j] = val
        rep = fe.check(x)
        assert not rep.feasible and rep.violations[name] > 0, name
    x = good.copy()
    x[[5, 10]] = 0                                                       # every slot switched off
    assert fe.check(x).violations["N"] > 0
    x = good.copy()
    x[5] = 0.5                                                           # a switch must be 0 or 1
    assert fe.check(x).violations["N"] > 0
    same = sp.encode(design(1.0, (40, 30, 0, 0), (40, 30, 0, 0)))      # two spacecraft in one place
    rep = fe.check(same)
    assert rep.min_separation == pytest.approx(0) and rep.violations["g3 separation"] > 0


def test_repair_maps_into_the_box(pipe):
    rng = np.random.default_rng(0)
    lo, hi = pipe.bounds()
    for _ in range(200):
        x = pipe.repair(rng.normal(0, 300, pipe.dim))
        assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)
        switches = x[pipe.integer_mask]
        assert set(switches.tolist()) <= {0.0, 1.0} and switches.sum() >= 1
        rep = pipe.feasibility.check(x)
        assert all(rep.violations[k] <= 0 for k in ("N", "T", "g1 r_min", "g2 r_max", "inc"))


# ---------------------------------------------------------------- engine
def test_fitness_terms(pipe):
    e, cfg = pipe.engine, pipe.cfg
    dv, e_elec = np.array([1e-3, 2e-3]), np.array([5e6, 6e6])
    c = e.cost(N=2, T=1.0 * 86400, S_a=0.5 * e.a_star, dv=dv, e_elec=e_elec)
    assert c.C == pytest.approx(0.5) and c.Z == pytest.approx(2 / 10)            # Z = N / n_ref
    assert c.f_time == pytest.approx(1.0 / cfg.t_max_days)
    E = 0.5 * dv.sum() / (10 * cfg.dv_ref) + 0.5 * e_elec.sum() / (10 * cfg.e_ref_j)
    assert c.E == pytest.approx(E)
    w_c, w_z, w_e, w_t = cfg.weights
    assert c.J == pytest.approx(w_c * 0.5 + w_z * 0.2 + w_e * E + w_t * c.f_time)
    best = e.cost(1, cfg.t_min, e.a_star, np.zeros(1), np.zeros(1))              # full map, 1 s/c, shortest
    assert 0 <= best.J < c.J <= 1
    assert e.cost(1, 86400, 0.0, np.zeros(1), np.zeros(1)).J <= 1               # empty map: finite, <= 1


def test_reference_normalization_does_not_saturate_above_ten_spacecraft(pipe):
    e, cfg = pipe.engine, pipe.cfg
    c10 = e.cost(10, cfg.t_min, e.a_star, np.zeros(10), np.zeros(10))
    c20 = e.cost(20, cfg.t_min, e.a_star, np.zeros(20), np.zeros(20))
    assert c10.Z == pytest.approx(1.0)
    assert c20.Z == pytest.approx(2.0)
    assert c20.E == pytest.approx(c10.E * 2.0)
    assert c20.J > c10.J




def test_engine_coverage_grows_with_time_and_fleet(pipe):
    o1, o2 = (45, 60, 0, 0), (55, 120, 90, 180)
    r_short = pipe.engine.simulate(design(0.5, o1))
    r_long = pipe.engine.simulate(design(2.0, o1))
    r_two = pipe.engine.simulate(design(2.0, o1, o2))
    assert 0 < r_short.S_a <= r_long.S_a <= r_two.S_a <= r_two.a_star + 1e-9
    assert r_long.times[-1] <= 2.0 * 86400 and len(r_short.times) == pipe.cfg.steps_within(0.5 * 86400)
    curve, each = r_two.coverage_curve()
    assert np.all(np.diff(curve) >= -1e-12) and curve[-1] == pytest.approx(r_two.C)
    # same spacecraft, shorter T: identical to cutting the long mission short (tracks are reused)
    n = pipe.engine.n_tracks
    again = pipe.engine.simulate(design(0.5, o1))
    assert pipe.engine.n_tracks == n and again.S_a == r_short.S_a


# ---------------------------------------------------------------- pipeline
def test_pipeline_rejects_before_the_engine(pipe):
    sims = pipe.engine.n_simulations
    x = pipe.space.encode(design(1.0, (20, 0, 0, 0)))                    # inside r_min
    ev = pipe.evaluate(x)
    assert not ev.feasible and ev.result is None and pipe.engine.n_simulations == sims
    assert ev.cost == pytest.approx(pipe.cfg.infeasible_cost + pipe.cfg.rho * ev.report.penalty)


def test_pipeline_feasible_cost_is_F_and_beats_any_infeasible(pipe):
    x = pipe.space.encode(design(1.0, (45, 60, 0, 0), (55, 120, 90, 180)))
    ev = pipe.evaluate(x)
    assert ev.feasible and ev.cost == pytest.approx(ev.result.J) and 0 < ev.cost < 1
    assert ev.report.min_soc is not None and ev.report.violations["g4 battery"] <= 0
    info = ev.info()
    assert info["N"] == 2 and info["T_days"] == 1.0 and info["C"] == pytest.approx(ev.result.C)


def test_example_scenario_round_trip(tmp_path):
    sc = Scenario.load("scenarios/orbital_example.json")
    assert sc.design.N == 3 and sc.algorithm == "sa"
    p = tmp_path / "s.json"
    sc.save(p)
    assert Scenario.load(p).to_dict() == sc.to_dict()
    with pytest.raises(ValueError, match="Milestone 3"):
        Scenario.load("spacecov/legacy/m3/example.json")


def test_fixed_fleet_size_mode():
    cfg = MissionConfig(n_max=3, t_max_days=1.0, dt_min=20, fixed_n=True)
    p = Pipeline(make_asteroid({"source": "eros", "faces": 1500}), cfg)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = p.repair(rng.normal(0, 300, p.dim))
        assert p.space.n_of(x) == 3                                     # every slot flies
    assert not np.any(p.active_mask(x) & p.integer_mask)                # switches never move
    x[5] = 0
    assert p.feasibility.check(x).violations["N"] > 0

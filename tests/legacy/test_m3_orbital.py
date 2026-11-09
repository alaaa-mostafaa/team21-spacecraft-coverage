"""Milestone 3 (frozen, spacecov.legacy.m3)."""
import numpy as np
import pytest

from spacecov.asteroid import Asteroid, make_asteroid
from spacecov.legacy.m3.config import OrbitalConfig
from spacecov.legacy.m3.design import OrbitalSolution, OrbitalSpacecraft
from spacecov.legacy.m3.engine import Engine, evaluate
from spacecov.legacy.m3.physics import (battery_track, delta_v, eclipse_track, lit_mask, propagate, solve_kepler, to_body,
                              visibility_track)
from spacecov.legacy.m3.problem import OrbitalProblem
from spacecov.legacy.m3.scenario import Scenario

CFG = OrbitalConfig()


def test_kepler_residual():
    M = np.linspace(0, 2 * np.pi, 1000)
    for e in (0.0, 0.1, 0.3, 0.5):
        E = solve_kepler(M, e)
        assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-10


def test_period_speed_and_step_count():
    sc = OrbitalSpacecraft(a=35.0)
    assert sc.period(CFG.mu) / 3600 == pytest.approx(17.1, abs=0.05)       # Section 4.5.1
    assert sc.speed(35.0, CFG.mu) * 1000 == pytest.approx(3.57, abs=0.01)  # circular, m/s
    assert OrbitalConfig(duration_days=3).n_steps == 433                    # 3 days at 10 min (report)
    assert CFG.n_steps == 10 * 24 * 6 + 1                                   # default: 10 days, 10 min


def test_scanner_max_ranges_and_footprints():
    # l_max = GSD_max p / (2 tan(psi/2)) with GSD_max = 3 m
    assert [round(CFG.max_range(t), 1) for t in (1, 2)] == [58.6, 29.2]
    # footprint width at 25 km: 2 l tan(psi/2)
    assert [round(2 * 25 * np.tan(CFG.fov(t) / 2), 1) for t in (1, 2)] == [2.6, 5.3]
    # a 2048 x 2048 x 12-bit frame is 6.3 MB: 503 s at 100 kbps, ~15 kJ at 30 W
    assert CFG.t_downlink == pytest.approx(503.3, abs=0.1) and CFG.e_tx_j == pytest.approx(15100, rel=1e-3)
    assert CFG.dt >= CFG.t_downlink                                         # one frame per step fits


def test_circular_orbit_radius_constant_and_ellipse_extremes():
    t = np.linspace(0, 2e5, 500)
    r = propagate(OrbitalSolution([OrbitalSpacecraft(40, 0, 50, 30, 10, 0)]), t, CFG.mu)[0]
    assert np.allclose(np.linalg.norm(r, axis=1), 40)
    r = propagate(OrbitalSolution([OrbitalSpacecraft(40, 0.4, 70, 30, 10, 0)]), t, CFG.mu)[0]
    rn = np.linalg.norm(r, axis=1)
    assert rn.min() == pytest.approx(24, abs=0.1) and rn.max() == pytest.approx(56, abs=0.1)


def test_energy_and_angular_momentum_conserved():
    """Two-body motion: vis-viva energy and h = r x v are constant along the ellipse."""
    s = OrbitalSpacecraft(45, 0.3, 40, 70, 25, 10)
    t = np.linspace(0, 3e5, 3001)
    r = propagate(OrbitalSolution([s]), t, CFG.mu)[0]
    v = np.gradient(r, t, axis=0)
    rn, vn = np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1)
    energy = 0.5 * vn ** 2 - CFG.mu / rn
    h = np.cross(r, v)
    inner = slice(5, -5)
    assert np.allclose(energy[inner], -CFG.mu / (2 * s.a), rtol=1e-3)
    assert np.allclose(np.linalg.norm(h[inner], axis=1), np.sqrt(CFG.mu * s.a * (1 - s.e ** 2)), rtol=1e-3)
    assert np.allclose(vn[inner], [s.speed(x, CFG.mu) for x in rn[inner]], rtol=1e-3)


def test_inclination_and_body_frame():
    t = np.linspace(0, 1e5, 200)
    r = propagate(OrbitalSolution([OrbitalSpacecraft(40, 0, 0, 0, 0, 0)]), t, CFG.mu)[0]
    assert np.allclose(r[:, 2], 0)                                         # equatorial
    h = np.cross(r[0], r[5])
    assert h[2] > 0                                                        # prograde
    rb = to_body(r, t, CFG.omega_ast)
    assert np.allclose(np.linalg.norm(rb, axis=1), 40)
    # after one asteroid rotation, the body frame coincides with inertial again
    tp = np.array([CFG.p_rot_h * 3600])
    assert np.allclose(to_body(np.array([[1.0, 2.0, 3.0]]), tp, CFG.omega_ast), [[1, 2, 3]])


def test_delta_v():
    """No manoeuvre to stay in the arrival orbit; a pure plane change costs 2 v0 sin(i/2)."""
    v0 = CFG.v0
    assert v0 * 1000 == pytest.approx(2.99, abs=0.01)
    assert delta_v(OrbitalSpacecraft(CFG.r0, 0, 0), CFG) == pytest.approx(0, abs=1e-12)
    assert delta_v(OrbitalSpacecraft(CFG.r0, 0, 90), CFG) == pytest.approx(2 * v0 * np.sin(np.pi / 4))
    # Hohmann from 50 km down to a 30 km circle: two burns of the textbook transfer
    mu, r1, r2 = CFG.mu, CFG.r0, 30.0
    at = (r1 + r2) / 2
    hoh = abs(np.sqrt(mu / r1) - np.sqrt(mu * (2 / r1 - 1 / at))) + abs(np.sqrt(mu / r2) - np.sqrt(mu * (2 / r2 - 1 / at)))
    assert delta_v(OrbitalSpacecraft(r2, 0, 0), CFG) == pytest.approx(abs(hoh))
    # every orbit in the search box stays below the normalisation dv_ref = 3 v0
    worst = OrbitalSpacecraft(63.2, 0.582, 180)
    assert delta_v(worst, CFG) < CFG.dv_ref


def test_eclipse_and_battery():
    s = CFG.sun_inertial                                                  # +x at equinox
    r = np.array([[-30.0, 0, 0], [-30.0, 9.0, 0], [30.0, 0, 0]])
    assert eclipse_track(r, CFG).tolist() == [True, False, False]          # behind / beside / sunward
    K = 50
    soc = battery_track(np.zeros(K, bool), np.zeros(K, bool), CFG)
    assert np.all(soc == 1.0)                                              # 180 W > 50 W: stays full
    ecl = np.ones(K, bool)
    soc = battery_track(np.ones(K, bool), ecl, CFG)
    used_wh = (CFG.p_base_w + (CFG.e_img_j + CFG.e_tx_j) / CFG.dt) * CFG.dt / 3600
    assert soc[0] == pytest.approx(1 - used_wh / CFG.battery_wh)
    assert np.all(np.diff(soc) < 0)                                        # drains in eclipse


def _plate(z=0.0, normal_up=True):
    v = np.array([[-0.1, -0.1, z], [0.1, -0.1, z], [0, 0.1, z]], float)
    f = np.array([[0, 1, 2]] if normal_up else [[0, 2, 1]])
    return Asteroid("plate", v, f)


def test_occlusion_blocks_hidden_facet():
    """A floating plate between the S/C and a facet must hide that facet."""
    v = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0],
                  [-10, -10, 5], [10, -10, 5], [0, 20, 5]], float)     # blocker centroid on the axis
    ast = Asteroid("plates", v, np.array([[0, 1, 2], [3, 4, 5]]))
    r = np.array([[0.0, 0.0, 20.0]])
    cfg = OrbitalConfig()
    assert visibility_track(r, 2, ast, OrbitalConfig(occlusion=False))[0, 0]
    assert not visibility_track(r, 2, ast, cfg)[0, 0]
    assert visibility_track(r, 2, ast, cfg)[0, 1]          # the blocker itself is visible


def test_gsd_and_emission_limits():
    ast = _plate()
    cfg = OrbitalConfig()
    assert visibility_track(np.array([[0, 0, 29.0]]), 2, ast, cfg)[0, 0]     # wide: 29.0 km < 29.2 km
    assert not visibility_track(np.array([[0, 0, 29.5]]), 2, ast, cfg)[0, 0]  # GSD > 3 m
    assert visibility_track(np.array([[0, 0, 58.0]]), 1, ast, cfg)[0, 0]     # narrow camera reaches far
    assert not visibility_track(np.array([[0, 0, -10.0]]), 2, ast, cfg)[0, 0] # back side


def test_sunlight_condition():
    """(e) An up-facing plate is lit only while the Sun is >= 10 deg above its horizon."""
    ast = _plate()
    # Sun at sub-solar latitude 30 deg -> 30 deg elevation above the z = 0 plate at all times
    lit = lit_mask(ast, OrbitalConfig(sun_lat_deg=30.0), np.array([0.0, 3600.0]))
    assert lit.all()
    lit = lit_mask(ast, OrbitalConfig(sun_lat_deg=5.0), np.array([0.0]))       # 5 deg: too low
    assert not lit.any()
    assert lit_mask(ast, OrbitalConfig(lighting=False)) is None
    r = np.array([[0.0, 0.0, 15.0]])
    cfg = OrbitalConfig(sun_lat_deg=-30.0)                                     # Sun below the plate
    assert not visibility_track(r, 2, ast, cfg, lit_mask(ast, cfg, np.array([0.0])))[0, 0]


def test_sun_turns_with_the_asteroid():
    """A facet facing +x (body) is lit at t = 0 and dark half a rotation later."""
    v = np.array([[0, -0.1, -0.1], [0, 0.1, -0.1], [0, 0, 0.1]], float)
    ast = Asteroid("wall", v, np.array([[0, 1, 2]]))
    assert ast.normals[0, 0] > 0.99
    half = CFG.p_rot_h * 3600 / 2
    lit = lit_mask(ast, OrbitalConfig(), np.array([0.0, half]))
    assert lit[0, 0] and not lit[1, 0]


def test_coverage_curve_and_cache_match_full_evaluation():
    sc = Scenario.load("spacecov/legacy/m3/example.json")
    ast = sc.build_asteroid()
    cfg = sc.config
    cfg.dt_min = 30
    full = evaluate(sc.fleet, ast, cfg)
    curve, each = full.coverage_curve()
    assert np.all(np.diff(curve) >= -1e-12) and curve[-1] == pytest.approx(full.C)
    assert each.shape == (sc.fleet.n, cfg.n_steps) and np.all(each[:, -1] <= curve[-1] + 1e-12)
    ev = Engine(ast, cfg)
    a, b = ev.evaluate(sc.fleet), ev.evaluate(sc.fleet)
    assert ev.n_tracks == sc.fleet.n                                # second call fully cached
    for r in (a, b):
        assert r.J == pytest.approx(full.J) and r.C == pytest.approx(full.C)
        assert np.array_equal(r.first_seen, full.first_seen)
    assert a.visible is None and full.visible is not None


def test_constraints_repair_and_problem_wrapper():
    sc = Scenario.load("spacecov/legacy/m3/example.json")
    ast = sc.build_asteroid()
    cfg = sc.config
    cfg.dt_min = 30
    prob = OrbitalProblem(ast, sc.fleet.n, cfg)
    res = prob.evaluate(sc.fleet.to_vector())
    assert res.feasible and res.J == pytest.approx(cfg.weights[0] - res.S)
    assert 0.0 <= res.J <= 1.0
    assert 0 <= res.C <= 1 and 0 <= res.Z <= 1 and 0 <= res.E <= 1
    assert res.r_body.shape == (3, cfg.n_steps, 3) and res.soc.shape == (3, cfg.n_steps)
    # periapsis inside r_min -> g1 > 0 and a quadratic penalty
    bad = OrbitalSolution([OrbitalSpacecraft(20, 0, 10, type=1)])
    rb = evaluate(bad, ast, cfg)
    assert rb.g[0] > 0 and not rb.feasible
    assert rb.J == pytest.approx(cfg.weights[0] - rb.S + cfg.rho * np.sum(np.maximum(0, rb.g) ** 2))
    # repair maps anything into the box with g1, g2 <= 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        X = prob.repair(rng.normal(0, 300, prob.dim))
        lo, hi = prob.bounds()
        assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)
        sol = prob.decode(X)
        assert all(s.periapsis >= prob.r_min - 1e-9 and s.apoapsis <= cfg.r_max + 1e-9 for s in sol.spacecraft)
        assert sol.n_active >= 1                                           # never an empty fleet
    obj, g = prob.multiobjective(prob.random_solution(rng))
    assert obj.shape == (3,) and g.shape == (4,) and g[0] <= 1e-12 and g[1] <= 1e-12


def test_slots_cost_and_unused_spacecraft():
    """Unused slots (type 0) are not simulated and cost nothing; Z, E grow with the fleet."""
    ast = make_asteroid({"source": "eros", "faces": 1500})
    cfg = OrbitalConfig(duration_days=1.0, n_max=5)
    prob = OrbitalProblem(ast, 5, cfg)
    one = OrbitalSolution([OrbitalSpacecraft(40, 0, 60, type=1)] + [OrbitalSpacecraft(40, type=0)] * 4)
    two = OrbitalSolution([OrbitalSpacecraft(40, 0, 60, type=1), OrbitalSpacecraft(30, 0, 120, 90, type=2)]
                          + [OrbitalSpacecraft(40, type=0)] * 3)
    r1, r2 = prob.evaluate(one.to_vector()), prob.evaluate(two.to_vector())
    assert (r1.n_active, r2.n_active) == (1, 2)
    assert r1.Z == pytest.approx(0.1) and r2.Z == pytest.approx((1.2 + 1.3) / (10 * 1.2))    # 0.1 per 6-deg spacecraft
    # normalised by the fixed reference size, not by n_max: same fleet, same cost
    other = OrbitalProblem(ast, 7, cfg).evaluate(np.concatenate([two.to_vector(), np.tile([40, 0, 0, 0, 0, 0, 0], 2)]))
    assert other.Z == pytest.approx(r2.Z) and other.E == pytest.approx(r2.E)
    assert r2.E > r1.E and r2.C >= r1.C
    X = prob.repair(np.tile([40, 0, 0, 0, 0, 0, 0.2], 5))                  # all slots unused ...
    assert prob.decode(X).n_active == 1                                    # ... -> one is activated


def test_cast_shadows():
    """A plate between the Sun and a facet shadows it (sunlight test (e) with shadows)."""
    v = np.array([[0, -0.1, -0.1], [0, 0.1, -0.1], [0, 0, 0.1],           # wall facing +x (the Sun)
                  [3, -5, -5], [3, 5, -5], [3, 0, 10]], float)              # screen at x = 3 facing +x
    ast = Asteroid("shade", v, np.array([[0, 1, 2], [3, 4, 5]]))
    t0 = np.array([0.0])
    assert lit_mask(ast, OrbitalConfig(shadows=False), t0)[0].tolist() == [True, True]
    assert lit_mask(ast, OrbitalConfig(shadows=True), t0)[0].tolist() == [False, True]

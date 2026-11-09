"""Viewer logic without Qt: off-screen PyVista plotter + Presenter."""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from spacecov.config import SPACECRAFT_SPAN_M  # noqa: E402
from spacecov.scenario import Scenario  # noqa: E402
from viz import colors  # noqa: E402
from viz.presenter import Presenter  # noqa: E402
from viz.scene import SceneManager  # noqa: E402


@pytest.fixture
def presenter():
    pl = pv.Plotter(off_screen=True, window_size=(320, 240))
    yield Presenter(SceneManager(pl))
    pl.close()


def test_scene_time_frames_and_true_scale(presenter):
    sc = Scenario.load("scenarios/orbital_example.json")
    sc.config.dt_min = 30
    sc.config.t_max_days = 3.0
    presenter.load(sc)
    assert presenter.coverage_now() is None                 # not evaluated yet
    presenter.evaluate()
    presenter.set_time(0.0)
    c0 = presenter.coverage_now()
    presenter.set_time(presenter.duration)
    c1 = presenter.coverage_now()
    assert c0 <= c1 == pytest.approx(presenter.result.C)
    assert presenter.duration <= presenter.design.T and presenter.evaluation.feasible
    for frame in ("inertial", "body"):
        presenter.frame = frame
        presenter.refresh(full=True)
        r = np.linalg.norm(presenter.poses[0].position)
        assert r == pytest.approx(presenter.design.orbits[0].a)          # circular orbit
        for p, s in zip(presenter.poses, presenter.fleet.spacecraft):   # cone rays end on the surface
            reach = np.linalg.norm(p.rim - p.position, axis=1)           # or at the usable range
            assert np.all(reach <= presenter.cfg.max_range(s.type) + 1e-6) and np.all(reach > 0)
    # the spacecraft model is drawn at its real size (km units)
    bounds = presenter.scene._sc[0].body[3].GetMapper().GetInput().GetBounds()
    assert bounds[1] - bounds[0] == pytest.approx(SPACECRAFT_SPAN_M / 1000, rel=0.01)
    for mode in colors.COLOR_MODES:
        presenter.set_color_mode(mode)
    html = presenter.metrics_html()
    assert "Fitness J" in html and "g3 separation" in html
    presenter.set_mission_time(1.0)                               # shorter mission -> fewer steps shown
    presenter.evaluate()
    assert presenter.n_steps == sc.config.steps_within(86400.0) and presenter.result.T == 86400.0
    presenter.scene.p.screenshot(None, return_img=True)

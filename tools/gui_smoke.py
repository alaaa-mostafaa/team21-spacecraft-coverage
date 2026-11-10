"""Smoke test of the GUI: drives MainWindow through its main features, including a short optimiser run.

    python tools/gui_smoke.py [outdir]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402

from spacecov.scenario import Scenario  # noqa: E402
from viz.app import MainWindow  # noqa: E402
from viz.panels import DARK_QSS  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
out.mkdir(exist_ok=True)
cache = Path(__file__).resolve().parents[1] / ".work" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(DARK_QSS)
win = MainWindow(Scenario.load("scenarios/orbital_example.json"))
win.show()


def pump(sec=0.3):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def key(k, mod=Qt.NoModifier):
    for t in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease):
        app.sendEvent(win.plotter, QtGui.QKeyEvent(t, k, mod))
    pump(0.05)


def shot(name):
    pump(0.2)
    win.grab().save(str(out / f"gui_{name}_window.png"))       # Qt panels + charts (GL area may be blank)
    win.plotter.screenshot(str(out / f"gui_{name}_3d.png"))    # the 3D view itself
    print("shot", name, flush=True)


def wait_eval(timeout=60):
    t0 = time.time()
    while (win.evaluator.busy() or win.presenter.stale) and time.time() - t0 < timeout:
        pump(0.1)
    assert not win.presenter.stale, "mission evaluation did not finish"


pump(0.5)
wait_eval()
print("example:", win.presenter.result.summary(), flush=True)
win.timeline.set_time(6 * 3600)
shot("inertial")

key(Qt.Key_B)                       # asteroid-fixed frame
assert win.presenter.frame == "body"
shot("body")
key(Qt.Key_B)

key(Qt.Key_2)                       # ride along SC2
pump(0.2)
assert win.camera.mode == "ride" and win.camera.target == 1
shot("ride_sc2")
key(Qt.Key_F)                       # chase selected (SC1) at true scale
assert win.camera.mode == "chase"
win.timeline.set_time(7 * 3600)
shot("chase_sc1")
key(Qt.Key_Escape)
key(Qt.Key_R)

key(Qt.Key_Space)                   # play for a moment
pump(1.0)
key(Qt.Key_Space)
assert win.presenter.t > 7 * 3600
print("t after play:", win.presenter.mission_clock(), flush=True)

win.design_panel.sliders["a"].setValue(55.0)         # edit -> re-evaluation in the worker
win.design_panel.sliders["a"].valueChanged.emit(55.0)
pump(0.1)
wait_eval()
assert abs(win.presenter.design.orbits[0].a - 55.0) < 1e-9
print("after edit:", win.presenter.result.summary(), flush=True)

win.design_panel.T.valueChanged.emit(1.5)             # mission duration T is a design variable
wait_eval()
assert win.presenter.design.T_days == 1.5 and win.presenter.result.T == 1.5 * 86400
assert win.presenter.n_steps == win.presenter.cfg.steps_within(1.5 * 86400)
print("T = 1.5 d:", win.presenter.result.summary(), flush=True)

win.edit_spacecraft(1, "a", 20.0)                     # inside r_min: feasibility rejects, still drawn
wait_eval()
assert not win.presenter.evaluation.feasible and win.presenter.result is not None
assert "INFEASIBLE" in win.presenter.metrics_html()
win.edit_spacecraft(1, "a", 45.0)
wait_eval()
assert win.presenter.evaluation.feasible

# short live optimiser run (simulated annealing) from random designs
win.set_config("n_max", 5)
wait_eval()
win.optimiser_panel.set_params("sa", {"max_evals": 300, "seed": 3})
assert win.optimiser_panel.params()["max_evals"] == 300 and win.optimiser_panel.params()["alpha"] is None
win.tabs.setCurrentIndex(1)
win.optimiser_panel.pace.setCurrentIndex(1)                  # 10 moves / s: watch the search
win.run_optimiser("sa", win.optimiser_panel.params(), False)
t0 = time.time()
while len(win.sa_data) < 20 and time.time() - t0 < 120:      # setup + T0 estimate, then moves stream in
    pump(0.2)
pump(1.0)
print("moves logged:", win.optimiser_panel.log.blockCount(), "| first:",
      win.optimiser_panel.log.toPlainText().splitlines()[0][:110], flush=True)
assert win.optimiser_panel.log.blockCount() > 5, "move log is empty"
shot("search_live")
win.optimiser_panel.pause_btn.setChecked(True)               # pause / resume
pump(1.5)
n_paused = len(win.sa_data)
pump(1.5)
assert len(win.sa_data) == n_paused, "search did not pause"
win.optimiser_panel.pause_btn.setChecked(False)
win.optimiser_panel.pace.setCurrentIndex(0)                  # full speed to finish
t0 = time.time()
while win.sa_runner.running() and time.time() - t0 < 300:
    pump(0.2)
assert not win.sa_runner.running(), "optimiser did not finish"
assert win.sa_result is not None and len(win.sa_data) > 200
wait_eval()
print("SA best:", win.presenter.result.summary(), flush=True)
assert win.presenter.evaluation.feasible and win.presenter.design.N <= 5
win.timeline.set_time(12 * 3600)
shot("after_sa")

# short live genetic-algorithm run: verify generation-aware telemetry and the GA chart mode
win.optimiser_panel.set_params("ga", {"max_evals": 300, "seed": 4, "pop_size": 20})
assert win.optimiser_panel.params()["pop_size"] == 20
win.run_optimiser("ga", win.optimiser_panel.params(), False)
t0 = time.time()
while len(win.sa_data) < 20 and time.time() - t0 < 120:
    pump(0.2)
pump(1.0)
ga_log = win.optimiser_panel.log.toPlainText()
assert "gen " in ga_log and ("crossover" in ga_log or "mutation" in ga_log), "GA log is empty"
shot("ga_search_live")
win.optimiser_panel.pause_btn.setChecked(True)
pump(1.5)
n_paused = len(win.sa_data)
pump(1.5)
assert len(win.sa_data) == n_paused, "GA search did not pause"
win.optimiser_panel.pause_btn.setChecked(False)
win.optimiser_panel.pace.setCurrentIndex(0)
t0 = time.time()
while win.sa_runner.running() and time.time() - t0 < 300:
    pump(0.2)
assert not win.sa_runner.running(), "GA optimiser did not finish"
assert win.sa_result is not None and len(win.sa_data) >= 20
wait_eval()
assert win.presenter.evaluation.feasible and win.presenter.design.N <= 5
print("GA best:", win.presenter.result.summary(), flush=True)
shot("after_ga")

for _ in range(3):
    key(Qt.Key_M)
key(Qt.Key_H)
shot("help")
key(Qt.Key_H)

for k in (Qt.Key_W, Qt.Key_A):      # WASD fly does not crash
    app.sendEvent(win.plotter, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, k, Qt.NoModifier))
    pump(0.2)
    app.sendEvent(win.plotter, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, k, Qt.NoModifier))

win.set_body("kleopatra")                            # switch the target body
wait_eval()
pr = win.presenter
assert pr.cfg.body == "kleopatra" and pr.asteroid.name == "216 Kleopatra" and pr.cfg.r_max == 650.0
assert all(o.a >= pr.cfg.resolve_r_min(pr.asteroid.r_max) for o in pr.design.orbits)
print("Kleopatra:", pr.result.summary(), flush=True)
win.timeline.set_time(12 * 3600)
shot("kleopatra")
win.set_body("eros")
wait_eval()
assert win.presenter.cfg.body == "eros" and win.presenter.cfg.r_max == 100.0

win.set_config("t_max_days", 1.0)                    # mission settings: T_max = 1 day
wait_eval()
assert win.presenter.cfg.n_steps == 145
win.close()
print("GUI smoke test OK", flush=True)

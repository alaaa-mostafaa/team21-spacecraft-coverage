"""Screenshot of the viewer while SA runs (Figure "gui_sa" of the report).

    python report/LaTeX/capture_gui.py

Only the application window is captured (Qt grab of the window + the 3D view rendered by VTK,
pasted at the position of the 3D widget); nothing else on the screen.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from spacecov.scenario import Scenario  # noqa: E402
from viz.app import MainWindow  # noqa: E402
from viz.panels import DARK_QSS  # noqa: E402

app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(DARK_QSS)
win = MainWindow(Scenario.load(str(ROOT / "scenarios" / "orbital_example.json")))
win.resize(1680, 1000)
win.show()


def pump(sec=0.3):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_eval(timeout=120):
    t0 = time.time()
    while (win.evaluator.busy() or win.presenter.stale) and time.time() - t0 < timeout:
        pump(0.1)


def shot(path):
    pump(0.3)
    img = win.grab().toImage()
    win.plotter.screenshot(str(path.with_suffix(".3d.png")))
    view = QtGui.QImage(str(path.with_suffix(".3d.png")))
    pos = win.plotter.mapTo(win, QtCore.QPoint(0, 0))
    view = view.scaled(win.plotter.width(), win.plotter.height())
    painter = QtGui.QPainter(img)
    painter.drawImage(pos, view)
    painter.end()
    img.save(str(path))
    path.with_suffix(".3d.png").unlink()


pump(0.5)
win.set_config("t_max_days", 2.0)            # case E setting: Eros, T_max = 2 days, N_max = 5
win.set_config("n_max", 5)
wait_eval()
win.optimiser_panel.set_params("sa", {"max_evals": 400, "seed": 2})
win.tabs.setCurrentIndex(1)
win.optimiser_panel.pace.setCurrentIndex(1)  # 10 moves / s: watch the search
win.run_optimiser("sa", win.optimiser_panel.params(), False)
t0 = time.time()
while len(win.sa_data) < 120 and time.time() - t0 < 600:
    pump(0.2)
win.optimiser_panel.pause_btn.setChecked(True)
pump(1.0)
shot(HERE / "Figures" / "gui_sa.png")
win.optimiser_panel.pause_btn.setChecked(False)
win.optimiser_panel.pace.setCurrentIndex(0)
t0 = time.time()
while win.sa_runner.running() and time.time() - t0 < 600:
    pump(0.2)
wait_eval()
win.timeline.set_time(win.presenter.design.T)
pump(0.5)
shot(HERE / "Figures" / "gui_after_sa.png")
print("SA best:", win.presenter.result.summary(), flush=True)
win.close()

"""Main window: true-scale 3D view, coverage-vs-time chart, live optimiser charts, control panels."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor

from spacecov.design import DesignSpace
from spacecov.physics import orbit_curve
from spacecov.scenario import Scenario

from . import colors
from .camera import HELP_TEXT, CameraController
from .panels import DARK_QSS, DesignPanel, MetricsPanel, MissionPanel, OptimiserPanel, ViewPanel
from .plots import PlotWidget
from .presenter import Presenter
from .scene import SceneManager
from .timeline import TimelineBar
from .worker import STEP_FIELDS, OptimiserRunner, OrbitalEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPO_ROOT / "scenarios" / "orbital_example.json"
COVER_COLOR = "#5fd38a"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scenario: Scenario):
        super().__init__()
        self.setWindowTitle("Cooperative spacecraft coverage of 433 Eros — Team 21")
        self.resize(1680, 1000)
        self.scenario_path: Path | None = None
        self.selected = 0
        self.sa_data = np.zeros((0, len(STEP_FIELDS)))
        self.sa_result = None
        self.reference_curve = None          # coverage curve of the design the SA started from
        self._plots_dirty = False

        # ---- 3D view + charts ------------------------------------------------------------
        self.plotter = QtInteractor(self, multi_samples=8, auto_update=False)
        self.scene = SceneManager(self.plotter)
        self.presenter = Presenter(self.scene)
        self.camera = CameraController(self.plotter, self.plotter, lambda: self.scene.scene_radius)
        self.camera.on_mode_change = lambda txt: self.statusBar().showMessage(txt, 3000)
        self.camera.request_follow = lambda: (self.camera.follow(self.presenter.poses), self.render())

        self.cov_plot = PlotWidget("Coverage vs mission time", "mission time [h]", "% of the sunlit area")
        self.cov_plot.y_range = (0.0, 100.0)
        self.cov_plot.yfmt = lambda v: f"{v:.0f}"
        self.j_plot = PlotWidget("Search: fitness J  (lower is better)", "evaluation", "J")
        self.c_plot = PlotWidget("Every design tried: coverage vs mission time", "mission time T [days]",
                                 "coverage C [%]")
        self.t_plot = PlotWidget("Temperature T", "evaluation", "T", log_y=True)
        self.cov_plot.legend_loc = "br"
        self.j_plot.legend_loc = "tr"
        self.c_plot.legend_loc = "br"
        charts = QtWidgets.QSplitter(Qt.Horizontal)
        for w, stretch in ((self.cov_plot, 3), (self.j_plot, 2), (self.c_plot, 2), (self.t_plot, 2)):
            charts.addWidget(w)
            charts.setStretchFactor(charts.count() - 1, stretch)
        central = QtWidgets.QSplitter(Qt.Vertical)
        central.addWidget(self.plotter)
        central.addWidget(charts)
        central.setStretchFactor(0, 3)
        central.setStretchFactor(1, 1)
        central.setSizes([700, 280])
        self.setCentralWidget(central)

        self.evaluator = OrbitalEvaluator(self)
        self.sa_runner = OptimiserRunner(self)

        # ---- side panel, timeline, menus ---------------------------------------------------
        self._build_dock()
        self.timeline = TimelineBar(self)
        self.addToolBar(Qt.BottomToolBarArea, self.timeline)
        self._build_menu()
        self.statusBar().showMessage("Ready — press H in the 3D view for controls")
        self._connect()
        self._bind_keys()
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.setInterval(100)
        self._plot_timer.timeout.connect(self._redraw_sa_plots)
        self._plot_timer.start()
        self._move_timer = QtCore.QTimer(self)
        self._move_timer.setSingleShot(True)
        self._move_timer.setInterval(80)
        self._move_timer.timeout.connect(self._draw_pending_move)

        self.load_scenario(scenario)

    # =============================================================================== UI build
    def _build_dock(self):
        dock = QtWidgets.QDockWidget("Mission control", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        root = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(root)
        self.metrics = MetricsPanel()
        lay.addWidget(self.metrics)
        self.tabs = QtWidgets.QTabWidget()
        self.mission_panel = MissionPanel()
        self.optimiser_panel = OptimiserPanel()
        self.design_panel = DesignPanel()
        self.view_panel = ViewPanel(self.scene.visible)
        for w, name in ((self.mission_panel, "Mission"), (self.optimiser_panel, "Optimiser"),
                        (self.design_panel, "Design"), (self.view_panel, "View")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(w)
            self.tabs.addTab(scroll, name)
        lay.addWidget(self.tabs, 1)
        dock.setWidget(root)
        dock.setMinimumWidth(410)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in (("&Open scenario…", "Ctrl+O", self.open_scenario),
                              ("&Save scenario", "Ctrl+S", self.save_scenario),
                              ("Save scenario &as…", "Ctrl+Shift+S", lambda: self.save_scenario(ask=True)),
                              ("Export search &history (CSV)…", "Ctrl+E", self.export_history),
                              ("Save screens&hot…", "Ctrl+P", self.screenshot),
                              ("&Quit", "Ctrl+Q", self.close)):
            a = m.addAction(text)
            a.setShortcut(key)
            a.triggered.connect(fn)
        v = self.menuBar().addMenu("&View")
        v.addAction("Reset camera  (R)").triggered.connect(self.reset_camera)
        v.addAction("Controls help  (H)").triggered.connect(self.toggle_help)

    def _connect(self):
        mp = self.mission_panel
        mp.configChanged.connect(self.set_config)
        mp.bodyRequested.connect(self.set_body)
        mp.shapeRequested.connect(self.set_shape)

        op = self.optimiser_panel
        op.runRequested.connect(self.run_optimiser)
        op.stopRequested.connect(self.sa_runner.stop)
        op.paceChanged.connect(self.sa_runner.set_pace)
        op.pauseToggled.connect(self.sa_runner.set_paused)

        dp = self.design_panel
        dp.selectionChanged.connect(self.select_spacecraft)
        dp.spacecraftEdited.connect(self.edit_spacecraft)
        dp.missionTimeChanged.connect(self.set_mission_time)
        dp.addRequested.connect(self.add_spacecraft)
        dp.removeRequested.connect(self.remove_spacecraft)

        vp = self.view_panel
        vp.colorModeChanged.connect(self.set_color_mode)
        vp.groupToggled.connect(self.set_group)
        vp.highlightToggled.connect(self.set_highlight)
        vp.edgesToggled.connect(lambda on: (self.scene.set_edges(on), self.render()))
        vp.frameChanged.connect(self.set_frame)
        vp.markerSizeChanged.connect(lambda s: (self.scene.set_marker_size(s), self.render()))

        self.timeline.timeChanged.connect(self.on_time)
        self.cov_plot.clicked.connect(lambda h: self.timeline.set_time(h * 3600))
        self.evaluator.started.connect(self._eval_started)
        self.evaluator.progress.connect(lambda f: self.metrics.progress.setValue(int(100 * f)))
        self.evaluator.finished.connect(self._eval_finished)
        self.evaluator.failed.connect(self._failed("Mission evaluation failed"))

        self.sa_runner.steps.connect(self._sa_steps)
        self.sa_runner.move.connect(self._sa_move)
        self.sa_runner.finished.connect(self._sa_finished)
        self.sa_runner.failed.connect(self._sa_failed)

    def _bind_keys(self):
        c, tl = self.camera, self.timeline
        c.bind(Qt.Key_Space, tl.toggle_play)
        c.bind(Qt.Key_Comma, lambda: tl.step(-1))
        c.bind(Qt.Key_Period, lambda: tl.step(1))
        c.bind(Qt.Key_BracketLeft, lambda: tl.faster(-1))
        c.bind(Qt.Key_BracketRight, lambda: tl.faster(1))
        c.bind(Qt.Key_R, self.reset_camera)
        c.bind(Qt.Key_F, lambda: c.chase(self.selected))
        c.bind(Qt.Key_H, self.toggle_help)
        c.bind(Qt.Key_P, self.screenshot)
        c.bind(Qt.Key_M, self.cycle_color_mode)
        c.bind(Qt.Key_B, lambda: self.set_frame("body" if self.presenter.frame == "inertial" else "inertial",
                                                 sync=True))
        for key, group in ((Qt.Key_C, "cones"), (Qt.Key_O, "orbits"), (Qt.Key_L, "labels"), (Qt.Key_T, "trails")):
            c.bind(key, lambda g=group: self.set_group(g, not self.scene.visible[g], sync=True))

    # =============================================================================== scenario
    def load_scenario(self, sc: Scenario):
        self.selected = 0
        self.scene.selected = 0
        self.camera.set_mode(self.camera.FREE)
        self.presenter.load(sc)
        self.optimiser_panel.set_params(sc.algorithm, sc.algorithm_params)
        self.clear_sa()
        self._sync_panels()
        self._request_evaluation()
        self.scene.reset_camera()
        self.update_view()

    def open_scenario(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open scenario", str(REPO_ROOT / "scenarios"),
                                                        "Scenario (*.json)")
        if path:
            try:
                sc = Scenario.load(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Cannot load scenario", str(e))
                return
            self.scenario_path = Path(path)
            self.load_scenario(sc)

    def save_scenario(self, ask: bool = False):
        path = self.scenario_path
        if ask or path is None:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save scenario", str(REPO_ROOT / "scenarios"),
                                                         "Scenario (*.json)")
            if not p:
                return
            path = Path(p if p.endswith(".json") else p + ".json")
        self.presenter.scenario.algorithm = self.optimiser_panel.algorithm()
        self.presenter.scenario.algorithm_params = self.optimiser_panel.params()
        self.presenter.scenario.save(path)
        self.scenario_path = path
        self.statusBar().showMessage(f"Saved {path}", 4000)

    def export_history(self):
        if len(self.sa_data) == 0:
            self.statusBar().showMessage("no optimisation run to export yet", 3000)
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export search history", str(REPO_ROOT / "results"),
                                                     "CSV (*.csv)")
        if p:
            p = p if p.endswith(".csv") else p + ".csv"
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(STEP_FIELDS)
                w.writerows(self.sa_data.tolist())
            self.statusBar().showMessage(f"Saved {p}", 4000)

    def screenshot(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", str(REPO_ROOT / "screenshots"),
                                                     "PNG (*.png)")
        if p:
            self.plotter.screenshot(p if p.endswith(".png") else p + ".png")
            self.statusBar().showMessage(f"Saved {p}", 4000)

    def _sync_panels(self):
        pr = self.presenter
        self.mission_panel.set_state(pr.cfg, pr.scenario.asteroid, pr.asteroid.r_max)
        self.design_panel.set_design(pr.design, self.selected, pr.cfg)
        self.view_panel.set_mode(pr.color_mode)
        self.view_panel.set_frame(pr.frame)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.timeline.set_time(pr.t, emit=False)

    # =============================================================================== design
    def select_spacecraft(self, i: int):
        self.selected = i
        self.scene.selected = i
        self.design_panel.show_spacecraft(self.presenter.design.orbits[i])
        self.update_view()

    def edit_spacecraft(self, i: int, field: str, value: float):
        if i < 0:
            return
        self.presenter.update_spacecraft(i, **{field: value})
        self.design_panel.update_row_text(self.presenter.design, i)
        self._request_evaluation()

    def set_mission_time(self, T_days: float):
        pr = self.presenter
        pr.set_mission_time(T_days)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.timeline.set_time(pr.t, emit=False)
        self._request_evaluation()

    def add_spacecraft(self):
        n = self.presenter.design.N
        self.selected = self.presenter.add_spacecraft()
        if self.presenter.design.N == n:
            self.statusBar().showMessage(f"N_max = {self.presenter.cfg.n_max}: raise it in the Mission tab "
                                         "to add more spacecraft", 4000)
        self.scene.selected = self.selected
        self.design_panel.set_design(self.presenter.design, self.selected)
        self._request_evaluation()

    def remove_spacecraft(self, i: int):
        self.presenter.remove_spacecraft(max(i, 0))
        self.selected = min(self.selected, self.presenter.design.N - 1)
        self.scene.selected = self.selected
        self.design_panel.set_design(self.presenter.design, self.selected)
        self._request_evaluation()

    def _request_evaluation(self):
        pr = self.presenter
        if pr.stale:
            self.evaluator.request(pr.design, pr.asteroid, pr.cfg)
        self.update_view()

    # =============================================================================== mission settings
    def set_config(self, name: str, value):
        setattr(self.presenter.cfg, name, value)
        self.presenter.config_changed()
        cfg = self.presenter.cfg
        self.timeline.set_mission(cfg.dt, self.presenter.n_steps)
        self.mission_panel.set_state(cfg, self.presenter.scenario.asteroid, self.presenter.asteroid.r_max)
        self.design_panel.set_design(self.presenter.design, self.selected, cfg)
        self.clear_sa()
        self._request_evaluation()

    def set_body(self, name: str, spec: dict | None = None):
        if self.sa_runner.running():
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.presenter.set_body(name, spec)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot switch body", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.selected = 0
        self.scene.selected = 0
        self.clear_sa()
        self._sync_panels()
        self._request_evaluation()
        self.statusBar().showMessage(f"target body: {name}", 4000)

    def set_shape(self, spec: dict):
        src = spec.get("source")
        if src in ("eros", "kleopatra") and src != self.presenter.cfg.body:
            self.set_body(src, spec)                  # another body: its constants change too
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.presenter.set_asteroid_spec(spec, render=False)
            self.presenter.refresh(full=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot build asteroid", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.mission_panel.set_state(self.presenter.cfg, self.presenter.scenario.asteroid,
                                     self.presenter.asteroid.r_max)
        self.clear_sa()
        self._request_evaluation()
        self.scene.reset_camera()
        self.render()

    # =============================================================================== optimiser
    def clear_sa(self):
        self.sa_data = np.zeros((0, len(STEP_FIELDS)))
        self.sa_result = None
        self.reference_curve = None
        for p in (self.j_plot, self.c_plot, self.t_plot):
            p.clear()
            p.update()
        self.optimiser_panel.status.setText("")
        self._plots_dirty = False

    def run_optimiser(self, name: str, params: dict, from_current: bool):
        if self.sa_runner.running():
            return
        pr = self.presenter
        self.clear_sa()
        if pr.result is not None and not pr.stale:
            self.reference_curve = (pr.result.times / 3600.0, pr.result.coverage_curve()[0])
        x0 = None
        if from_current:                                   # current design, unused slots mid-range
            if pr.design.N > pr.cfg.n_max:
                QtWidgets.QMessageBox.warning(self, "Cannot start from this design",
                                              f"It has {pr.design.N} spacecraft but N_max = {pr.cfg.n_max}.")
                return
            x0 = DesignSpace.from_config(pr.cfg, pr.asteroid.r_max).encode(pr.design)
        self.sa_params = params
        self.optimiser_panel.set_running(True)
        for i in (0, 2):                                   # no hand edits while the search runs
            self.tabs.widget(i).setEnabled(False)
        self.optimiser_panel.status.setText("starting: sampling the start designs …")
        self.optimiser_panel.log.clear()
        # while searching, the 3D view shows the design SA is currently on, with the coverage it
        # reaches by the end of the mission (live footprints are switched off to keep it fast)
        self._highlight_before = pr.highlight_current
        pr.highlight_current = False
        self.timeline.set_time(pr.duration)
        self.sa_runner.set_pace(float(self.optimiser_panel.pace.currentData()))
        self.sa_runner.start(pr.asteroid, pr.cfg, name, params, x0)

    def _sa_steps(self, rows: np.ndarray, descriptions: list):
        self.sa_data = np.vstack([self.sa_data, rows])
        self._plots_dirty = True
        f = {k: i for i, k in enumerate(STEP_FIELDS)}
        lines = []
        for r, d in zip(rows, descriptions):
            verdict = "ACCEPT" if r[f["accepted"]] else "reject"
            lines.append(f"#{int(r[f['iteration']]):5d}  T={r[f['temperature']]:.1e}  dJ={r[f['dJ']]:+.4f}  "
                         f"p={r[f['p_accept']]:4.2f}  {verdict}  {d}")
        self.optimiser_panel.log.appendPlainText("\n".join(lines))

    def _sa_move(self, mv):
        """Keep only the newest move; it is drawn by a timer, so a slow redraw never builds a backlog."""
        self._pending_move = mv
        if not self._move_timer.isActive():
            self._move_timer.start()

    def _draw_pending_move(self):
        """Show the design SA is on now, and the orbit it just proposed (green accepted / red rejected)."""
        mv, self._pending_move = getattr(self, "_pending_move", None), None
        if mv is None or not self.sa_runner.running():
            return
        pr = self.presenter
        space = DesignSpace.from_config(pr.cfg, pr.asteroid.r_max)
        design, proposal = space.decode(mv.x), space.decode(mv.candidate)
        cand = None
        if 0 <= mv.slot < proposal.N:                            # the orbit the move proposed
            cand = proposal.orbits[mv.slot].spacecraft(pr.cfg.camera)
        verdict = "ACCEPTED" if mv.accepted else "rejected"
        color = (0.3, 1.0, 0.53) if mv.accepted else (1.0, 0.35, 0.35)
        pr.banner_override = (f"move #{mv.iteration}   T = {mv.temperature:.1e}   {mv.description}   "
                              f"dJ = {mv.dJ:+.4f}   P(accept) = {100 * mv.p_accept:.0f} %   -> {verdict}", color)
        pr.set_design(design, mv.evaluation)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.timeline.set_time(pr.duration, emit=False)
        curve = orbit_curve(cand) if cand is not None and pr.frame == "inertial" else None
        self.scene.set_candidate(curve, mv.accepted)
        self.update_view()

    def _end_search_view(self):
        self.presenter.banner_override = None
        self.presenter.highlight_current = getattr(self, "_highlight_before", True)
        self.scene.set_candidate(None)

    def _sa_finished(self, res, pipe):
        self.sa_result = res
        self._end_search_view()
        self.optimiser_panel.set_running(False)
        for i in (0, 2):
            self.tabs.widget(i).setEnabled(True)
        pr = self.presenter
        pr.set_design(pipe.space.decode(res.x_best), res.best)
        self.selected = min(self.selected, pr.design.N - 1)
        self.design_panel.set_design(pr.design, self.selected, pr.cfg)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        pr.set_result(None)                                  # re-evaluate for the display
        self._request_evaluation()
        col = STEP_FIELDS.index("accepted")
        acc = float(np.nanmean(self.sa_data[:, col])) if len(self.sa_data) else float("nan")
        ex = res.extra
        sched = (f"T0 = {ex['T0']:.3g}, α = {ex['alpha']:.3f}, L = {ex['chain_length']}, accepted {100 * acc:.0f} %<br>"
                 if "T0" in ex else "")
        b = res.best
        best = (f"N = {b.design.N} spacecraft, T = {b.design.T_days:.2f} d, C = {100 * b.result.C:.1f} %"
                if b.result is not None else b.report.summary())
        self.optimiser_panel.status.setText(
            f"<b>done</b> ({res.stop_reason}) — {res.n_evals} evaluations in {res.seconds:.0f} s "
            f"({1000 * res.seconds / max(res.n_evals, 1):.0f} ms each), engine runs {pipe.engine.n_simulations}, "
            f"rejected by feasibility {pipe.n_rejected}<br>{sched}"
            f"best J = {res.cost_best:.4f}: {best}")
        self._plots_dirty = True
        self.statusBar().showMessage(f"{res.algorithm} finished — best design is shown", 6000)

    def _sa_failed(self, msg):
        self._end_search_view()
        self.optimiser_panel.set_running(False)
        for i in (0, 2):
            self.tabs.widget(i).setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Optimisation failed", msg)

    def _redraw_sa_plots(self):
        if not self._plots_dirty:
            return
        self._plots_dirty = False
        d = self.sa_data
        if len(d) == 0:
            return
        col = {k: d[:, i] for i, k in enumerate(STEP_FIELDS)}
        it = col["iteration"]
        jp = self.j_plot
        jp.set_series("candidates", it, col["cost"], "#6c7899", width=1.0, style="dots", alpha=0.35)
        jp.set_series("current", it, col["current_cost"], "#7aa2ff", width=1.2)
        jp.set_series("best", it, col["best_cost"], "#ffb347", width=2.4, style="step")
        feas = col["current_cost"][np.isfinite(col["current_cost"])]
        lo = float(np.nanmin(col["best_cost"]))
        hi = float(np.percentile(feas, 98)) if len(feas) else lo + 1.0
        hi = min(hi, 3.0 * abs(lo) + 1.0)                     # rejected designs (cost ~1000) stay off-scale
        jp.y_range = (lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
        jp.subtitle = f"best {col['best_cost'][-1]:.4f}"
        cp = self.c_plot                                      # what the designs achieve: C against T
        acc = col["accepted"] > 0
        ok = col["feasible"] > 0
        cp.set_series("rejected", col["T_days"][~acc & ok], 100 * col["C"][~acc & ok], "#8a8f9f", width=1.6,
                      style="dots", alpha=0.45)
        cp.set_series("accepted", col["T_days"][acc & ok], 100 * col["C"][acc & ok], "#7aa2ff", width=2.0,
                      style="dots", alpha=0.8)
        tail = slice(max(0, len(it) - 150), None)
        cp.set_series("current path (last 150)", col["T_cur"][tail], 100 * col["C_cur"][tail], "#ffffff",
                      width=1.2, alpha=0.7)
        cp.set_series("best", col["T_best"][-1:], 100 * col["C_best"][-1:], "#ffb347", width=6, style="dots")
        cp.subtitle = (f"best: C {100 * col['C_best'][-1]:.1f} % in T = {col['T_best'][-1]:.2f} d with "
                       f"N = {col['N_best'][-1]:.0f}   (infeasible tries: {int((~ok).sum())})")
        tp = self.t_plot
        temp = col["temperature"]
        if np.isfinite(temp).any():
            tp.set_series("T", it, temp, "#ff6b6b", width=1.8, style="step", legend=False)
            a = col["accepted"]
            w = min(len(a), 50)
            tp.subtitle = f"T = {temp[-1]:.2e}   accepted (last {w} moves) {100 * np.nanmean(a[-w:]):.0f} %"
        else:
            tp.subtitle = "this algorithm has no temperature"
        total = getattr(self, "sa_params", None)
        if self.sa_runner.running() and total is not None:
            self.optimiser_panel.status.setText(
                f"running … evaluation {int(it[-1])} / {total.get('max_evals', '?')}, best J "
                f"{col['best_cost'][-1]:.4f}: C {100 * col['C_best'][-1]:.1f} % in {col['T_best'][-1]:.2f} d "
                f"with {col['N_best'][-1]:.0f} spacecraft")
        for p in (jp, cp, tp):
            p.update()

    # =============================================================================== display
    def set_color_mode(self, mode: str):
        self.presenter.set_color_mode(mode)
        self.view_panel.set_mode(mode)
        self.render()

    def cycle_color_mode(self):
        modes = colors.COLOR_MODES
        self.set_color_mode(modes[(modes.index(self.presenter.color_mode) + 1) % len(modes)])
        self.statusBar().showMessage(f"colour mode: {self.presenter.color_mode}", 2000)

    def set_group(self, group: str, on: bool, sync: bool = False):
        self.scene.set_group_visible(group, on)
        if sync:
            self.view_panel.set_check(group, on)
        self.render()

    def set_highlight(self, on: bool):
        self.presenter.highlight_current = on
        self.presenter._color_key = None
        self.update_view()

    def set_frame(self, frame: str, sync: bool = False):
        self.presenter.frame = frame
        if frame == "body" and not self.scene.visible["tracks"]:
            self.set_group("tracks", True, sync=True)
        if sync:
            self.view_panel.set_frame(frame)
        self.update_view()
        self.statusBar().showMessage(f"{frame} frame", 2000)

    def toggle_help(self):
        self.scene.toggle_help(HELP_TEXT)
        self.render()

    def reset_camera(self):
        self.camera.set_mode(self.camera.FREE)
        self.scene.fit(max(s.apoapsis for s in self.presenter.fleet.spacecraft))
        self.scene.reset_camera()
        self.render()

    # =============================================================================== time & evaluation
    def on_time(self, t: float):
        self.presenter.set_time(t)
        self.camera.follow(self.presenter.poses)
        self._update_coverage_plot()
        self.render()

    def _eval_started(self):
        self.metrics.progress.setValue(0)
        self.metrics.progress.show()

    def _eval_finished(self, out, _rid):
        ev, res = out                                          # (Evaluation | None, EngineResult)
        self.presenter.set_result(res, ev)
        if not self.evaluator.busy():
            self.metrics.progress.hide()
        verdict = "" if ev is None or ev.feasible else "  —  INFEASIBLE: " + ev.report.summary()[12:]
        self.statusBar().showMessage(f"mission evaluated in {res.eval_seconds:.2f} s{verdict}", 4000)
        self.update_view()

    def _failed(self, title):
        def show(msg):
            self.metrics.progress.hide()
            QtWidgets.QMessageBox.critical(self, title, msg)
        return show

    def _update_coverage_plot(self):
        p = self.cov_plot
        r = self.presenter.result
        p.x_range = (0.0, max(self.presenter.duration / 3600.0, 1e-3))
        if r is None or self.presenter.stale:
            p.clear()
            p.subtitle = "evaluating …"
            p.update()
            return
        hours = r.times / 3600.0
        fleet, each = r.coverage_curve()
        p.clear()
        if self.reference_curve is not None:
            p.set_series("start design", self.reference_curve[0], 100 * self.reference_curve[1], "#8a8f9f",
                         width=1.4)
        for i, c in enumerate(each):
            p.set_series(f"SC{i + 1} alone", hours, 100 * c, colors.sc_color_hex(i), width=1.0, alpha=0.75)
        p.set_series("whole fleet", hours, 100 * fleet, COVER_COLOR, width=2.6)
        k = self.presenter.current_step()
        t_h = self.presenter.t / 3600
        p.set_cursor(t_h, f"t = {t_h:.1f} h  →  {100 * fleet[min(k, len(fleet) - 1)]:.1f} %")
        half = np.flatnonzero(fleet >= 0.5 * fleet[-1])
        p.subtitle = (f"after {hours[-1]:.1f} h: {100 * fleet[-1]:.1f} %   (half of that by {hours[half[0]]:.1f} h)"
                      if len(half) else "")
        p.update()

    def update_view(self):
        self.presenter.refresh()
        self.metrics.set_html(self.presenter.metrics_html())
        self.camera.follow(self.presenter.poses)
        self._update_coverage_plot()
        self.render()

    def render(self):
        self.plotter.render()

    def closeEvent(self, ev):
        self.sa_runner.shutdown()
        self.evaluator.shutdown()
        self.plotter.close()
        super().closeEvent(ev)


def run(scenario: Scenario | str | Path | None = None):
    if scenario is None:
        scenario = DEFAULT_SCENARIO if DEFAULT_SCENARIO.exists() else Scenario()
    if isinstance(scenario, (str, Path)):
        scenario = Scenario.load(scenario)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    win = MainWindow(scenario)
    win.show()
    return app.exec_()

"""Main window: true-scale 3D view, coverage-vs-time chart, live optimiser charts, control panels."""
from __future__ import annotations

import copy
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
from .dashboard import MetricsPanel
from .panels import DARK_QSS, DesignPanel, MissionPanel, OptimiserPanel, ViewPanel
from .plots import PlotWidget
from .presenter import Presenter
from .scene import SceneManager
from .timeline import TimelineBar
from .worker import STEP_FIELDS, OptimiserRunner, OrbitalEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPO_ROOT / "scenarios" / "orbital_example.json"
COVER_COLOR = "#5fd38a"
ALGO_NAMES = {"sa": "SA", "ga": "GA", "random": "random search"}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scenario: Scenario):
        super().__init__()
        self.setWindowTitle("Cooperative spacecraft coverage of 433 Eros — Team 21")
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1680, 1000)
        self.resize(min(1680, int(avail.width() * 0.96)), min(1000, int(avail.height() * 0.94)))
        self.scenario_path: Path | None = None
        self.selected = 0
        self.sa_data = np.zeros((0, len(STEP_FIELDS)))
        self.sa_result = None
        self.reference_curve = None          # coverage curve of the design the SA started from
        self.last_run: dict | None = None    # best of the finished run: shown as a reference in the next
        self.prev_run: dict | None = None    # the reference shown during / after the current run
        # the initial design: what the user loaded or edited, never an optimiser result.  Every run
        # starts from it, so SA and GA runs are independent of each other.
        self.base: dict | None = None
        self._base_pending = False           # the next evaluation result belongs to self.base
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
        self.cov_plot.empty_text = "evaluating the mission …"
        self.cov_plot.hover_unit = " %"
        self.j_plot = PlotWidget("Fitness J during the search", "evaluation", "J  (lower is better)")
        self.j_plot.empty_text = "Run an optimiser (Optimiser tab)\nto watch J fall here"
        self.c_plot = PlotWidget("Designs tried: coverage vs time", "mission time T [days]", "coverage C [%]")
        self.c_plot.empty_text = "Every design the search tries\nappears here as a dot"
        self.c_plot.yfmt = lambda v: f"{v:.0f}"
        self.c_plot.hover_unit = " %"
        self.t_plot = PlotWidget("Temperature T", "evaluation", "T", log_y=True)
        self.t_plot.empty_text = "The cooling schedule of\nsimulated annealing appears here"
        self.cov_plot.legend_loc = "tl"            # curves rise to the right: top-left stays empty
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
        keys = QtWidgets.QLabel("H help · drag rotate · wheel zoom · WASD fly · 1–7 ride · F chase · "
                                "Space play · M colours · B frame")
        keys.setStyleSheet("color:#6c7899; padding-right:8px;")
        self.statusBar().addPermanentWidget(keys)
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
        root = QtWidgets.QSplitter(Qt.Vertical)
        root.setChildrenCollapsible(False)
        self.dock_split = root
        self.metrics = MetricsPanel()
        top = QtWidgets.QScrollArea()
        top.setWidgetResizable(True)
        top.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        top.setWidget(self.metrics)
        top.setMinimumHeight(150)
        root.addWidget(top)
        self.tabs = QtWidgets.QTabWidget()
        self.mission_panel = MissionPanel()
        self.optimiser_panel = OptimiserPanel()
        self.design_panel = DesignPanel()
        self.view_panel = ViewPanel(self.scene.visible)
        for w, name in ((self.mission_panel, "Mission"), (self.optimiser_panel, "Optimiser"),
                        (self.design_panel, "Design"), (self.view_panel, "View")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(w)
            self.tabs.addTab(scroll, name)
        root.addWidget(self.tabs)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([self.metrics.sizeHint().height() + 40, 600])
        for cb in root.findChildren(QtWidgets.QComboBox):          # long items must not force a wide dock
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
        dock.setWidget(root)
        dock.setMinimumWidth(400)
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
        op.algo.currentIndexChanged.connect(
            lambda _: None if self.sa_runner.running() or len(self.sa_data) else
            self._set_algorithm_charts(op.algorithm()))
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
        self._set_algorithm_charts(self.optimiser_panel.algorithm())
        self._sync_panels()
        self._remember_base()
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
    def _remember_base(self):
        """The displayed design is the user's own: make it the starting point of every run."""
        self.base = dict(design=copy.deepcopy(self.presenter.design), ev=None, res=None)
        self._base_pending = True

    def _restore_base(self):
        """Put the initial design back on screen (with its stored evaluation) before a run."""
        b, pr = self.base, self.presenter
        if b is None:
            return
        pr.set_design(b["design"], b["ev"], b["res"])
        self.selected = min(self.selected, max(pr.design.N - 1, 0))
        self.scene.selected = self.selected
        self.design_panel.set_design(pr.design, self.selected, pr.cfg)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.update_view()

    def _set_source(self, text: str):
        """Say which design the dashboard, coverage chart and 3D view are showing."""
        self.metrics.setTitle(f"Displayed design — {text}")

    def select_spacecraft(self, i: int):
        self.selected = i
        self.scene.selected = i
        self.design_panel.show_spacecraft(self.presenter.design.orbits[i])
        self.update_view()

    def edit_spacecraft(self, i: int, field: str, value: float):
        if i < 0:
            return
        self._set_source("your design (edited)")
        self.presenter.update_spacecraft(i, **{field: value})
        self.design_panel.update_row_text(self.presenter.design, i)
        self._remember_base()
        self._request_evaluation()

    def set_mission_time(self, T_days: float):
        pr = self.presenter
        self._set_source("your design (edited)")
        pr.set_mission_time(T_days)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.timeline.set_time(pr.t, emit=False)
        self._remember_base()
        self._request_evaluation()

    def add_spacecraft(self):
        n = self.presenter.design.N
        self._set_source("your design (edited)")
        self.selected = self.presenter.add_spacecraft()
        if self.presenter.design.N == n:
            self.statusBar().showMessage(f"N_max = {self.presenter.cfg.n_max}: raise it in the Mission tab "
                                         "to add more spacecraft", 4000)
        self.scene.selected = self.selected
        self.design_panel.set_design(self.presenter.design, self.selected)
        self._remember_base()
        self._request_evaluation()

    def remove_spacecraft(self, i: int):
        self._set_source("your design (edited)")
        self.presenter.remove_spacecraft(max(i, 0))
        self.selected = min(self.selected, self.presenter.design.N - 1)
        self.scene.selected = self.selected
        self.design_panel.set_design(self.presenter.design, self.selected)
        self._remember_base()
        self._request_evaluation()

    def _request_evaluation(self):
        pr = self.presenter
        if pr.stale and not self.sa_runner.running() and self.sa_result is None:
            self._set_source("your design")
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
        self._remember_base()
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
        self._remember_base()
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
        self._remember_base()
        self._request_evaluation()
        self.scene.reset_camera()
        self.render()

    # =============================================================================== optimiser
    def clear_sa(self, keep_reference: bool = False):
        if not keep_reference:                   # another problem: the last run is no reference
            self.last_run = self.prev_run = None
        self.sa_data = np.zeros((0, len(STEP_FIELDS)))
        self.sa_result = None
        self.reference_curve = None
        for p in (self.j_plot, self.c_plot, self.t_plot):
            p.clear()
            p.update()
        self.optimiser_panel.status.setText("")
        self.optimiser_panel.run_progress.setVisible(False)
        self._plots_dirty = False

    def _set_algorithm_charts(self, name: str):
        """Keep the chart row relevant to the selected search strategy."""
        is_ga = name == "ga"
        self.t_plot.setVisible(name != "random")               # random search has neither
        self.t_plot.title = "GA population spread" if is_ga else "Temperature T"
        self.t_plot.xlabel = "generation" if is_ga else "evaluation"
        self.t_plot.ylabel = "std of J (feasible)" if is_ga else "T"
        self.t_plot.log_y = not is_ga
        self.t_plot.empty_text = ("waiting for the first generation" if is_ga else
                                  "The cooling schedule of\nsimulated annealing appears here")

    def run_optimiser(self, name: str, params: dict, from_current: bool):
        if self.sa_runner.running():
            return
        pr = self.presenter
        self.clear_sa(keep_reference=True)
        self.prev_run = self.last_run
        self._restore_base()                               # never continue from the previous run's result
        if pr.result is not None and not pr.stale:
            self.reference_curve = (pr.result.times / 3600.0, pr.result.coverage_curve()[0], "initial design")
        x0 = None
        if from_current:                                   # initial design, unused slots mid-range
            if pr.design.N > pr.cfg.n_max:
                QtWidgets.QMessageBox.warning(self, "Cannot start from this design",
                                              f"It has {pr.design.N} spacecraft but N_max = {pr.cfg.n_max}.")
                return
            x0 = DesignSpace.from_config(pr.cfg, pr.asteroid.r_max).encode(pr.design)
        self.sa_params = params
        self._set_algorithm_charts(name)
        self.optimiser_panel.set_running(True)
        for i in (0, 2):                                   # no hand edits while the search runs
            self.tabs.widget(i).setEnabled(False)
        label = ALGO_NAMES.get(name, name)
        self._set_source(f"initial design — {label} starts from it" if from_current else
                         f"initial design — {label} starts from random designs")
        for p_ in (self.j_plot, self.c_plot):
            p_.empty_text = f"{label} is sampling its start designs …"
        self.optimiser_panel.status.setText(
            f"starting {label}: sampling the start designs …"
            + (f"<br>reference: last {self.prev_run['name']} run best J {self.prev_run['J']:.4f}"
               if self.prev_run else ""))
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
        is_ga = self.optimiser_panel.algorithm() == "ga"
        for r, d in zip(rows, descriptions):
            if is_ga:
                lines.append(f"#{int(r[f['iteration']]):5d}  {d}  J={r[f['cost']]:.4f}  "
                             f"best={r[f['best_cost']]:.4f}")
            else:
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
        desc = mv.description if len(mv.description) <= 70 else mv.description[:67] + "..."
        if np.isfinite(mv.temperature) and np.isfinite(mv.p_accept):           # simulated annealing
            text = (f"move #{mv.iteration}   T = {mv.temperature:.1e}   {desc}   "
                    f"dJ = {mv.dJ:+.4f}   P(accept) = {100 * mv.p_accept:.0f} %   -> {verdict}")
        else:                                                                    # GA / random search
            ev = mv.evaluation
            cost = f"   J = {ev.cost:.4f}" if ev is not None and ev.feasible else "   infeasible"
            text, color = f"evaluation #{mv.iteration}   {desc}{cost}", (0.55, 0.75, 1.0)
        pr.banner_override = (text, color)
        self._set_source(f"{ALGO_NAMES.get(self.optimiser_panel.algorithm(), 'search')} live: "
                         "the design it is on now")
        pr.set_design(design, mv.evaluation)
        self.timeline.set_mission(pr.cfg.dt, pr.n_steps)
        self.timeline.set_time(pr.duration, emit=False)
        curve = orbit_curve(cand) if cand is not None and pr.frame == "inertial" else None
        self.scene.set_candidate(curve, mv.accepted)
        self.update_view()

    def _end_search_view(self):
        self.j_plot.empty_text = "Run an optimiser (Optimiser tab)\nto watch J fall here"
        self.c_plot.empty_text = "Every design the search tries\nappears here as a dot"
        self.presenter.banner_override = None
        self.presenter.highlight_current = getattr(self, "_highlight_before", True)
        self.scene.set_candidate(None)

    def _sa_finished(self, res, pipe):
        self.sa_result = res
        label = ALGO_NAMES.get(res.algorithm, res.algorithm)
        b = res.best
        self.last_run = dict(name=label, J=float(res.cost_best),
                             C=float(b.result.C) if b.result is not None else float("nan"),
                             T=float(b.design.T_days) if b.result is not None else float("nan"))
        self._set_source(f"best design of the {label} run")
        self._end_search_view()
        self.optimiser_panel.set_running(False)
        self.optimiser_panel.set_progress(res.n_evals, res.n_evals)
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
        accepted = self.sa_data[:, col]
        acc = float(np.nanmean(accepted)) if np.isfinite(accepted).any() else float("nan")
        ex = res.extra
        if res.algorithm == "ga":
            gs = ex.get("generations", {})
            feasible = gs.get("feasible", [])
            diversity = gs.get("diversity", [])
            final_feasible = 100 * feasible[-1] if len(feasible) else float("nan")
            final_diversity = diversity[-1] if len(diversity) else float("nan")
            gains = ex.get("improvements_by_operator", {})
            sched = (f"generations {ex.get('generations_run', '?')} / {ex.get('generations_planned', '?')}, "
                     f"final feasible {final_feasible:.0f} %, diversity {final_diversity:.3f}<br>"
                     f"improvements — swap {gains.get('crossover (swap)', 0)}, "
                     f"blend {gains.get('crossover (blend)', 0)}, mutation {gains.get('mutation', 0)}<br>")
        else:
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
        is_ga = self.optimiser_panel.algorithm() == "ga"
        gen = col["generation"]
        gens = np.unique(gen[np.isfinite(gen)]) if is_ga else np.array([])
        jp.set_series("candidates", it, col["cost"], "#6c7899", width=1.0, style="dots", alpha=0.35)
        if is_ga:                       # a GA has no single "current" design: show the population median
            gx, gy = [], []
            for g in gens:
                m = (gen == g) & (col["feasible"] > 0)
                if m.any():
                    gx.append(float(it[gen == g].max()))
                    gy.append(float(np.median(col["cost"][m])))
            jp.remove_series("current")
            jp.set_series("generation median", gx, gy, "#7aa2ff", width=1.6)
        else:
            jp.remove_series("generation median")
            jp.set_series("current", it, col["current_cost"], "#7aa2ff", width=1.2)
        jp.set_series("best", it, col["best_cost"], "#ffb347", width=2.4, style="step")
        ref = self.prev_run
        ref_name = f"last {ref['name']} run best" if ref else ""
        for p_ in (jp, self.c_plot):                         # a stale reference from an older run
            for n in [n for n in p_.order if n.startswith("last ") and n != ref_name]:
                p_.remove_series(n)
        if ref:
            jp.set_series(ref_name, [it[0], max(it[-1], it[0] + 1)], [ref["J"], ref["J"]], "#c8cedd", width=1.4,
                          style="dash")
        feas = col["current_cost"][np.isfinite(col["current_cost"])]
        lo = float(np.nanmin(col["best_cost"]))
        if ref:
            lo = min(lo, ref["J"])
        hi = float(np.percentile(feas, 98)) if len(feas) else lo + 1.0
        hi = min(hi, 3.0 * abs(lo) + 1.0)                     # rejected designs (cost ~1000) stay off-scale
        jp.y_range = (lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
        jp.subtitle = f"best {col['best_cost'][-1]:.4f}"
        if ref:
            d = col["best_cost"][-1] - ref["J"]
            jp.subtitle += f"   ·   last {ref['name']} run {ref['J']:.4f}  ({d:+.4f})"
        cp = self.c_plot                                      # what the designs achieve: C against T
        ok = col["feasible"] > 0
        if is_ga:                       # earlier generations recede, the newest one is highlighted
            last = (gen == gens[-1]) if len(gens) else np.zeros(len(it), bool)
            for name in ("rejected", "accepted", "current path (last 150)"):
                cp.remove_series(name)
            cp.set_series("earlier generations", col["T_days"][~last & ok], 100 * col["C"][~last & ok], "#8a8f9f",
                          width=1.6, style="dots", alpha=0.4)
            cp.set_series("latest generation", col["T_days"][last & ok], 100 * col["C"][last & ok], "#7aa2ff",
                          width=2.2, style="dots", alpha=0.9)
        else:
            acc = col["accepted"] > 0
            for name in ("earlier generations", "latest generation"):
                cp.remove_series(name)
            cp.set_series("rejected", col["T_days"][~acc & ok], 100 * col["C"][~acc & ok], "#8a8f9f", width=1.6,
                          style="dots", alpha=0.45)
            cp.set_series("accepted", col["T_days"][acc & ok], 100 * col["C"][acc & ok], "#7aa2ff", width=2.0,
                          style="dots", alpha=0.8)
            tail = slice(max(0, len(it) - 150), None)
            cp.set_series("current path (last 150)", col["T_cur"][tail], 100 * col["C_cur"][tail], "#ffffff",
                          width=1.2, alpha=0.7)
        if ref and np.isfinite(ref["C"]):
            cp.set_series(ref_name, [ref["T"]], [100 * ref["C"]], "#c8cedd", width=5, style="dots")
        cp.set_series("best", col["T_best"][-1:], 100 * col["C_best"][-1:], "#ffb347", width=6, style="dots")
        cp.subtitle = (f"best C {100 * col['C_best'][-1]:.1f} % · T {col['T_best'][-1]:.2f} d · "
                       f"N {col['N_best'][-1]:.0f} · {int((~ok).sum())} infeasible")
        tp = self.t_plot
        temp = col["temperature"]
        if self.optimiser_panel.algorithm() == "ga":
            gen = col["generation"]
            ok_gen = np.isfinite(gen)
            if ok_gen.any():
                unique = np.unique(gen[ok_gen])
                spread, feasible = [], []
                for g in unique:
                    mask = gen == g
                    ok_cost = col["cost"][mask & (col["feasible"] > 0)]       # infeasible cost ~1000
                    spread.append(float(np.std(ok_cost)) if len(ok_cost) > 1 else float("nan"))
                    feasible.append(100 * np.nanmean(col["feasible"][mask]))
                tp.set_series("fitness spread", unique, spread, "#c084fc", width=1.8, legend=False)
                tp.subtitle = (f"generation {int(unique[-1])}   feasible {feasible[-1]:.0f}%   "
                               f"spread {spread[-1]:.3f}")
            else:
                tp.subtitle = "waiting for the first generation"
        else:
            if np.isfinite(temp).any():
                tp.set_series("T", it, temp, "#ff6b6b", width=1.8, style="step", legend=False)
                a = col["accepted"]
                w = min(len(a), 50)
                tp.subtitle = f"T = {temp[-1]:.2e}   accepted (last {w} moves) {100 * np.nanmean(a[-w:]):.0f} %"
            else:
                tp.subtitle = "this algorithm has no temperature"
        total = getattr(self, "sa_params", None)
        if self.sa_runner.running() and total is not None and total.get("max_evals"):
            self.optimiser_panel.set_progress(int(it[-1]), int(total["max_evals"]))
        if self.sa_runner.running() and total is not None:
            if self.optimiser_panel.algorithm() == "ga":
                gen = col["generation"]
                generation = int(np.nanmax(gen)) if np.isfinite(gen).any() else 0
                recent = col["feasible"][-min(len(col["feasible"]), 40):]
                feasible = 100 * np.nanmean(recent)
                self.optimiser_panel.status.setText(
                    f"<b>running GA</b> · generation {generation} · evaluation {int(it[-1])} / "
                    f"{total.get('max_evals', '?')} · best J <b>{col['best_cost'][-1]:.4f}</b>")
                self.optimiser_panel.population.setText(
                    f"<b>Live population</b><br>feasible candidates: {feasible:.0f}%<br>"
                    f"best coverage: {100 * col['C_best'][-1]:.1f}% · mission: {col['T_best'][-1]:.2f} d · "
                    f"fleet: {col['N_best'][-1]:.0f} spacecraft")
            else:
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
        if self._base_pending and self.base is not None and not self.sa_runner.running():
            self.base["ev"], self.base["res"] = ev, res
            self._base_pending = False
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
            p.set_series(self.reference_curve[2], self.reference_curve[0], 100 * self.reference_curve[1],
                         "#8a8f9f", width=1.4)
        for i, c in enumerate(each):
            p.set_series(f"SC{i + 1} alone", hours, 100 * c, colors.sc_color_hex(i), width=1.0, alpha=0.75)
        p.set_series("whole fleet", hours, 100 * fleet, COVER_COLOR, width=2.6, style="area")
        k = self.presenter.current_step()
        t_h = self.presenter.t / 3600
        p.set_cursor(t_h, f"t = {t_h:.1f} h  →  {100 * fleet[min(k, len(fleet) - 1)]:.1f} %")
        half = np.flatnonzero(fleet >= 0.5 * fleet[-1])
        p.subtitle = (f"{100 * fleet[-1]:.1f} % after {hours[-1]:.1f} h   ·   half of that by {hours[half[0]]:.1f} h"
                      if len(half) else "")
        p.update()

    def update_view(self):
        self.presenter.refresh()
        self.metrics.set_data(self.presenter.metrics_data())
        self._fit_metrics_pane()
        self.camera.follow(self.presenter.poses)
        self._update_coverage_plot()
        self.render()

    def _fit_metrics_pane(self):
        """Grow the dashboard pane to show every row (up to 60 % of the dock); never shrink it."""
        sp = self.dock_split
        sizes = sp.sizes()
        need = self.metrics.sizeHint().height() + 6
        total = sum(sizes)
        if total > 0 and sizes[0] < need:
            top = min(need, int(0.6 * total))
            sp.setSizes([top, total - top])

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

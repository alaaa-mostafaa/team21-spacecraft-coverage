"""Qt side-panel widgets.  They only emit signals; app.py wires them to the Presenter."""
from __future__ import annotations

import dataclasses

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from spacecov import algorithms
from spacecov.config import MissionConfig

from . import colors

# (name, label, min, max, step, decimals, unit)
DESIGN_FIELDS = [
    ("a", "a  orbit radius (circular)", 20.0, 100.0, 0.5, 1, " km"),
    ("inc", "ι  inclination", 0.0, 180.0, 1.0, 1, "°"),
    ("raan", "Ω  where the orbit crosses the equator", 0.0, 360.0, 1.0, 1, "°"),
    ("M0", "M₀  where the spacecraft starts", 0.0, 360.0, 1.0, 1, "°"),
]
T_MAX_DAYS = [1, 2, 3, 5, 10]
TIME_STEPS_MIN = [10, 15, 20, 30]
SHAPES = [("Eros, compute on 1500 facets (fast)", {"source": "eros", "faces": 1500}),
          ("Kleopatra, radar model (4092 facets)", {"source": "kleopatra"}),
          ("Eros, compute on 3000 facets", {"source": "eros", "faces": 3000}),
          ("Eros, compute on 6000 facets (slow)", {"source": "eros", "faces": 6000}),
          ("Ellipsoid 34.4 x 11.2 x 11.2 km", {"source": "ellipsoid", "semi_axes": [17.2, 5.6, 5.6],
                                               "n_lon": 64, "n_lat": 32, "bump_amplitude": 0.0,
                                               "n_bumps": 0, "seed": 0})]


def _note(text: str) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    w.setWordWrap(True)
    w.setTextFormat(Qt.RichText)
    w.setStyleSheet("color:#9aa8c8")
    return w


def _group(title):
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g)
    lay.setContentsMargins(8, 8, 8, 8)
    return g, lay


class LabeledSlider(QtWidgets.QWidget):
    """Slider + spin box pair editing one float."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label, lo, hi, step, decimals, unit="", parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setVerticalSpacing(0)
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(lo * self._scale)), int(round(hi * self._scale)))
        self.slider.setSingleStep(max(1, int(round(step * self._scale))))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(unit)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(96)
        lay.addWidget(self.label, 0, 0, 1, 2)
        lay.addWidget(self.slider, 1, 0)
        lay.addWidget(self.spin, 1, 1)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        val = v / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self.spin.value()

    def set_range(self, lo: float, hi: float):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setRange(lo, hi)
        self.slider.setRange(int(round(lo * self._scale)), int(round(hi * self._scale)))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def setValue(self, v: float):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(int(round(v * self._scale)))
        for w in (self.slider, self.spin):
            w.blockSignals(False)


# =========================================================================== Mission
class MissionPanel(QtWidgets.QWidget):
    configChanged = pyqtSignal(str, object)
    shapeRequested = pyqtSignal(dict)
    bodyRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)

        g, gl = _group("Search space")
        form = QtWidgets.QFormLayout()
        self.body = QtWidgets.QComboBox()
        self.body.addItem("433 Eros (34 km)", "eros")
        self.body.addItem("216 Kleopatra (219 km)", "kleopatra")
        self.body.activated.connect(lambda _: self.bodyRequested.emit(self.body.currentData()))
        form.addRow("target body", self.body)
        self.n_max = QtWidgets.QSpinBox()
        self.n_max.setRange(1, 10)
        self.t_max = QtWidgets.QComboBox()
        for d in T_MAX_DAYS:
            self.t_max.addItem(f"{d} day" + ("s" if d > 1 else ""), float(d))
        self.t_min = QtWidgets.QDoubleSpinBox()
        self.t_min.setRange(0.05, 10.0)
        self.t_min.setDecimals(2)
        self.t_min.setSingleStep(0.25)
        self.t_min.setSuffix(" d")
        self.camera = QtWidgets.QComboBox()
        form.addRow("spacecraft slots N_max", self.n_max)
        form.addRow("mission time T  ≤  T_max", self.t_max)
        form.addRow("mission time T  ≥  T_min", self.t_min)
        form.addRow("camera (all spacecraft)", self.camera)
        gl.addLayout(form)
        gl.addWidget(_note("Decision vector x = [T, (a, ι, Ω, M₀, t) × N_max]: a slot flies when its switch "
                           "t = 1, so N = number of slots switched on. Orbits are circular. T_max is also the "
                           "simulation horizon: the mission is simulated up to T_max once per spacecraft and "
                           "a design uses the first T of it. Choosing a body sets its constants (GM, spin, "
                           "3 m / 20 m pixels, orbit band)."))
        self.n_max.valueChanged.connect(lambda v: self.configChanged.emit("n_max", int(v)))
        self.t_max.currentIndexChanged.connect(lambda _: self.configChanged.emit("t_max_days", self.t_max.currentData()))
        self.t_min.valueChanged.connect(lambda v: self.configChanged.emit("t_min_days", float(v)))
        self.camera.currentIndexChanged.connect(lambda _: self.configChanged.emit("camera", self.camera.currentData()))
        lay.addWidget(g)

        g, gl = _group("Fitness  J = w_C·(1−C) + w_Z·Z + w_E·E + w_T·T/T_max")
        row = QtWidgets.QHBoxLayout()
        self.w = []
        for label in ("w_C", "w_Z", "w_E", "w_T"):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(0.0, 10.0)
            box.setDecimals(3)
            box.setSingleStep(0.05)
            box.valueChanged.connect(self._weights)
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(box)
            self.w.append(box)
        gl.addLayout(row)
        gl.addWidget(_note("Minimised. Coverage C (maximised) enters as 1 − C, the share of the sunlit area "
                           "missed; Z = fleet cost (0.1 per spacecraft), E = energy (fuel + electricity), "
                           "T/T_max = mission time (all minimised). Every term is in [0, 1]. A design that "
                           "fails feasibility costs 1000 + ρ·Σ violation² and never reaches the engine."))
        lay.addWidget(g)

        g, gl = _group("Simulation")
        form = QtWidgets.QFormLayout()
        self.dt = QtWidgets.QComboBox()
        for m in TIME_STEPS_MIN:
            self.dt.addItem(f"{m} min", m)
        self.shape = QtWidgets.QComboBox()
        for label, spec in SHAPES:
            self.shape.addItem(label, spec)
        form.addRow("frame interval Δt", self.dt)
        form.addRow("objective mesh", self.shape)
        gl.addLayout(form)
        self.dt.currentIndexChanged.connect(lambda _: self.configChanged.emit("dt_min", float(self.dt.currentData())))
        self.shape.activated.connect(lambda _: self.shapeRequested.emit(dict(self.shape.currentData())))
        self.lighting = QtWidgets.QCheckBox("only sunlit facets can be imaged  (e)")
        self.lighting.toggled.connect(lambda v: self.configChanged.emit("lighting", v))
        gl.addWidget(self.lighting)
        self.shadows = QtWidgets.QCheckBox("terrain casts shadows")
        self.shadows.toggled.connect(lambda v: self.configChanged.emit("shadows", v))
        gl.addWidget(self.shadows)
        self.sun_lat = LabeledSlider("sub-solar latitude (season)", -90, 90, 1, 0, "°")
        self.sun_lat.valueChanged.connect(lambda v: self.configChanged.emit("sun_lat_deg", v))
        gl.addWidget(self.sun_lat)
        self.occlusion = QtWidgets.QCheckBox("terrain occlusion by ray casting  (c)")
        self.occlusion.toggled.connect(lambda v: self.configChanged.emit("occlusion", v))
        gl.addWidget(self.occlusion)
        form = QtWidgets.QFormLayout()
        self.dsep = QtWidgets.QDoubleSpinBox()
        self.dsep.setRange(0, 20)
        self.dsep.setSuffix(" km")
        self.dsep.valueChanged.connect(lambda v: self.configChanged.emit("d_sep", v))
        form.addRow("min. separation d_sep", self.dsep)
        gl.addLayout(form)
        lay.addWidget(g)

        g, gl = _group("Real-world constants")
        self.facts = _note("")
        gl.addWidget(self.facts)
        lay.addWidget(g)
        lay.addStretch(1)

    def _weights(self, *_):
        self.configChanged.emit("weights", tuple(b.value() for b in self.w))

    def set_state(self, cfg: MissionConfig, asteroid_spec: dict, r_max: float):
        ws = (self.body, self.n_max, self.t_max, self.t_min, self.camera, self.dt, self.lighting, self.shadows, self.sun_lat,
              self.occlusion, self.dsep, self.shape, *self.w)
        for w in ws:
            w.blockSignals(True)
        self.body.setCurrentIndex(max(0, self.body.findData(cfg.body)))
        self.n_max.setValue(cfg.n_max)
        i = self.t_max.findData(float(cfg.t_max_days))
        if i < 0:
            self.t_max.addItem(f"{cfg.t_max_days:g} days", float(cfg.t_max_days))
            i = self.t_max.count() - 1
        self.t_max.setCurrentIndex(i)
        self.t_min.setMaximum(cfg.t_max_days)
        self.t_min.setValue(cfg.t_min_days)
        self.camera.clear()
        for t, sc in cfg.scanners.items():
            self.camera.addItem(f"{sc.name} {sc.fov_deg:g}° (usable to {cfg.max_range(t):.0f} km)", t)
        self.camera.setCurrentIndex(max(0, self.camera.findData(cfg.camera)))
        for b, v in zip(self.w, cfg.weights):
            b.setValue(v)
        i = self.dt.findData(int(cfg.dt_min))
        if i < 0:
            self.dt.addItem(f"{cfg.dt_min:g} min", int(cfg.dt_min))
            i = self.dt.count() - 1
        self.dt.setCurrentIndex(i)
        for j in range(self.shape.count()):
            spec = self.shape.itemData(j)
            if all(asteroid_spec.get(k) == v for k, v in spec.items()):
                self.shape.setCurrentIndex(j)
        self.lighting.setChecked(cfg.lighting)
        self.shadows.setChecked(cfg.shadows)
        self.shadows.setEnabled(cfg.lighting)
        self.sun_lat.setValue(cfg.sun_lat_deg)
        self.sun_lat.setEnabled(cfg.lighting)
        self.occlusion.setChecked(cfg.occlusion)
        self.dsep.setValue(cfg.d_sep)
        for w in ws:
            w.blockSignals(False)
        r_min = cfg.resolve_r_min(r_max)
        v_lo, v_hi = np.sqrt(cfg.mu / cfg.r_max) * 1e3, np.sqrt(cfg.mu / r_min) * 1e3
        P_lo = 2 * np.pi * np.sqrt(r_min ** 3 / cfg.mu) / 3600
        P_hi = 2 * np.pi * np.sqrt(cfg.r_max ** 3 / cfg.mu) / 3600
        tip = 2 * np.pi * r_max / (cfg.p_rot_h * 3600) * 1e3
        self.facts.setText(
            f"GM of {cfg.body.capitalize()} {cfg.mu * 1e9:.4g} m³/s², spin period {cfg.p_rot_h:g} h "
            f"(surface tip moves {tip:.1f} m/s).<br>"
            f"Allowed orbits {r_min:.1f}-{cfg.r_max:g} km: circular speed {v_lo:.1f}-{v_hi:.1f} m/s, "
            f"period {P_lo:.0f}-{P_hi:.0f} h.<br>"
            f"Imaging: GSD ≤ {cfg.gsd_max_m:g} m, emission ≤ {cfg.eta_max_deg:g}°, "
            f"Sun incidence ≤ {cfg.incidence_max_deg:g}°.<br>"
            f"Power: bus {cfg.p_base_w:g} W, {cfg.e_img_j:g} J + {cfg.e_tx_j / 1000:.1f} kJ per frame "
            f"({cfg.t_downlink:.0f} s downlink at {cfg.downlink_bps / 1000:g} kbps); "
            f"solar {cfg.p_solar_w:g} W (0 in eclipse); battery {cfg.battery_wh:g} Wh, "
            f"≥ {100 * cfg.soc_min:.0f} %.")


# =========================================================================== Optimiser
PACES = [("full speed", 0.0), ("10 moves / s", 0.1), ("3 moves / s", 0.33), ("1 move / s", 1.0),
         ("1 move / 3 s", 3.0)]

# label, minimum, maximum, step (None-valued "auto" fields show their text at the minimum)
FIELD_HINTS = {
    "max_evals": ("evaluation budget", 20, 100000, 250),
    "chain_length": ("moves per temperature L", 0, 2000, 5, "auto: max(10, dim)"),
    "alpha": ("cooling factor α", 0.0, 0.999, 0.01, "auto: fits the budget"),
    "p0": ("initial acceptance p₀", 0.01, 0.99, 0.05),
    "t_min_ratio": ("final T / T₀", 1e-6, 0.5, 1e-3),
    "step0": ("initial step (× range)", 0.001, 1.0, 0.05),
    "step_min": ("smallest step (× range)", 0.0001, 1.0, 0.005),
    "max_vars_per_move": ("variables changed per move (max)", 1, 20, 1),
    "n_init": ("random designs to pick the start", 0, 1000, 5),
    "n_t0_samples": ("trial moves to set T₀", 1, 500, 5),
    "seed": ("random seed", 0, 1_000_000, 1),
    # GA
    "pop_size": ("population size P", 1, 5000, 10, "auto: 1.5·dim, 30-60"),
    "stratified": ("stratified start over fleet size N", 0, 1, 1),
    "elite_frac": ("elite share ε", 0.0, 0.5, 0.025),
    "crossover_frac": ("crossover fraction χ", 0.0, 1.0, 0.05),
    "tournament_k": ("tournament size k", 1, 20, 1),
    "align": ("align slots before crossover", 0, 1, 1),
    "p_blend": ("blend probability p_b", 0.0, 1.0, 0.05),
    "alpha_blend": ("BLX-α", 0.0, 1.0, 0.05),
    "blend_weight": ("fixed blend weight λ", 0.0, 1.0, 0.05, "random: BLX-α"),
}


class ParamsForm(QtWidgets.QWidget):
    """Form generated from an algorithm's Params dataclass (works for any future algorithm)."""

    def __init__(self, params_cls, parent=None):
        super().__init__(parent)
        self.params_cls = params_cls
        lay = QtWidgets.QFormLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.widgets = {}
        for f in dataclasses.fields(params_cls):
            default = f.default
            hint = FIELD_HINTS.get(f.name, (f.name, 0, 1e6, 1))
            label, lo, hi, step = hint[:4]
            is_bool = isinstance(default, bool)
            is_int = not is_bool and (isinstance(default, int) or (default is None and "int" in str(f.type)))
            if is_bool:
                w = QtWidgets.QCheckBox()
                w.setText("enabled")
            else:
                w = QtWidgets.QSpinBox() if is_int else QtWidgets.QDoubleSpinBox()
            if is_bool:
                pass
            elif is_int:
                w.setRange(int(lo), int(hi))
            else:
                w.setDecimals(6 if step < 1e-3 else 4 if step < 0.01 else 3)
                w.setRange(float(lo), float(hi))
            if not is_bool:
                w.setSingleStep(step)
            if default is None and not is_bool:
                w.setSpecialValueText(hint[4] if len(hint) > 4 else "auto")
            lay.addRow(label, w)
            self.widgets[f.name] = (w, default is None)

    def set_values(self, values: dict):
        defaults = dataclasses.asdict(self.params_cls())
        for name, (w, optional) in self.widgets.items():
            v = values.get(name, defaults[name])
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(w.minimum() if v is None else v)

    def values(self) -> dict:
        out = {}
        for name, (w, optional) in self.widgets.items():
            if isinstance(w, QtWidgets.QCheckBox):
                out[name] = w.isChecked()
            else:
                v = w.value()
                out[name] = None if optional and v == w.minimum() else v
        return out


class OptimiserPanel(QtWidgets.QWidget):
    runRequested = pyqtSignal(str, object, bool)   # algorithm, params dict, start from current design
    stopRequested = pyqtSignal()
    paceChanged = pyqtSignal(float)                # seconds per move
    pauseToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        g, gl = _group("Algorithm")
        self.algo = QtWidgets.QComboBox()
        names = {"sa": "Simulated annealing", "ga": "Genetic algorithm", "random": "Random search (baseline)"}
        for a in algorithms.available():
            self.algo.addItem(names.get(a, a), a)
        gl.addWidget(self.algo)
        self.form_box = QtWidgets.QVBoxLayout()
        gl.addLayout(self.form_box)
        self.form: ParamsForm | None = None
        form = QtWidgets.QFormLayout()
        self.start_from = QtWidgets.QComboBox()
        self.start_from.addItem("random designs", False)
        self.start_from.addItem("initial design (before any run)", True)
        self.start_from.setToolTip("Every run starts fresh: from random designs, or from the design you loaded "
                                   "or edited. A previous SA / GA result is never used as the start.")
        form.addRow("start from", self.start_from)
        gl.addLayout(form)
        self.algo.currentIndexChanged.connect(lambda _: self._build_form({}))
        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("▶  Run")
        self.run_btn.setMinimumHeight(32)
        self.stop_btn = QtWidgets.QPushButton("■  Stop")
        self.stop_btn.setMinimumHeight(32)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        gl.addLayout(row)
        self.run_btn.clicked.connect(lambda: self.runRequested.emit(self.algorithm(), self.params(),
                                                                     bool(self.start_from.currentData())))
        self.stop_btn.clicked.connect(self.stopRequested)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("watch pace"))
        self.pace = QtWidgets.QComboBox()
        for label, delay in PACES:
            self.pace.addItem(label, delay)
        self.pace.currentIndexChanged.connect(lambda _: self.paceChanged.emit(float(self.pace.currentData())))
        row.addWidget(self.pace, 1)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.toggled.connect(self._on_pause)
        row.addWidget(self.pause_btn)
        gl.addLayout(row)
        self.run_progress = QtWidgets.QProgressBar()
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(0)
        self.run_progress.setFormat("%v / %m evaluations")
        self.run_progress.setMaximumHeight(16)
        self.run_progress.setVisible(False)
        gl.addWidget(self.run_progress)
        self.status = _note("")
        gl.addWidget(self.status)
        self.population = _note("")
        self.population.setTextFormat(Qt.RichText)
        self.population.setVisible(False)
        gl.addWidget(self.population)
        lay.addWidget(g)

        g, gl = _group("Search log  (what the algorithm is trying)")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(220)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSize(8)
        self.log.setFont(font)
        gl.addWidget(self.log)
        lay.addWidget(g)
        self.help = _note("")
        lay.addWidget(self.help)
        lay.addStretch(1)
        self._build_form({})

    def _build_form(self, values: dict):
        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()
        self.form = ParamsForm(algorithms.get(self.algorithm()).Params)
        self.form.set_values(values)
        self.form_box.addWidget(self.form)
        self._update_mode_help()

    def _update_mode_help(self):
        if self.algorithm() == "ga":
            self.population.setVisible(True)
            self.help.setText(
                "<b>GA loop:</b> evaluate a population → keep the elites → pick parents by tournament → "
                "make crossover and mutation children → repair them and re-mutate duplicates → "
                "evaluate → next generation.<br><br>"
                "<b>Genome:</b> mission time plus one orbit slot per spacecraft. Slot alignment keeps "
                "the same spacecraft together when two parents cross over. Lower <i>J</i> is better; "
                "feasible designs always beat infeasible designs.")
        elif self.algorithm() == "sa":
            self.population.setVisible(False)
            self.help.setText(
                "<b>Loop:</b> Algorithm → Feasibility → Engine → cost → Algorithm.<br><br>"
                "<b>Simulated annealing:</b> change 1–2 active variables by a shrinking Gaussian step. "
                "Better designs are always accepted; worse designs may be accepted early to escape local minima.")
        else:
            self.population.setVisible(False)
            self.help.setText(
                "<b>Random search:</b> sample repaired designs, evaluate them, and keep the best feasible result. "
                "Use it as a baseline for comparing GA and simulated annealing.")

    def algorithm(self) -> str:
        return self.algo.currentData()

    def set_params(self, name: str, values: dict):
        i = self.algo.findData(name)
        self.algo.blockSignals(True)
        self.algo.setCurrentIndex(max(i, 0))
        self.algo.blockSignals(False)
        self._build_form(values)
        self._update_mode_help()

    def params(self) -> dict:
        return self.form.values()

    def _on_pause(self, on: bool):
        self.pause_btn.setText("Resume" if on else "Pause")
        self.pauseToggled.emit(on)

    def set_progress(self, done: int, total: int):
        self.run_progress.setVisible(True)
        self.run_progress.setRange(0, max(total, 1))
        self.run_progress.setValue(min(done, max(total, 1)))

    def set_running(self, on: bool):
        if on:
            self.set_progress(0, 1)
        self.run_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.pause_btn.setEnabled(on)
        self.algo.setEnabled(not on)
        if not on and self.pause_btn.isChecked():
            self.pause_btn.setChecked(False)


# =========================================================================== Design
class DesignPanel(QtWidgets.QWidget):
    selectionChanged = pyqtSignal(int)
    spacecraftEdited = pyqtSignal(int, str, float)     # index, field, value
    missionTimeChanged = pyqtSignal(float)             # days
    addRequested = pyqtSignal()
    removeRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        g, gl = _group("Mission")
        self.T = LabeledSlider("T  mission duration", 0.25, 10.0, 0.05, 2, " d")
        self.T.valueChanged.connect(self.missionTimeChanged)
        gl.addWidget(self.T)
        lay.addWidget(g)

        g, gl = _group("Spacecraft  (N = number in the list)")
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(150)
        self.list.currentRowChanged.connect(self._on_row)
        gl.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("+ Add")
        rem = QtWidgets.QPushButton("− Remove")
        add.clicked.connect(self.addRequested)
        rem.clicked.connect(lambda: self.removeRequested.emit(self.list.currentRow()))
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        gl.addLayout(row)
        lay.addWidget(g)

        self.editor_box, self.editor = _group("Selected spacecraft")
        lay.addWidget(self.editor_box)
        self.sliders: dict[str, LabeledSlider] = {}
        for name, label, lo, hi, step, dec, unit in DESIGN_FIELDS:
            s = LabeledSlider(label, lo, hi, step, dec, unit)
            s.valueChanged.connect(lambda v, n=name: self.spacecraftEdited.emit(self.list.currentRow(), n, v))
            self.editor.addWidget(s)
            self.sliders[name] = s
        self.info = _note("Circular orbit, scanner pointing at the asteroid centre (nadir). A radius "
                          "below r_min or above r_max, or two spacecraft closer than d_sep, shows as a "
                          "violated constraint.")
        self.editor.addWidget(self.info)
        lay.addStretch(1)

    @staticmethod
    def _row_text(i, o):
        return f"SC{i + 1}   r {o.a:5.1f} km   ι {o.inc:5.1f}°   Ω {o.raan:5.1f}°   M₀ {o.M0:5.1f}°"

    def set_design(self, design, selected: int, cfg: MissionConfig | None = None):
        if cfg is not None:
            self.T.set_range(cfg.t_min_days, cfg.t_max_days)
        self.T.setValue(design.T_days)
        self.list.blockSignals(True)
        self.list.clear()
        for i, o in enumerate(design.orbits):
            it = QtWidgets.QListWidgetItem(self._row_text(i, o))
            pix = QtGui.QPixmap(12, 12)
            pix.fill(QtGui.QColor(colors.sc_color_hex(i)))
            it.setIcon(QtGui.QIcon(pix))
            self.list.addItem(it)
        selected = max(0, min(selected, design.N - 1))
        self.list.setCurrentRow(selected)
        self.list.blockSignals(False)
        if design.N:
            self.show_spacecraft(design.orbits[selected])

    def update_row_text(self, design, i):
        if 0 <= i < self.list.count():
            self.list.item(i).setText(self._row_text(i, design.orbits[i]))

    def show_spacecraft(self, orbit):
        for name, s in self.sliders.items():
            s.setValue(getattr(orbit, name))
        self.editor_box.setTitle(f"Selected spacecraft — SC{self.list.currentRow() + 1}")

    def _on_row(self, row):
        if row >= 0:
            self.selectionChanged.emit(row)


# =========================================================================== View
class ViewPanel(QtWidgets.QWidget):
    colorModeChanged = pyqtSignal(str)
    groupToggled = pyqtSignal(str, bool)
    highlightToggled = pyqtSignal(bool)
    edgesToggled = pyqtSignal(bool)
    frameChanged = pyqtSignal(str)
    markerSizeChanged = pyqtSignal(float)

    LABELS = {"orbits": "orbits — exact Kepler ellipses  (O)",
              "tracks": "whole-mission paths (body frame)",
              "trails": "trails, last 3 h  (T)",
              "cones": "scanner FOV cones  (C)",
              "labels": "labels  (L)",
              "sun": "Sun direction",
              "safety": "r_min / r_max spheres",
              "axes": "axes widget"}

    def __init__(self, scene_visible: dict, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        g, gl = _group("Frame  (B)")
        self.frame = QtWidgets.QComboBox()
        self.frame.addItem("Inertial — asteroid spins, orbits are fixed ellipses", "inertial")
        self.frame.addItem("Asteroid-fixed — paths wind around the body", "body")
        self.frame.currentIndexChanged.connect(lambda _: self.frameChanged.emit(self.frame.currentData()))
        gl.addWidget(self.frame)
        lay.addWidget(g)

        g, gl = _group("Surface colouring  (M)")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(colors.COLOR_MODES)
        self.mode.currentTextChanged.connect(self.colorModeChanged)
        gl.addWidget(self.mode)
        self.highlight = QtWidgets.QCheckBox("highlight what each scanner sees right now")
        self.highlight.setChecked(True)
        self.highlight.toggled.connect(self.highlightToggled)
        gl.addWidget(self.highlight)
        self.edges = QtWidgets.QCheckBox("show facet edges")
        self.edges.toggled.connect(self.edgesToggled)
        gl.addWidget(self.edges)
        lay.addWidget(g)

        g, gl = _group("Show")
        self.checks = {}
        for key, label in self.LABELS.items():
            c = QtWidgets.QCheckBox(label)
            c.setChecked(scene_visible.get(key, True))
            c.toggled.connect(lambda v, k=key: self.groupToggled.emit(k, v))
            gl.addWidget(c)
            self.checks[key] = c
        self.marker = LabeledSlider("spacecraft marker size (screen pixels, not to scale)", 0, 30, 1, 0, " px")
        self.marker.setValue(11)
        self.marker.valueChanged.connect(self.markerSizeChanged)
        gl.addWidget(self.marker)
        lay.addWidget(g)
        lay.addStretch(1)

    def set_check(self, key, on):
        c = self.checks[key]
        c.blockSignals(True)
        c.setChecked(on)
        c.blockSignals(False)

    def set_frame(self, frame):
        self.frame.blockSignals(True)
        self.frame.setCurrentIndex(self.frame.findData(frame))
        self.frame.blockSignals(False)

    def set_mode(self, mode):
        self.mode.blockSignals(True)
        self.mode.setCurrentText(mode)
        self.mode.blockSignals(False)


DARK_QSS = """
QWidget { background: #10131f; color: #dfe6f5; font-size: 12px; }
QGroupBox { border: 1px solid #2a3150; border-radius: 6px; margin-top: 14px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #8fb4ff; font-weight: bold; }
QPushButton { background: #1d2440; border: 1px solid #34406c; border-radius: 4px; padding: 4px 8px; }
QPushButton:hover { background: #2a3460; }
QPushButton:checked { background: #3a5ab0; }
QPushButton:disabled { color: #59627e; }
QComboBox, QSpinBox, QDoubleSpinBox, QListWidget { background: #171c30; border: 1px solid #2c3558;
    border-radius: 3px; padding: 2px; }
QListWidget::item:selected { background: #2c3f7a; }
QSlider::groove:horizontal { height: 4px; background: #2a3150; border-radius: 2px; }
QSlider::handle:horizontal { background: #7aa2ff; width: 12px; margin: -5px 0; border-radius: 6px; }
QTabWidget::pane { border: 1px solid #2a3150; }
QTabBar::tab { background: #161a2c; padding: 6px 10px; border: 1px solid #2a3150; }
QTabBar::tab:selected { background: #24305a; color: #ffffff; }
QProgressBar { border: 1px solid #2a3150; border-radius: 3px; text-align: center; font-size: 11px;
    background: #151a2b; }
QProgressBar::chunk { background: #2a78d6; border-radius: 2px; }
QToolButton { background: #1d2440; border: 1px solid #34406c; border-radius: 4px; padding: 1px 6px;
    color: #9aa8c8; font-size: 11px; }
QToolButton:checked { background: #2c3f7a; color: #ffffff; }
QToolTip { background: #0e1222; color: #e8eeff; border: 1px solid #2a3150; padding: 4px; }
QTabBar::tab:hover { background: #1d2440; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #4a5a90; }
QScrollBar:vertical { background: #10131f; width: 10px; }
QScrollBar::handle:vertical { background: #2a3150; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolBar { background: #0c0f1a; border-top: 1px solid #2a3150; spacing: 6px; }
QStatusBar { background: #0c0f1a; color: #9aa8c8; }
QScrollArea { border: none; }
QSplitter::handle { background: #1a2036; }
"""

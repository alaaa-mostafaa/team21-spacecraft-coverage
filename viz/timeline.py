"""Mission timeline bar: play / pause, scrub, speed (simulated seconds per real second)."""
from __future__ import annotations

import time

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

SPEEDS = [1, 10, 60, 300, 900, 1800, 3600, 7200, 14400, 28800]      # simulated s per real s


def _speed_label(s):
    if s == 1:
        return "×1  (real time)"
    if s < 60:
        return f"×{s}  ({s} s/s)"
    return f"×{s}  ({s / 60:g} min/s)" if s < 3600 else f"×{s}  ({s / 3600:g} h/s)"


class TimelineBar(QtWidgets.QToolBar):
    timeChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__("Mission timeline", parent)
        self.setMovable(False)
        self.dt = 600.0
        self.n_steps = 433
        self.t = 0.0
        self.speed = 3600.0
        self.loop = False                     # stop at the end of the mission

        self.play_btn = QtWidgets.QPushButton("▶  Play")
        self.play_btn.setCheckable(True)
        self.play_btn.setFixedWidth(90)
        self.play_btn.toggled.connect(self._on_play)
        self.addWidget(self.play_btn)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.setMinimumWidth(300)
        self.addWidget(self.slider)

        self.clock = QtWidgets.QLabel()
        self.clock.setMinimumWidth(190)
        self.addWidget(self.clock)

        self.speed_box = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem(_speed_label(s), s)
        self.speed_box.setCurrentIndex(SPEEDS.index(3600))
        self.speed_box.currentIndexChanged.connect(lambda _: setattr(self, "speed", float(self.speed_box.currentData())))
        self.addWidget(QtWidgets.QLabel(" speed "))
        self.addWidget(self.speed_box)

        self.loop_box = QtWidgets.QCheckBox("loop")
        self.loop_box.setChecked(False)
        self.loop_box.toggled.connect(lambda v: setattr(self, "loop", v))
        self.addWidget(self.loop_box)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._last = None
        self._update_clock()

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return (self.n_steps - 1) * self.dt

    def set_mission(self, dt: float, n_steps: int):
        self.dt, self.n_steps = dt, n_steps
        self.slider.blockSignals(True)
        self.slider.setRange(0, n_steps - 1)
        self.slider.blockSignals(False)
        self.set_time(min(self.t, self.duration), emit=True)

    def set_time(self, t: float, emit: bool = True):
        self.t = max(0.0, min(t, self.duration))
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.t // self.dt))
        self.slider.blockSignals(False)
        self._update_clock()
        if emit:
            self.timeChanged.emit(self.t)

    def toggle_play(self):
        self.play_btn.toggle()

    def step(self, n: int):
        self.set_time((int(self.t // self.dt) + n) * self.dt)

    def faster(self, d: int):
        i = max(0, min(self.speed_box.currentIndex() + d, len(SPEEDS) - 1))
        self.speed_box.setCurrentIndex(i)

    def is_playing(self) -> bool:
        return self.play_btn.isChecked()

    # ------------------------------------------------------------------
    def _on_play(self, on):
        self.play_btn.setText("❚❚  Pause" if on else "▶  Play")
        if on:
            if self.t >= self.duration:
                self.set_time(0.0)
            self._last = time.perf_counter()
            self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        now = time.perf_counter()
        dt_real = min(now - self._last, 0.1)
        self._last = now
        t = self.t + dt_real * self.speed
        if t > self.duration:
            if self.loop:
                t = 0.0
            else:
                t = self.duration
                self.play_btn.setChecked(False)
        self.set_time(t)

    def _on_slider(self, k):
        self.set_time(k * self.dt)

    def _update_clock(self):
        day, rem = divmod(int(self.t), 86400)
        k = int(self.t // self.dt)
        self.clock.setText(f"  Day {day + 1}  {rem // 3600:02d}:{(rem % 3600) // 60:02d}   "
                           f"step {k}/{self.n_steps - 1}")

"""Game-style camera: WASD/QE fly, arrow-key look, ride-along and chase cameras.

Keys are taken at the Qt level (event filter) so VTK's built-in single-letter
shortcuts (w = wireframe, 3 = stereo, q/e = quit ...) never fire.
Mouse handling stays with VTK's trackball style (rotate / pan / zoom).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from spacecov.geometry import unit

MOVE_KEYS = {Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E,
             Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down}

HELP_TEXT = """CONTROLS
Mouse  left drag  rotate     right drag / wheel  zoom
       middle drag / shift+left  pan
W A S D        fly forward / left / back / right
Q E            fly down / up        Shift  x4 speed
Arrow keys     look around
1 .. 7         ride along spacecraft i (look through its scanner)
F              chase selected spacecraft: 30 m behind it, true size
0 / Esc        free camera          R  reset view
Space          play / pause         , .  step -/+     [ ]  slower / faster
M  colour mode    B  inertial / asteroid-fixed frame
C  cones   O  orbits   L  labels   T  trails
P  screenshot     H  toggle this help"""


class CameraController(QtCore.QObject):
    FREE, RIDE, CHASE = "free", "ride", "chase"

    def __init__(self, plotter, widget, scene_radius: Callable[[], float]):
        super().__init__(widget)
        self.p = plotter
        self.widget = widget
        self.scene_radius = scene_radius
        self.mode = self.FREE
        self.target = 0
        self._prev = None
        self.eye_offset = 0.003        # km (3 m) in front of the S/C centre (lens position)
        self._held: set[int] = set()
        self._shift = False
        self._actions: dict[int, Callable[[], None]] = {}
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._fly_tick)
        self._last = time.perf_counter()
        self.on_mode_change: Callable[[str], None] | None = None
        self.request_follow: Callable[[], None] | None = None      # app re-applies follow() + renders
        widget.installEventFilter(self)

    # ------------------------------------------------------------------ bindings
    def bind(self, key: int, fn: Callable[[], None]):
        self._actions[key] = fn

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return True
            key = ev.key()
            self._shift = bool(ev.modifiers() & Qt.ShiftModifier)
            if key in MOVE_KEYS:
                if self.mode == self.RIDE:
                    self.set_mode(self.FREE)
                self._held.add(key)
                if not self._timer.isActive():
                    self._last = time.perf_counter()
                    self._timer.start()
            elif Qt.Key_1 <= key <= Qt.Key_7:
                self.ride(key - Qt.Key_1)
            elif key in (Qt.Key_0, Qt.Key_Escape):
                self.set_mode(self.FREE)
            elif key in self._actions:
                self._actions[key]()
            return True
        if t == QtCore.QEvent.KeyRelease:
            if not ev.isAutoRepeat():
                self._held.discard(ev.key())
                self._shift = bool(ev.modifiers() & Qt.ShiftModifier)
            return True
        if t == QtCore.QEvent.FocusOut:
            self._held.clear()
        return False

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: str, target: int | None = None):
        self.mode = mode
        if target is not None:
            self.target = target
        self._prev = None
        if mode == self.FREE:
            cam = self.p.camera
            cam.view_angle = 35.0
        if self.on_mode_change:
            self.on_mode_change(self.describe())
        if mode != self.FREE and self.request_follow:
            self.request_follow()

    def ride(self, i: int):
        self.set_mode(self.RIDE, i)

    def chase(self, i: int):
        if self.mode == self.CHASE and self.target == i:
            self.set_mode(self.FREE)
        else:
            self.set_mode(self.CHASE, i)

    def describe(self) -> str:
        if self.mode == self.RIDE:
            return f"camera: riding SC{self.target + 1}"
        if self.mode == self.CHASE:
            return f"camera: chasing SC{self.target + 1}"
        return "camera: free"

    def follow(self, poses):
        """Called after every scene update with the current spacecraft poses."""
        if self.mode == self.FREE or not poses:
            return
        if self.target >= len(poses):
            self.set_mode(self.FREE)
            return
        pose = poses[self.target]
        cam = self.p.camera
        pos, R = np.asarray(pose.position, float), pose.rotation
        if self.mode == self.RIDE:              # look out through the scanner lens
            d, up = R[:, 2], R[:, 1]
            cam.position = tuple(pos + d * self.eye_offset)
            reach = float(np.min(np.linalg.norm(pose.rim - pos, axis=1)))
            cam.focal_point = tuple(pos + d * max(reach, 5.0))
            cam.up = tuple(up)
            cam.view_angle = min(110.0, max(30.0, np.rad2deg(pose.fov) * 1.25))
        else:                                   # chase: offset kept in the spacecraft frame
            if self._prev is None:
                local = np.array([0.0, 0.008, -0.03])          # 30 m behind (away from asteroid), 8 m above
                up_local = np.array([0.0, 1.0, 0.0])
            else:
                p0, R0 = self._prev
                local = R0.T @ (np.asarray(cam.position) - p0)  # keeps mouse orbiting around the S/C
                up_local = R0.T @ np.asarray(cam.up)
            cam.position = tuple(pos + R @ local)
            cam.focal_point = tuple(pos)
            cam.up = tuple(R @ up_local)
        self._prev = (pos, R)
        self.p.renderer.ResetCameraClippingRange()

    # ------------------------------------------------------------------ fly
    def _fly_tick(self):
        now = time.perf_counter()
        dt = min(now - self._last, 0.05)
        self._last = now
        if not self._held:
            self._timer.stop()
            return
        cam = self.p.camera
        pos, foc = np.asarray(cam.position), np.asarray(cam.focal_point)
        fwd = unit(foc - pos)
        up = unit(np.asarray(cam.up))
        right = unit(np.cross(fwd, up))
        speed = 0.6 * self.scene_radius() * (4.0 if self._shift else 1.0)
        h = self._held
        move = (fwd * ((Qt.Key_W in h) - (Qt.Key_S in h)) + right * ((Qt.Key_D in h) - (Qt.Key_A in h))
                + up * ((Qt.Key_E in h) - (Qt.Key_Q in h)))
        if np.any(move) and self.mode == self.CHASE:
            self.set_mode(self.FREE)
        cam.position = tuple(pos + move * speed * dt)
        cam.focal_point = tuple(foc + move * speed * dt)
        yaw = ((Qt.Key_Left in h) - (Qt.Key_Right in h)) * 60.0 * dt
        pitch = ((Qt.Key_Up in h) - (Qt.Key_Down in h)) * 45.0 * dt
        if yaw:
            cam.Yaw(yaw)
        if pitch:
            cam.Pitch(pitch)
            cam.OrthogonalizeViewUp()
        self.p.renderer.ResetCameraClippingRange()
        self.p.render()

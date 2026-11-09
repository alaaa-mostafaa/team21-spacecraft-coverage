"""Small, fast line-chart widget drawn with QPainter (no plotting library needed).

Used for the coverage-vs-time chart and the live simulated-annealing charts.  Long
series are reduced to a min/max envelope per pixel column, so redrawing thousands of
points ten times a second stays cheap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

BG = QtGui.QColor("#0b0e19")
FRAME = QtGui.QColor("#2a3150")
GRID = QtGui.QColor("#1a2036")
TEXT = QtGui.QColor("#aab6d6")
TITLE = QtGui.QColor("#8fb4ff")


@dataclass
class Series:
    name: str
    x: np.ndarray
    y: np.ndarray
    color: str
    width: float = 1.5
    style: str = "line"          # line | dots | step
    alpha: float = 1.0
    legend: bool = True


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step + 1e-9) + 1)]


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f"{v:.0e}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.3g}"
    return f"{v:.3g}"


class PlotWidget(QtWidgets.QWidget):
    clicked = pyqtSignal(float)          # x value under the mouse

    def __init__(self, title: str, xlabel: str = "", ylabel: str = "", log_y: bool = False, parent=None):
        super().__init__(parent)
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.log_y = log_y
        self.series: dict[str, Series] = {}
        self.order: list[str] = []
        self.x_range: tuple[float, float] | None = None
        self.y_range: tuple[float, float] | None = None
        self.cursor_x: float | None = None
        self.cursor_text = ""
        self.subtitle = ""
        self.legend_loc = "tl"             # tl | tr | bl | br
        self.yfmt: Callable[[float], str] = fmt
        self.xfmt: Callable[[float], str] = fmt
        self.setMinimumSize(260, 170)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # ------------------------------------------------------------------ data
    def set_series(self, name, x, y, color, width=1.5, style="line", alpha=1.0, legend=True):
        if name not in self.series:
            self.order.append(name)
        self.series[name] = Series(name, np.asarray(x, float), np.asarray(y, float), color, width, style,
                                   alpha, legend)

    def remove_series(self, name):
        self.series.pop(name, None)
        if name in self.order:
            self.order.remove(name)

    def clear(self):
        self.series.clear()
        self.order.clear()
        self.cursor_x = None
        self.cursor_text = ""
        self.subtitle = ""

    def set_cursor(self, x: float | None, text: str = ""):
        self.cursor_x, self.cursor_text = x, text

    # ------------------------------------------------------------------ geometry
    def _limits(self):
        xs = [s.x for s in self.series.values() if len(s.x)]
        ys = [s.y[np.isfinite(s.y)] for s in self.series.values() if len(s.y)]
        ys = [y[y > 0] if self.log_y else y for y in ys]
        ys = [y for y in ys if len(y)]
        if self.x_range:
            x0, x1 = self.x_range
        elif xs:
            x0, x1 = min(float(x.min()) for x in xs), max(float(x.max()) for x in xs)
        else:
            x0, x1 = 0.0, 1.0
        if self.y_range:
            y0, y1 = self.y_range
        elif ys:
            y0, y1 = min(float(y.min()) for y in ys), max(float(y.max()) for y in ys)
        else:
            y0, y1 = (1e-3, 1.0) if self.log_y else (0.0, 1.0)
        if x1 <= x0:
            x1 = x0 + 1.0
        if self.log_y:
            y0, y1 = math.log10(max(y0, 1e-12)), math.log10(max(y1, 1e-12))
        if y1 <= y0:
            pad = abs(y0) * 0.05 or 0.05
            y0, y1 = y0 - pad, y1 + pad
        elif not self.y_range:
            pad = 0.06 * (y1 - y0)
            y0, y1 = y0 - pad, y1 + pad
        return x0, x1, y0, y1

    def _plot_rect(self) -> QtCore.QRectF:
        top = 44 if self.subtitle else 28
        return QtCore.QRectF(54, top, max(10, self.width() - 66), max(10, self.height() - top - 32))

    # ------------------------------------------------------------------ paint
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        R = self._plot_rect()
        x0, x1, y0, y1 = self._limits()

        def X(x):
            return R.left() + (np.asarray(x, float) - x0) / (x1 - x0) * R.width()

        def Y(y):
            y = np.asarray(y, float)
            if self.log_y:
                y = np.log10(np.maximum(y, 1e-12))
            return R.bottom() - (y - y0) / (y1 - y0) * R.height()

        small = QtGui.QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.5))
        p.setFont(small)
        fm = QtGui.QFontMetrics(small)

        # grid + tick labels
        p.setPen(QtGui.QPen(GRID, 1))
        for tx in nice_ticks(x0, x1, max(2, int(R.width() / 90))):
            px = float(X(tx))
            p.setPen(QtGui.QPen(GRID, 1))
            p.drawLine(QtCore.QPointF(px, R.top()), QtCore.QPointF(px, R.bottom()))
            p.setPen(TEXT)
            s = self.xfmt(tx)
            p.drawText(QtCore.QPointF(px - fm.horizontalAdvance(s) / 2, R.bottom() + fm.height()), s)
        if self.log_y:
            yt = [10.0 ** e for e in range(math.floor(y0), math.ceil(y1) + 1) if y0 <= e <= y1] or [10 ** y0]
        else:
            yt = nice_ticks(y0, y1, max(2, int(R.height() / 40)))
        for ty in yt:
            py = float(Y(ty))
            p.setPen(QtGui.QPen(GRID, 1))
            p.drawLine(QtCore.QPointF(R.left(), py), QtCore.QPointF(R.right(), py))
            p.setPen(TEXT)
            s = self.yfmt(ty)
            p.drawText(QtCore.QPointF(R.left() - 6 - fm.horizontalAdvance(s), py + fm.ascent() / 2 - 1), s)

        # series
        p.setClipRect(R.adjusted(-2, -2, 2, 2))
        for name in self.order:
            s = self.series[name]
            if len(s.x) == 0:
                continue
            col = QtGui.QColor(s.color)
            col.setAlphaF(s.alpha)
            px, py = self._reduce(X(s.x), Y(s.y), R.width())
            ok = np.isfinite(px) & np.isfinite(py)
            px, py = px[ok], py[ok]
            if len(px) == 0:
                continue
            if s.style == "dots":
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                for a, b in zip(px, py):
                    p.drawEllipse(QtCore.QPointF(a, b), s.width, s.width)
                continue
            if s.style == "step":
                sx = np.repeat(px, 2)[1:]
                sy = np.repeat(py, 2)[:-1]
                px, py = sx, sy
            path = QtGui.QPainterPath(QtCore.QPointF(px[0], py[0]))
            for a, b in zip(px[1:], py[1:]):
                path.lineTo(a, b)
            p.setBrush(Qt.NoBrush)
            p.setPen(QtGui.QPen(col, s.width))
            p.drawPath(path)
        p.setClipping(False)

        # cursor
        if self.cursor_x is not None and x0 <= self.cursor_x <= x1:
            cx = float(X(self.cursor_x))
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1, Qt.DashLine))
            p.drawLine(QtCore.QPointF(cx, R.top()), QtCore.QPointF(cx, R.bottom()))
            if self.cursor_text:
                w = fm.horizontalAdvance(self.cursor_text) + 8
                tx = min(cx + 4, R.right() - w)
                box = QtCore.QRectF(tx, R.bottom() - fm.height() - 6, w, fm.height() + 2)
                p.fillRect(box, QtGui.QColor(10, 14, 28, 220))
                p.setPen(QtGui.QColor("#ffffff"))
                p.drawText(box, Qt.AlignCenter, self.cursor_text)

        # frame, title, labels, legend
        p.setBrush(Qt.NoBrush)
        p.setPen(QtGui.QPen(FRAME, 1))
        p.drawRect(R)
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(TITLE)
        p.drawText(QtCore.QPointF(8, 18), self.title)
        if self.subtitle:
            p.setFont(small)
            p.setPen(QtGui.QColor("#e8eeff"))
            p.drawText(QtCore.QPointF(8, 34), self.subtitle)
        p.setFont(small)
        p.setPen(TEXT)
        if self.xlabel:
            p.drawText(QtCore.QPointF(R.right() - fm.horizontalAdvance(self.xlabel), self.height() - 6),
                       self.xlabel)
        if self.ylabel:
            p.save()
            p.translate(12, R.center().y() + fm.horizontalAdvance(self.ylabel) / 2)
            p.rotate(-90)
            p.drawText(QtCore.QPointF(0, 0), self.ylabel)
            p.restore()
        entries = [self.series[n] for n in self.order if self.series[n].legend]
        lw = max((fm.horizontalAdvance(s.name) for s in entries), default=0) + 24
        lh = len(entries) * (fm.height() + 1)
        lx = R.left() + 8 if self.legend_loc[1] == "l" else R.right() - lw
        ly = R.top() + 6 if self.legend_loc[0] == "t" else R.bottom() - lh - 6
        if entries:
            p.fillRect(QtCore.QRectF(lx - 4, ly, lw, lh + 4), QtGui.QColor(11, 14, 25, 200))
        for s in entries:
            col = QtGui.QColor(s.color)
            p.fillRect(QtCore.QRectF(lx, ly + 4, 12, 3), col)
            p.setPen(TEXT)
            p.drawText(QtCore.QPointF(lx + 16, ly + fm.ascent()), s.name)
            ly += fm.height() + 1
        p.end()

    @staticmethod
    def _reduce(px: np.ndarray, py: np.ndarray, width: float):
        """Min/max envelope per pixel column for long series."""
        n = len(px)
        if n <= 4 * max(width, 1):
            return px, py
        cols = np.floor(px).astype(int)
        order = np.argsort(cols, kind="stable")
        cols, px, py = cols[order], px[order], py[order]
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
        lo = np.minimum.reduceat(py, starts)
        hi = np.maximum.reduceat(py, starts)
        x = px[starts]
        return np.repeat(x, 2), np.column_stack([lo, hi]).ravel()

    def mousePressEvent(self, ev):
        R = self._plot_rect()
        if R.contains(ev.pos()):
            x0, x1, _, _ = self._limits()
            self.clicked.emit(x0 + (ev.pos().x() - R.left()) / R.width() * (x1 - x0))

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self.mousePressEvent(ev)

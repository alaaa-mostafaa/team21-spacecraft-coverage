"""Metrics dashboard for the displayed design: KPI tiles, what J is made of, constraints, fleet.

Fed by ``Presenter.metrics_data()``; pure display, no signals.
"""
from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# text tokens (values and labels never wear a series colour)
INK = "#e8eeff"
INK_2 = "#9aa8c8"
INK_3 = "#6c7899"
SURFACE = "#151a2b"
TRACK = "#232a44"
GOOD = "#5fd38a"
BAD = "#ff5c5c"
# the four weighted terms of J (categorical slots 1-4, validated on the dark surface)
TERM_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]


def _label(text="", size=None, color=INK_2, bold=False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    css = f"color:{color}; background:transparent;"
    if size:
        css += f" font-size:{size}px;"
    if bold:
        css += " font-weight:600;"
    w.setStyleSheet(css)
    return w


class KpiTile(QtWidgets.QFrame):
    """One headline number: small caption, big value, one line of context."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setObjectName("kpi")
        self.setStyleSheet(f"#kpi {{ background:{SURFACE}; border:1px solid #262e4a; border-radius:6px; }}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 4)
        lay.setSpacing(0)
        self.caption = _label(caption, 11, INK_3)
        self.value = _label("—", 17, INK, bold=True)
        self.sub = _label("", 11, INK_2)
        self.sub.setTextFormat(Qt.RichText)
        for w in (self.caption, self.value, self.sub):
            lay.addWidget(w)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set(self, value: str, sub: str = "", tip: str = ""):
        self.value.setText(value)
        self.sub.setText(sub)
        self.setToolTip(tip)


class CostBar(QtWidgets.QWidget):
    """Stacked bar of the weighted terms of J, with a legend; hover a segment for its formula."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.terms: list[tuple[str, float, str]] = []
        self.total = 0.0
        self.note = ""
        self._rects: list[tuple[QtCore.QRectF, int]] = []
        self.setMouseTracking(True)
        self.setMinimumHeight(74)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_terms(self, terms, total):
        self.terms, self.total = list(terms), float(total)
        self.update()

    def sizeHint(self):
        return QtCore.QSize(380, 74)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        f = QtGui.QFont(self.font())
        f.setPixelSize(11)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        W = self.width()
        p.setPen(QtGui.QColor(INK_3))
        title = "What J is made of  (each term = weight × objective)"
        if self.note:
            title = f"What J is made of  ({self.note})"
        p.drawText(QtCore.QPointF(0, fm.ascent()), fm.elidedText(title, Qt.ElideRight, W))
        top, h = fm.height() + 4, 14
        bar = QtCore.QRectF(0, top, W, h)
        path = QtGui.QPainterPath()
        path.addRoundedRect(bar, 4, 4)
        p.fillPath(path, QtGui.QColor(TRACK))
        self._rects = []
        vals = np.array([max(v, 0.0) for _, v, _ in self.terms])
        tot = float(vals.sum())
        if tot > 0:
            p.save()
            p.setClipPath(path)
            x = 0.0
            for i, v in enumerate(vals):
                w = W * v / tot
                if w <= 0:
                    continue
                r = QtCore.QRectF(x, top, max(w - 2, 1), h)          # 2px surface gap between segments
                p.fillRect(r, QtGui.QColor(TERM_COLORS[i % len(TERM_COLORS)]))
                self._rects.append((QtCore.QRectF(x, top - 4, w, h + 8), i))
                x += w
            p.restore()
        # legend: 2 x 2, swatch + name + value (+ share of J)
        y0 = top + h + 8
        col_w = W / 2
        for i, (name, v, _) in enumerate(self.terms):
            cx = (i % 2) * col_w
            cy = y0 + (i // 2) * (fm.height() + 3)
            p.fillRect(QtCore.QRectF(cx, cy + fm.ascent() / 2 - 4, 9, 9), QtGui.QColor(TERM_COLORS[i]))
            p.setPen(QtGui.QColor(INK_2))
            p.drawText(QtCore.QPointF(cx + 14, cy + fm.ascent()), name)
            share = f"{v:.3f}" + (f"  {100 * v / tot:.0f}%" if tot > 0 else "")
            p.setPen(QtGui.QColor(INK))
            p.drawText(QtCore.QPointF(cx + col_w - 8 - fm.horizontalAdvance(share), cy + fm.ascent()), share)
        p.end()

    def mouseMoveEvent(self, ev):
        for r, i in self._rects:
            if r.contains(ev.pos()):
                name, v, detail = self.terms[i]
                QtWidgets.QToolTip.showText(ev.globalPos(), f"<b>{name}</b>: {v:.4f}<br>{detail}", self)
                return
        QtWidgets.QToolTip.hideText()


class FleetBars(QtWidgets.QWidget):
    """One row per spacecraft: identity swatch, coverage it would get alone, orbit and cost."""

    ROW = 19

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_rows(self, rows):
        self.rows = list(rows)
        self.setFixedHeight(18 + self.ROW * max(len(self.rows), 1))
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        f = QtGui.QFont(self.font())
        f.setPixelSize(11)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        W = self.width()
        p.setPen(QtGui.QColor(INK_3))
        p.drawText(QtCore.QPointF(0, fm.ascent()), "Sunlit area each spacecraft would image alone")
        name_w, val_w = 34, 40
        text_w = min(190, int(W * 0.46))
        bar_x, bar_w = name_w + val_w, max(20, W - name_w - val_w - text_w - 8)
        for k, r in enumerate(self.rows):
            y = 18 + k * self.ROW
            mid = y + self.ROW / 2
            p.fillRect(QtCore.QRectF(0, mid - 4, 8, 8), QtGui.QColor(r["color"]))
            p.setPen(QtGui.QColor(INK_2))
            p.drawText(QtCore.QPointF(12, mid + fm.ascent() / 2 - 1), r["name"])
            p.setPen(QtGui.QColor(INK))
            v = f"{100 * r['alone']:.0f}%"
            p.drawText(QtCore.QPointF(bar_x - 6 - fm.horizontalAdvance(v), mid + fm.ascent() / 2 - 1), v)
            track = QtCore.QRectF(bar_x, mid - 3, bar_w, 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(TRACK))
            p.drawRoundedRect(track, 3, 3)
            p.setBrush(QtGui.QColor(r["color"]))
            p.drawRoundedRect(QtCore.QRectF(bar_x, mid - 3, max(bar_w * min(r["alone"], 1.0), 2), 6), 3, 3)
            p.setPen(QtGui.QColor(INK_2))
            txt = f"r {r['a']:.1f} km · ΔV {r['dv']:.1f} m/s · {r['frames']} fr"
            txt = fm.elidedText(txt, Qt.ElideRight, text_w)
            p.drawText(QtCore.QPointF(W - text_w, mid + fm.ascent() / 2 - 1), txt)
        if not self.rows:
            p.setPen(QtGui.QColor(INK_3))
            p.drawText(QtCore.QPointF(0, 18 + fm.ascent()), "no spacecraft")
        p.end()

    def mouseMoveEvent(self, ev):
        k = int((ev.pos().y() - 18) // self.ROW)
        if 0 <= k < len(self.rows) and ev.pos().y() >= 18:
            r = self.rows[k]
            QtWidgets.QToolTip.showText(
                ev.globalPos(),
                f"<b>{r['name']}</b><br>alone it images {100 * r['alone']:.1f} % of the sunlit area<br>"
                f"orbit radius {r['a']:.1f} km, inclination {r['inc']:.1f}°<br>"
                f"ΔV {r['dv']:.2f} m/s, {r['frames']} frames taken", self)
        else:
            QtWidgets.QToolTip.hideText()


class MetricsPanel(QtWidgets.QGroupBox):
    """Cost and constraints of the displayed design."""

    def __init__(self, parent=None):
        super().__init__("Displayed design", parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(8)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        self.tile_J = KpiTile("FITNESS J")
        self.tile_C = KpiTile("COVERAGE C")
        self.tile_N = KpiTile("FLEET N")
        self.tile_T = KpiTile("MISSION T")
        for k, t in enumerate((self.tile_J, self.tile_C, self.tile_N, self.tile_T)):
            grid.addWidget(t, 0, k)
            grid.setColumnStretch(k, 1)
        lay.addLayout(grid)
        self.cost = CostBar()
        lay.addWidget(self.cost)

        row = QtWidgets.QHBoxLayout()
        self.checks = _label("", 12, INK_2)
        self.checks.setTextFormat(Qt.RichText)
        self.checks.setWordWrap(True)
        row.addWidget(self.checks, 1)
        self.details_btn = QtWidgets.QToolButton()
        self.details_btn.setText("details")
        self.details_btn.setCheckable(True)
        self.details_btn.setToolTip("show every constraint and its margin (≤ 0 = satisfied)")
        row.addWidget(self.details_btn, 0, Qt.AlignTop)
        lay.addLayout(row)
        self.details = _label("", 11, INK_2)
        self.details.setTextFormat(Qt.RichText)
        self.details.setVisible(False)
        self.details_btn.toggled.connect(self.details.setVisible)
        lay.addWidget(self.details)

        self.fleet = FleetBars()
        lay.addWidget(self.fleet)
        self.footer = _label("", 11, INK_3)
        self.footer.setWordWrap(True)
        lay.addWidget(self.footer)

        self.waiting = _label("evaluating mission …", 12, INK_3)
        lay.addWidget(self.waiting)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("evaluating mission  %p%")
        self.progress.setMaximumHeight(14)
        self.progress.hide()
        lay.addWidget(self.progress)
        self._content = [self.cost, self.checks, self.details_btn, self.fleet, self.footer]

    def set_data(self, d: dict | None):
        self.waiting.setVisible(d is None)
        if d is None:
            for t in (self.tile_J, self.tile_C, self.tile_N, self.tile_T):
                t.value.setStyleSheet(t.value.styleSheet().replace(INK, INK_3))
            return
        for t in (self.tile_J, self.tile_C, self.tile_N, self.tile_T):
            t.value.setStyleSheet(t.value.styleSheet().replace(INK_3, INK))
        ok, bad = f"<span style='color:{GOOD}'>✓</span>", f"<span style='color:{BAD}'>✗</span>"
        if d["feasible"]:
            self.tile_J.set(f"{d['J']:.4f}", f"{ok} feasible",
                            "minimised: lower is better; the sum of the four terms below")
        else:
            self.tile_J.set(f"{d['J']:.0f}", f"{bad} <b>infeasible</b>",
                            "1000 + penalty: this design breaks a constraint and never reaches the engine; "
                            "it is drawn for reference")
        self.tile_C.set(f"{100 * d['C']:.1f}%", f"{d['S_a']:.0f}/{d['a_star']:.0f} km²",
                        f"{d['S_a']:.0f} of the {d['a_star']:.0f} km² that are sunlit at some time during the "
                        "mission were imaged (maximised)")
        self.tile_N.set(f"{d['N']} / {d['n_max']}", "flying",
                        "spacecraft slots switched on out of N_max; each costs 0.1 in Z")
        self.tile_T.set(f"{d['T_days']:.2f} d", f"of {d['t_max_days']:g} d",
                        f"mission duration T (a design variable), at most T_max = {d['t_max_days']:g} days")
        self.cost.set_terms(d["terms"], d["J"])
        self.cost.note = "" if d["feasible"] else "engine terms only; J adds the 1000 + penalty"

        violated = [(n, v) for n, v in d["checks"] if v > 0]
        sep = d["min_sep"]
        extras = []
        if np.isfinite(sep):
            extras.append(f"closest pair <b style='color:{INK}'>{sep:.1f} km</b> (≥ {d['d_sep']:g})")
        extras.append(f"lowest battery <b style='color:{INK}'>{100 * d['low_batt']:.0f} %</b> "
                      f"(≥ {100 * d['soc_min']:.0f} %)")
        if not d["checks"]:
            head = ""
        elif violated:
            names = ", ".join(f"{n} ({v:+.2f})" for n, v in violated)
            head = f"{bad} <b style='color:{BAD}'>{len(violated)} violated:</b> {names}<br>"
        else:
            head = f"{ok} all {len(d['checks'])} constraints met<br>"
        self.checks.setText(head + " · ".join(extras))
        rows = "".join(
            f"<tr><td width='18' style='color:{GOOD if v <= 0 else BAD}'>{'✓' if v <= 0 else '✗'}</td>"
            f"<td align='left'>{n}</td>"
            f"<td align='right' style='color:{INK if v <= 0 else BAD}'>{v:+.3f}</td></tr>"
            for n, v in d["checks"])
        head_row = (f"<tr><td></td><td style='color:{INK_3}'>constraint</td>"
                    f"<td align='right' style='color:{INK_3}'>margin (≤ 0 = met)</td></tr>")
        self.details.setText(f"<table width='100%' cellspacing='1'>{head_row}{rows}</table>"
                             if rows else "no constraint report")
        self.details_btn.setVisible(bool(d["checks"]))
        self.fleet.set_rows(d["fleet"])
        self.footer.setText(f"camera {d['camera']} · a frame every {d['dt_min']:g} min · {d['light']}")

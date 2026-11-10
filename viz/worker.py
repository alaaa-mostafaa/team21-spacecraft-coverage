"""Background threads so the UI never freezes:

* ``OrbitalEvaluator``   evaluates the displayed design: Feasibility -> Engine (a pipeline per
                         asteroid + config is kept, so a slider drag only re-simulates what changed)
* ``OptimiserRunner``    runs any registered algorithm on its own Pipeline, streaming its
                         progress to the live charts
"""
from __future__ import annotations

import copy
import time
import traceback
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

from spacecov import algorithms
from spacecov.physics import Cancelled
from spacecov.pipeline import Pipeline


class OrbitalEvaluator(QtCore.QObject):
    """Debounced, cancellable, one-at-a-time orbital evaluation.

    ``request()`` may be called on every slider tick: the newest request wins,
    a running evaluation is cancelled and restarted with the latest inputs.
    """
    started = pyqtSignal()
    progress = pyqtSignal(float)
    finished = pyqtSignal(object, int)       # (Evaluation | None, EngineResult | None), request id
    failed = pyqtSignal(str)

    def __init__(self, parent=None, debounce_ms: int = 250):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._launch)
        self._pending = None
        self._thread: _EvalJob | None = None
        self._alive: set[QtCore.QThread] = set()     # keep QThread objects alive until they exit
        self._request_id = 0
        self._pipe_key = None
        self._pipe: Pipeline | None = None           # reused while asteroid and config stay the same

    def request(self, design, asteroid, cfg):
        self._request_id += 1
        self._pending = (copy.deepcopy(design), asteroid, copy.deepcopy(cfg), self._request_id)
        if self._thread is not None:
            self._thread.cancel = True
        self._debounce.start()

    def _launch(self):
        if self._thread is not None or self._pending is None:
            return                                   # relaunched from _done when the old job exits
        design, ast, cfg, rid = self._pending
        self._pending = None
        key = (id(ast), repr(cfg.to_dict()))
        if key != self._pipe_key:                    # jobs run one at a time, so sharing is safe
            self._pipe_key, self._pipe = key, None
        job = _EvalJob(design, ast, cfg, rid, self)
        job.progress_sig.connect(self.progress)
        job.done_sig.connect(self._done)
        job.fail_sig.connect(self._failed)
        self._thread = job
        self._alive.add(job)
        job.finished.connect(lambda j=job: self._alive.discard(j))
        self.started.emit()
        job.start()

    def _done(self, result, rid):
        self._thread = None
        if result is not None and result[1] is not None and rid == self._request_id:
            self.finished.emit(result, rid)
        if self._pending is not None:
            self._launch()

    def _failed(self, msg):
        self._thread = None
        self.failed.emit(msg)
        if self._pending is not None:
            self._launch()

    def busy(self) -> bool:
        return self._thread is not None or self._pending is not None

    def shutdown(self):
        self._debounce.stop()
        self._pending = None
        for job in list(self._alive):
            job.cancel = True
            job.wait(5000)


class _EvalJob(QtCore.QThread):
    progress_sig = pyqtSignal(float)
    done_sig = pyqtSignal(object, int)
    fail_sig = pyqtSignal(str)

    def __init__(self, design, ast, cfg, rid, owner):
        super().__init__()
        self.design, self.ast, self.cfg, self.rid, self.owner = design, ast, cfg, rid, owner
        self.cancel = False

    def run(self):
        from .presenter import evaluate_design

        def prog(frac):
            self.progress_sig.emit(frac)
            return self.cancel
        try:
            if self.owner._pipe is None:
                self.owner._pipe = Pipeline(self.ast, self.cfg)
            self.done_sig.emit(evaluate_design(self.owner._pipe, self.design, prog), self.rid)
        except Cancelled:
            self.done_sig.emit(None, self.rid)
        except Exception:
            self.fail_sig.emit(traceback.format_exc())


# ===========================================================================
# Optimisation algorithms
# ===========================================================================
STEP_FIELDS = ("iteration", "temperature", "cost", "best_cost", "current_cost", "accepted", "feasible",
               "C", "T_days", "N", "C_best", "T_best", "N_best", "C_cur", "T_cur", "N_cur", "p_accept", "dJ",
               "generation", "operator")


@dataclass
class Move:
    """One step of the search, for the step-by-step visualisation."""
    iteration: int
    temperature: float
    x: np.ndarray                 # current design after the decision
    evaluation: object            # its Evaluation (Feasibility + Engine)
    candidate: np.ndarray         # proposed design
    slot: int                     # spacecraft changed by the move (-1: none / N or T only)
    description: str
    accepted: bool
    dJ: float
    p_accept: float
    cost: float
    best_cost: float


def _num(v) -> float:
    return float("nan") if v is None else float(v)


class OptimiserRunner(QtCore.QObject):
    """Runs ``algorithms.make(name, **params).run(pipeline)`` in a thread.

    steps     (rows of STEP_FIELDS, move descriptions), in batches ~20 times a second
    move      a Move snapshot of the current design: every move at a slow pace, ~5 per second at full speed
    finished  (OptimResult, Pipeline) - also after Stop
    """
    started = pyqtSignal(int)                # problem dimension
    steps = pyqtSignal(object, object)
    move = pyqtSignal(object)
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: _OptJob | None = None
        self._alive: set[QtCore.QThread] = set()
        self.delay = 0.0
        self.paused = False

    def start(self, asteroid, cfg, name: str, params: dict, x0: np.ndarray | None):
        if self.running():
            return
        job = _OptJob(asteroid, copy.deepcopy(cfg), name, dict(params), x0)
        job.delay, job.paused = self.delay, self.paused
        job.steps_sig.connect(self.steps)
        job.move_sig.connect(self.move)
        job.done_sig.connect(self._done)
        job.fail_sig.connect(self._failed)
        self._job = job
        self._alive.add(job)
        job.finished.connect(lambda j=job: self._alive.discard(j))
        self.started.emit(2 + 4 * cfg.n_max)
        job.start()

    def set_pace(self, seconds_per_move: float):
        self.delay = float(seconds_per_move)
        if self._job is not None:
            self._job.delay = self.delay

    def set_paused(self, on: bool):
        self.paused = bool(on)
        if self._job is not None:
            self._job.paused = self.paused

    def stop(self):
        if self._job is not None:
            self._job.paused = False
            self._job.cancel = True

    def running(self) -> bool:
        return self._job is not None

    def _done(self, res, pipe):
        self._job = None
        self.finished.emit(res, pipe)

    def _failed(self, msg):
        self._job = None
        self.failed.emit(msg)

    def shutdown(self):
        for job in list(self._alive):
            job.paused = False
            job.cancel = True
            job.wait(10000)


class _OptJob(QtCore.QThread):
    steps_sig = pyqtSignal(object, object)
    move_sig = pyqtSignal(object)
    done_sig = pyqtSignal(object, object)
    fail_sig = pyqtSignal(str)

    EMIT_EVERY = 0.05          # s, step batches
    MOVE_EVERY = 0.2           # s, current-design snapshots at full speed

    def __init__(self, ast, cfg, name, params, x0):
        super().__init__()
        self.ast, self.cfg, self.name, self.params, self.x0 = ast, cfg, name, params, x0
        self.cancel = False
        self.paused = False
        self.delay = 0.0

    def _wait(self, seconds: float):
        """Sleep for the pace delay and while paused, but react to Stop at once."""
        end = time.perf_counter() + seconds
        while not self.cancel and (self.paused or time.perf_counter() < end):
            time.sleep(0.02)

    def run(self):
        try:
            pipe = Pipeline(self.ast, self.cfg)
            algo = algorithms.make(self.name, **self.params)
            buf: list[tuple] = []
            descs: list[str] = []
            last_emit = last_move = 0.0

            def callback(step):
                nonlocal last_emit, last_move
                ex = step.extra
                info, best = step.evaluation.info(), step.best.info()
                cur_ev = ex.get("evaluation_current", step.evaluation)
                cur = cur_ev.info()
                x_cur = ex.get("x_current", step.x)
                x_prev = ex.get("x_prev", step.x)
                buf.append((step.iteration, _num(ex.get("T")), step.cost, step.best_cost, cur["cost"],
                            _num(step.accepted), info["feasible"], info["C"], info["T_days"], info["N"],
                            best["C"], best["T_days"], best["N"], cur["C"], cur["T_days"], cur["N"],
                            _num(ex.get("p_accept")), _num(ex.get("dJ")),
                            _num(ex.get("generation")), _num(ex.get("operator"))))
                slot, desc = pipe.space.describe_change(x_prev, step.x)
                if self.name == "ga":
                    from spacecov.algorithms.ga import OP_NAMES
                    operator = OP_NAMES.get(int(ex.get("operator", -1)), "operator")
                    verdict = "improved" if step.improved else "evaluated"
                    desc = f"gen {int(ex.get('generation', 0))} · {operator} · {verdict} · {desc}"
                if not step.evaluation.feasible:
                    desc += "  [infeasible: " + ", ".join(step.evaluation.report.violated) + "]"
                descs.append(desc)
                now = time.perf_counter()
                slow = self.delay > 0 or self.paused
                if slow or now - last_emit > self.EMIT_EVERY:
                    self.steps_sig.emit(np.array(buf, float), list(descs))
                    buf.clear()
                    descs.clear()
                    last_emit = now
                if slow or now - last_move > self.MOVE_EVERY:
                    self.move_sig.emit(Move(step.iteration, _num(ex.get("T")), np.asarray(x_cur).copy(), cur_ev,
                                            step.x.copy(), slot, desc,
                                            bool(step.accepted) if step.accepted is not None else True,
                                            _num(ex.get("dJ")), _num(ex.get("p_accept")), cur["cost"],
                                            step.best_cost))
                    last_move = now
                if slow:
                    self._wait(self.delay)
                return self.cancel

            x0 = None if self.x0 is None else np.asarray(self.x0, float)
            res = algo.run(pipe, x0, callback)
            if buf:
                self.steps_sig.emit(np.array(buf, float), list(descs))
            self.done_sig.emit(res, pipe)
        except Exception:
            self.fail_sig.emit(traceback.format_exc())

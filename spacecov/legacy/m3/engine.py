"""Engine: simulate a fleet and score it.

    fleet --> one ``physics.track`` per flying spacecraft (cached) --> objectives C, Z, E
          --> constraints g1-g4 (``feasibility``) --> fitness J

    C   coverage: share of the area that is sunlit at some time in the mission (A*) that was imaged
    Z   cost: sum of (bus + camera) costs / cost of n_ref = 10 spacecraft  (0.1 per spacecraft)
    E   energy: 1/2 Delta-V from the arrival orbit + 1/2 electrical energy, each / value of 10 spacecraft
    J = w1 (1 - C) + w2 Z + w3 E + rho sum max(0, g_j)^2        (to MINIMISE; 0 = perfect, 1 = worst)

A track depends only on one spacecraft's own variables, so ``Engine`` caches tracks: an optimiser
move that changes one slot re-simulates only that spacecraft.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .asteroid import Asteroid
from .config import BUS_COST, OrbitalConfig
from .design import OrbitalSolution, OrbitalSpacecraft
from .feasibility import constraints, is_feasible, penalty
from .physics import Cancelled, Track, illuminable, lit_mask, track


@dataclass
class OrbitalResult:
    # objectives (all normalised to [0, 1])
    C: float                         # coverage of the illuminable area (maximise)
    Z: float                         # cost (minimise)
    E: float                         # energy (minimise) = 0.5 E_dv + 0.5 E_elec
    E_dv: float
    E_elec: float
    S: float                         # score w1 C - w2 Z - w3 E
    g: np.ndarray                    # (4,) normalised violations g1..g4 (<= 0 feasible)
    J: float                         # fitness to MINIMISE
    n_active: int
    # mission data for analysis / animation (active spacecraft only)
    times: np.ndarray                # (K,) s
    r_inertial: np.ndarray           # (N, K, 3) km
    r_body: np.ndarray               # (N, K, 3) km
    first_seen: np.ndarray           # (N, F) int, first step facet f was imaged by i (K if never)
    m: np.ndarray                    # (F,) number of distinct spacecraft that imaged f
    u: np.ndarray                    # (F,) bool, imaged by anyone
    areas: np.ndarray                # (F,) facet areas
    a_star: float                    # illuminable area A* [km^2]
    illuminable: np.ndarray          # (F,) bool, facet is sunlit at least once (part of A*)
    soc: np.ndarray                  # (N, K) battery state of charge
    eclipse: np.ndarray              # (N, K) bool
    n_img: np.ndarray                # (N,) frames taken
    dv: np.ndarray                   # (N,) km/s
    e_elec: np.ndarray               # (N,) J
    min_separation: float
    r_min: float
    visible: np.ndarray | None = None    # (N, K, F) bool, None for optimiser evaluations
    eval_seconds: float = 0.0

    @property
    def feasible(self) -> bool:
        return is_feasible(self.g)

    @property
    def objectives(self) -> np.ndarray:
        """NSGA-II objective vector: [-C, Z, E], all minimised."""
        return np.array([-self.C, self.Z, self.E])

    def cumulative(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(m_f, per-spacecraft seen mask) using observations up to and including step k."""
        seen_i = self.first_seen <= k                                    # (N, F)
        return seen_i.sum(axis=0), seen_i

    def coverage_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """C(t_k) of the fleet (K,) and of each spacecraft alone (N, K), as fractions of A*."""
        K = len(self.times)

        def curve(fs):
            return np.cumsum(np.bincount(fs, weights=self.areas, minlength=K + 1)[:K]) / self.a_star

        return curve(self.first_seen.min(axis=0)), np.array([curve(fs) for fs in self.first_seen])

    def summary(self) -> str:
        g = ", ".join(f"{x:+.3f}" for x in self.g)
        return (f"N = {self.n_active}   C = {self.C:.3f}   Z = {self.Z:.3f}   E = {self.E:.3f}   "
                f"S = {self.S:.3f}   g = [{g}]   J = {self.J:.4f}   "
                f"({'feasible' if self.feasible else 'INFEASIBLE'})")


def combine(sol: OrbitalSolution, tracks: list[Track], ast: Asteroid, cfg: OrbitalConfig,
            times: np.ndarray, lit_any: np.ndarray, t0: float | None = None) -> OrbitalResult:
    """sol and tracks hold the ACTIVE spacecraft only; lit_any marks the facets of A*."""
    a_star = float(ast.areas[lit_any].sum())
    K, N = len(times), sol.n
    n_ref = max(cfg.n_ref, N)                                             # normalisation of Z and E
    first_seen = np.array([t.first_seen for t in tracks])                 # (N, F)
    v = first_seen < K
    u = v.any(axis=0)
    m = v.sum(axis=0)
    A = ast.areas
    C = float((A * u).sum() / a_star) if a_star > 0 else 0.0              # coverage of A*

    # reference: n_ref spacecraft with the most expensive camera allowed (6 deg only: Z = 0.1 per spacecraft)
    c_ref = BUS_COST + max(sc.cost for t, sc in cfg.scanners.items() if t <= cfg.max_type)
    Z = sum(cfg.unit_cost(s.type) for s in sol.spacecraft) / (n_ref * c_ref)
    dv = np.array([t.dv for t in tracks])
    e_el = np.array([t.e_elec for t in tracks])
    E_dv = float(dv.sum() / (n_ref * cfg.dv_ref))
    E_elec = float(e_el.sum() / (n_ref * cfg.e_ref_j))
    E = 0.5 * E_dv + 0.5 * E_elec
    w1, w2, w3 = cfg.weights
    S = w1 * C - w2 * Z - w3 * E

    r_I = np.array([t.r_inertial for t in tracks])
    soc = np.array([t.soc for t in tracks])
    r_min = cfg.resolve_r_min(ast.r_max)
    a = np.array([s.a for s in sol.spacecraft])
    e = np.array([s.e for s in sol.spacecraft])
    g, min_sep = constraints(a, e, r_I, soc, r_min, cfg)
    # J = w1 (1 - C) + w2 Z + w3 E + penalty = w1 - S + penalty: 0 = perfect, 1 = worst (feasible designs)
    J = w1 - S + penalty(g, cfg)

    visible = None
    if all(t.visible is not None for t in tracks):
        visible = np.array([t.visible for t in tracks])
    return OrbitalResult(C, float(Z), float(E), E_dv, E_elec, float(S), g, float(J), N, times, r_I,
                         np.array([t.r_body for t in tracks]), first_seen, m, u, A, a_star, lit_any, soc,
                         np.array([t.eclipse for t in tracks]), np.array([t.n_img for t in tracks]),
                         dv, e_el, min_sep, r_min, visible,
                         0.0 if t0 is None else time.perf_counter() - t0)


def evaluate(sol: OrbitalSolution, ast: Asteroid, cfg: OrbitalConfig | None = None,
             progress: Callable[[float], bool | None] | None = None,
             keep_visible: bool = True, lit: np.ndarray | None = None) -> OrbitalResult:
    """Full mission evaluation of the active spacecraft.  ``progress(frac)`` may return True to cancel."""
    t0 = time.perf_counter()
    cfg = cfg or OrbitalConfig()
    active = sol.active()
    if active.n == 0:
        raise ValueError("the fleet has no active spacecraft")
    times = cfg.times
    if lit is None:
        lit = lit_mask(ast, cfg, times)
    tracks = []
    for i, s in enumerate(active.spacecraft):
        tracks.append(track(s, ast, cfg, times, lit, keep_visible))
        if progress is not None and progress((i + 1) / active.n):
            raise Cancelled
    return combine(active, tracks, ast, cfg, times, illuminable(ast, lit), t0)


class Engine:
    """Evaluates solutions while reusing the stage 1-3 track of every spacecraft seen before.

    Simulated annealing changes one slot per move, so a move costs one track instead of N.
    The cache is keyed by the spacecraft's exact seven variables.  The sunlight mask and the
    illuminable area do not depend on the design and are computed once."""

    def __init__(self, ast: Asteroid, cfg: OrbitalConfig, max_tracks: int = 96):
        self.ast, self.cfg = ast, cfg
        self.times = cfg.times
        self.lit = lit_mask(ast, cfg, self.times)
        self.lit_any = illuminable(ast, self.lit)
        self.a_star = float(ast.areas[self.lit_any].sum())
        self.max_tracks = max_tracks
        self._cache: OrderedDict[tuple, Track] = OrderedDict()
        self.n_evaluations = 0          # full fitness evaluations requested
        self.n_tracks = 0               # spacecraft actually simulated

    def _track(self, s: OrbitalSpacecraft) -> Track:
        k = s.key()
        t = self._cache.get(k)
        if t is None:
            t = track(s, self.ast, self.cfg, self.times, self.lit)
            self.n_tracks += 1
            self._cache[k] = t
            if len(self._cache) > self.max_tracks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(k)
        return t

    def evaluate(self, sol: OrbitalSolution) -> OrbitalResult:
        t0 = time.perf_counter()
        self.n_evaluations += 1
        active = sol.active()
        tracks = [self._track(s) for s in active.spacecraft]
        return combine(active, tracks, self.ast, self.cfg, self.times, self.lit_any, t0)

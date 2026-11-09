"""Engine: simulate a (feasible) design and compute its fitness J.

    J(x) = w_C (1 - C)  +  w_Z Z  +  w_E E  +  w_T T / T_max                  (minimise)

    C        coverage: area imaged within T / A*            -> MAXIMISED (enters as 1 - C, the share missed)
    Z        fleet cost: N (bus + camera) / (n_ref (bus + camera)) = N / 10      -> minimised
    E        energy: 1/2 sum Delta-V / (n_ref 3 v0) + 1/2 sum E_elec / (n_ref E_ref)   -> minimised
             E_elec,i = P_base T + frames_i (e_img + e_tx), E_ref = (P_base + P_cam + P_tx) T_max
    T/T_max  mission duration chosen by the algorithm                            -> minimised
    A*       area that is sunlit at some time within T_max (the most that could be imaged)

Every term lies in [0, 1] and the weights add to 1, so a feasible J lies in [0, 1]:
0 = everything imaged instantly at no cost, 1 = the worst.  Coverage enters as 1 - C, not 1/C:
1/C is infinite for an empty map and makes the last percentages of coverage almost worthless.

How the simulation is organised: every flying spacecraft is simulated on its own
("track", spacecov.physics stages 1-3) over the whole horizon T_max, and the track is
cached by the spacecraft's four orbit variables.  A design with duration T then just
uses the steps t_k <= T of its tracks, so changing T or switching a slot on/off costs no new
simulation, and moving one orbit costs one.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import physics
from .asteroid import Asteroid
from .config import MissionConfig
from .design import CircularOrbit, Design


@dataclass
class CostTerms:
    C: float               # coverage of A* (maximised: enters as 1 - C)
    Z: float               # fleet cost (minimised)
    E: float               # energy = 0.5 E_dv + 0.5 E_elec (minimised)
    E_dv: float
    E_elec: float
    f_time: float          # T / T_max (minimised)
    J: float               # weighted fitness to minimise

    def as_dict(self) -> dict:
        return {"C": self.C, "Z": self.Z, "E": self.E, "E_dv": self.E_dv, "E_elec": self.E_elec,
                "f_time": self.f_time, "J": self.J}


@dataclass
class EngineResult:
    N: int
    T: float                         # mission duration [s]
    S_a: float                       # area imaged within T [km^2]
    a_star: float                    # sunlit area over T_max [km^2]
    cost: CostTerms
    # mission data, steps t_k <= T only, flying spacecraft only
    times: np.ndarray                # (K,) s
    r_inertial: np.ndarray           # (N, K, 3) km
    r_body: np.ndarray               # (N, K, 3) km
    first_seen: np.ndarray           # (N, F) first step facet f was imaged by i (K if not within T)
    m: np.ndarray                    # (F,) number of distinct spacecraft that imaged f
    u: np.ndarray                    # (F,) imaged by anyone
    areas: np.ndarray                # (F,) facet areas
    illuminable: np.ndarray          # (F,) sunlit at least once within T_max (part of A*)
    soc: np.ndarray                  # (N, K) battery state of charge
    eclipse: np.ndarray              # (N, K) spacecraft in Eros' shadow
    n_img: np.ndarray                # (N,) frames taken within T
    dv: np.ndarray                   # (N,) km/s, Delta-V from the arrival orbit
    e_elec: np.ndarray               # (N,) J, electrical energy used within T
    visible: np.ndarray | None = None    # (N, K, F) only when asked for
    eval_seconds: float = 0.0

    @property
    def C(self) -> float:
        """Coverage: fraction of A* imaged within T."""
        return self.S_a / self.a_star if self.a_star > 0 else 0.0

    @property
    def J(self) -> float:
        return self.cost.J

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
        c = self.cost
        return (f"N = {self.N}   T = {self.T / 86400:.2f} d   C = {100 * self.C:.1f} % "
                f"({self.S_a:.0f} / {self.a_star:.0f} km2)   Z = {c.Z:.3f}   E = {c.E:.3f}   "
                f"T/Tmax = {c.f_time:.3f}   J = {c.J:.4f}")


class Engine:
    """Simulates designs on one asteroid with one configuration; keeps a cache of tracks."""

    def __init__(self, asteroid: Asteroid, cfg: MissionConfig, max_tracks: int = 1024):
        self.ast, self.cfg = asteroid, cfg
        self.times = cfg.times                                   # grid over T_max
        self.lit = physics.lit_mask(asteroid, cfg, self.times)   # independent of the design
        self.lit_any = physics.illuminable(asteroid, self.lit)
        self.a_star = float(asteroid.areas[self.lit_any].sum())
        self.max_tracks = max_tracks    # ~0.1 MB per track (10-day horizon): a GA population holds up to
                                        # P * N_max orbits (200-600), so 256 would evict parents' tracks
        self._cache: OrderedDict[tuple, physics.Track] = OrderedDict()
        self.n_simulations = 0          # designs simulated
        self.n_tracks = 0               # spacecraft actually propagated + imaged

    # ------------------------------------------------------------------ fitness
    def cost(self, N: int, T: float, S_a: float, dv: np.ndarray, e_elec: np.ndarray) -> CostTerms:
        """Objectives and the weighted fitness J of a fleet of N spacecraft flying for T seconds."""
        cfg = self.cfg
        n_ref = max(cfg.n_ref, N)
        C = S_a / self.a_star if self.a_star > 0 else 0.0
        Z = N * cfg.unit_cost() / (n_ref * cfg.unit_cost())
        E_dv = float(np.sum(dv) / (n_ref * cfg.dv_ref))
        E_elec = float(np.sum(e_elec) / (n_ref * cfg.e_ref_j))
        E = 0.5 * E_dv + 0.5 * E_elec
        f_time = T / cfg.t_max
        w_c, w_z, w_e, w_t = cfg.weights
        J = w_c * (1.0 - C) + w_z * Z + w_e * E + w_t * f_time
        return CostTerms(float(C), float(Z), float(E), E_dv, E_elec, float(f_time), float(J))

    # ------------------------------------------------------------------ simulation
    def track(self, orbit: CircularOrbit, keep_visible: bool = False) -> physics.Track:
        if keep_visible:                    # (K, F) visibility is large: never cached
            return physics.track(orbit.spacecraft(self.cfg.camera), self.ast, self.cfg, self.times, self.lit,
                                 keep_visible=True)
        k = orbit.key()
        t = self._cache.get(k)
        if t is None:
            t = physics.track(orbit.spacecraft(self.cfg.camera), self.ast, self.cfg, self.times, self.lit)
            self.n_tracks += 1
            self._cache[k] = t
            if len(self._cache) > self.max_tracks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(k)
        return t

    def simulate(self, design: Design, keep_visible: bool = False,
                 progress: Callable[[float], bool | None] | None = None) -> EngineResult:
        """Simulate the steps t_k <= T of the design.  ``progress(frac)`` may return True to cancel."""
        t0 = time.perf_counter()
        if design.N == 0:
            raise ValueError("the design has no spacecraft")
        self.n_simulations += 1
        K = self.cfg.steps_within(design.T)
        tracks = []
        for i, o in enumerate(design.orbits):
            tracks.append(self.track(o, keep_visible))
            if progress is not None and progress((i + 1) / design.N):
                raise physics.Cancelled
        first = np.array([np.minimum(t.first_seen, K) for t in tracks])   # (N, F), K = not within T
        u = (first < K).any(axis=0)
        m = (first < K).sum(axis=0)
        S_a = float(self.ast.areas[u].sum())
        n_img = np.array([int(t.imaging[:K].sum()) for t in tracks])
        dv = np.array([t.dv for t in tracks])
        cfg = self.cfg
        e_elec = cfg.p_base_w * design.T + n_img * (cfg.e_img_j + cfg.e_tx_j)
        visible = np.array([t.visible[:K] for t in tracks]) if keep_visible else None
        return EngineResult(
            N=design.N, T=design.T, S_a=S_a, a_star=self.a_star, cost=self.cost(design.N, design.T, S_a, dv, e_elec),
            times=self.times[:K], r_inertial=np.array([t.r_inertial[:K] for t in tracks]),
            r_body=np.array([t.r_body[:K] for t in tracks]), first_seen=first, m=m, u=u, areas=self.ast.areas,
            illuminable=self.lit_any, soc=np.array([t.soc[:K] for t in tracks]),
            eclipse=np.array([t.eclipse[:K] for t in tracks]),
            n_img=n_img, dv=dv, e_elec=e_elec, visible=visible, eval_seconds=time.perf_counter() - t0)

"""Presenter: turns a Scenario (design + evaluation + mission time) into scene updates.

Spacecraft positions are computed from Kepler's equation at the exact displayed time
(not snapped to the evaluation steps), and the asteroid is rotated by omega * t with
the same clock, so orbital speeds and the spin are shown in their true ratio.

Qt-free on purpose: the GUI (app.py) and the off-screen snapshot tool both drive it.
"""
from __future__ import annotations

import copy

import numpy as np

from scipy.spatial import cKDTree

from spacecov import physics
from spacecov.asteroid import PRESETS, Asteroid, display_spec, make_asteroid
from spacecov.config import SPACECRAFT_SPAN_M, for_body
from spacecov.design import CircularOrbit, Design
from spacecov.geometry import frame_from_direction
from spacecov.engine import EngineResult
from spacecov.feasibility import LABELS, POST_CHECKS, PRE_CHECKS
from spacecov.physics import OrbitalSolution
from spacecov.pipeline import Evaluation, Pipeline
from spacecov.scenario import Scenario, default_design

from . import colors
from .scene import SceneManager, SpacecraftPose
from .spacecraft_model import cone_directions

TRAIL_HOURS = 3.0
TRAIL_POINTS = 90
TRACK_STEP_S = 120.0


class Presenter:
    def __init__(self, scene: SceneManager):
        self.scene = scene
        self.scenario = Scenario()
        self.asteroid: Asteroid | None = None     # mesh the objectives are computed on
        self.display: Asteroid | None = None      # mesh that is drawn (full 49,152-plate Eros)
        self.fine_map: np.ndarray | None = None   # display face -> computation facet
        self.imaged_km2: list[float] = []         # area each scanner images right now
        self.shadow_fine: np.ndarray | None = None
        self.banner_override: tuple[str, tuple] | None = None     # SA move text while the search runs
        self._r_inertial_now = np.zeros((0, 3))
        self.frame = "inertial"            # inertial (asteroid spins) | body (asteroid fixed)
        self.t = 0.0                       # mission time [s]
        self.color_mode = "Coverage"
        self.highlight_current = True
        self.result: EngineResult | None = None    # what is drawn (also for an infeasible design)
        self.evaluation: Evaluation | None = None  # Feasibility + Engine verdict of the design
        self.pipeline: Pipeline | None = None      # for synchronous evaluation (snapshot tool, tests)
        self.stale = True
        self._color_key = None
        self._lines_key = None
        self.poses: list[SpacecraftPose] = []

    # ================================================================= loading
    @property
    def cfg(self):
        return self.scenario.config

    @property
    def design(self) -> Design:
        return self.scenario.design

    @property
    def fleet(self) -> OrbitalSolution:
        """The flying spacecraft as general orbits (circular, the configured camera)."""
        return self.design.fleet(self.cfg.camera)

    def load(self, scenario: Scenario):
        self.scenario = copy.deepcopy(scenario)
        if self.scenario.design is None:
            self.scenario.design = default_design(min(3, self.cfg.n_max))
        self.set_asteroid_spec(self.scenario.asteroid, render=False)
        self.t = 0.0
        self.refresh(full=True)
        self.scene.fit(max(s.apoapsis for s in self.fleet.spacecraft))
        self.scene.reset_camera()

    def set_asteroid_spec(self, spec: dict, render: bool = True):
        self.scenario.asteroid = dict(spec)
        self.asteroid = make_asteroid(spec)
        self.display = make_asteroid(display_spec(spec))
        self.fine_map = cKDTree(self.asteroid.centers).query(self.display.centers)[1]
        self._base_rgb = colors.natural(self.display.n_facets)
        c, n = self.display.centers, self.display.normals
        self._fine_c32, self._fine_n32 = c.astype(np.float32), n.astype(np.float32)
        self._fine_cc = np.einsum("ij,ij->i", c, c).astype(np.float32)   # |c|^2 (footprint tests -> mat-vecs)
        self._fine_nc = np.einsum("ij,ij->i", n, c).astype(np.float32)   # n . c
        self.scene.set_asteroid(self.display)
        self.pipeline = None
        self.set_result(None)
        if render:
            self.refresh(full=True)

    def set_body(self, name: str, spec: dict | None = None):
        """Switch the target body: its constants (GM, spin, resolution, orbit band), its mesh, and a
        hand-made starting design that fits its orbit band."""
        self.scenario.config = for_body(name, self.cfg)
        self.set_asteroid_spec(spec or dict(PRESETS[name]), render=False)
        r_min = self.cfg.resolve_r_min(self.asteroid.r_max)
        n = min(max(self.design.N, 1), self.cfg.n_max)
        self.scenario.design = default_design(n, self.design.T_days, a=1.7 * r_min)
        self.t = min(self.t, self.duration)
        self._lines_key = None
        self.refresh(full=True)
        self.scene.fit(max(s.apoapsis for s in self.fleet.spacecraft))
        self.scene.reset_camera()

    # ================================================================= design editing
    def update_spacecraft(self, i: int, **values):
        o = self.design.orbits[i]
        for k, v in values.items():
            setattr(o, k, float(v))
        self.design_changed()

    def add_spacecraft(self) -> int:
        """Add a spacecraft (up to N_max) on an orbit spread from the others."""
        i = self.design.N
        if i >= self.cfg.n_max:
            return i - 1
        self.design.orbits.append(CircularOrbit(a=45.0 + 5.0 * (i % 2), inc=45.0 + 20 * i, raan=(97.0 * i) % 360,
                                                M0=(137.5 * i) % 360))
        self.design_changed()
        return i

    def remove_spacecraft(self, i: int):
        if self.design.N > 1:
            self.design.orbits.pop(i)
            self.design_changed()

    def set_mission_time(self, T_days: float):
        self.design.T_days = float(T_days)
        self.t = min(self.t, self.duration)
        self.design_changed()

    def set_design(self, design: Design, evaluation: Evaluation | None = None, result: EngineResult | None = None):
        """Show an externally produced design (e.g. the one the optimiser is on)."""
        self.scenario.design = copy.deepcopy(design)
        self.t = min(self.t, self.duration)
        self.set_result(result if result is not None else (evaluation.result if evaluation else None), evaluation)
        self._lines_key = None
        self.refresh()

    def design_changed(self):
        self.set_result(None)
        self._lines_key = None
        self.refresh()

    # ================================================================= evaluation
    def get_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            self.pipeline = Pipeline(self.asteroid, self.cfg)
        return self.pipeline

    def evaluate(self, progress=None) -> EngineResult:
        """Synchronous evaluation (the GUI runs ``evaluate_design`` in a worker thread instead)."""
        ev, res = evaluate_design(self.get_pipeline(), self.design, progress)
        self.set_result(res, ev)
        return res

    def set_result(self, res: EngineResult | None, evaluation: Evaluation | None = None):
        self.result = res
        self.evaluation = evaluation
        self.stale = res is None
        self._color_key = None

    @property
    def n_steps(self) -> int:
        """Steps of the displayed mission (t_k <= T)."""
        return self.cfg.steps_within(self.design.T)

    @property
    def duration(self) -> float:
        return (self.n_steps - 1) * self.cfg.dt

    def current_step(self) -> int:
        return int(np.clip(np.floor(self.t / self.cfg.dt + 1e-9), 0, self.n_steps - 1))

    def config_changed(self):
        self.pipeline = None
        self.set_result(None)
        self._lines_key = None
        self.t = min(self.t, self.duration)
        self.refresh(full=True)

    # ================================================================= scene refresh
    def refresh(self, full: bool = False):
        """Push the current state to the scene.  ``full`` also rebuilds fixed geometry."""
        sc, cfg = self.scene, self.cfg
        if full:
            sc.set_safety(cfg.resolve_r_min(self.asteroid.r_max), cfg.r_max)
            sc.set_legend(self.color_mode)
            self._lines_key = None
        R_ast = physics.body_rotation(self.t, cfg.omega_ast)       # body -> inertial
        sun = cfg.sun_inertial
        if self.frame == "inertial":
            sc.set_asteroid_rotation(R_ast)
            sc.set_sun(sun)
        else:
            sc.set_asteroid_rotation(None)
            sc.set_sun(R_ast.T @ sun)
        self._refresh_lines()
        self.poses, r_now = self._poses()
        self._r_now = r_now
        self._r_inertial_now = physics.propagate(self.fleet, np.array([self.t]), cfg.mu)[:, 0]
        sc.set_spacecraft(self.poses)
        sc.set_trails(self._trails())
        links = [(r_now[i], r_now[j]) for i in range(len(r_now)) for j in range(i + 1, len(r_now))
                 if np.linalg.norm(r_now[i] - r_now[j]) < cfg.d_sep]
        sc.set_links(links)
        self._refresh_colors()
        sc.set_hud(self.hud_text())
        if self.banner_override is not None:
            sc.set_banner(*self.banner_override)
        else:
            sc.set_banner(self.banner_text())

    def _refresh_lines(self):
        key = (self.frame, tuple(o.key() for o in self.design.orbits), self.duration, self.cfg.mu)
        if key == self._lines_key:
            return
        self._lines_key = key
        if self.frame == "inertial":
            self.scene.set_orbits([physics.orbit_curve(s) for s in self.fleet.spacecraft])
            self.scene.set_tracks(None)
        else:
            self.scene.set_orbits(None)
            times = np.arange(0.0, self.duration + TRACK_STEP_S, TRACK_STEP_S)
            self.scene.set_tracks(list(self._positions(times)))

    def set_time(self, t: float):
        self.t = float(np.clip(t, 0.0, self.duration))
        self.refresh()

    # ----------------------------------------------------------------- poses
    def _cone_rim(self, pos_body: np.ndarray, R_body: np.ndarray, fov: float, l_max: float) -> np.ndarray:
        """Body-frame end points of the cone's edge rays: the surface hit or the usable range, whichever is nearer."""
        dirs = cone_directions(fov, R_body)
        ends = pos_body + dirs * l_max
        try:
            loc, idx, _ = self.asteroid.ray.intersects_location(np.repeat(pos_body[None], len(dirs), 0), dirs,
                                                               multiple_hits=False)
        except Exception:
            loc, idx = [], []
        if len(loc):
            dist = np.linalg.norm(loc - pos_body, axis=1)
            best = np.full(len(dirs), np.inf)
            np.minimum.at(best, idx, dist)
            hit = best < l_max                              # beyond l_max the surface is too coarse to count
            ends[hit] = pos_body + dirs[hit] * best[hit, None]
        return ends

    def _positions(self, times: np.ndarray) -> np.ndarray:
        """(N, K, 3) positions in the displayed frame."""
        r = physics.propagate(self.fleet, times, self.cfg.mu)
        if self.frame == "body":
            r = physics.to_body(r, times, self.cfg.omega_ast)
        return r

    def _poses(self):
        cfg = self.cfg
        r = self._positions(np.array([self.t]))[:, 0]
        poses = []
        Rb = physics.body_rotation(self.t, cfg.omega_ast).T              # inertial -> body
        for i, s in enumerate(self.fleet.spacecraft):
            normal = s.rotation()[:, 2]
            if self.frame == "body":
                normal = Rb @ normal
            R = frame_from_direction(-r[i], up_hint=normal)                     # nadir pointing
            fov = cfg.fov(s.type)
            to_body = Rb if self.frame == "inertial" else np.eye(3)
            rim = self._cone_rim(to_body @ r[i], to_body @ R, fov, cfg.max_range(s.type)) @ to_body
            poses.append(SpacecraftPose(r[i], R, fov, rim, f"SC{i + 1}"))
        return poses, r

    def _trails(self):
        times = np.linspace(max(0.0, self.t - TRAIL_HOURS * 3600), self.t, TRAIL_POINTS)
        return list(self._positions(times))

    # ----------------------------------------------------------------- colours
    def coverage_state(self):
        """(m_f, seen (N,F)) on the computation mesh up to the current time, or Nones."""
        r = self.result
        if r is None or self.stale or r.first_seen.shape[1] != self.asteroid.n_facets:
            return None, None
        k = min(self.current_step(), len(r.times) - 1)
        return r.cumulative(k)

    def footprints(self) -> tuple[np.ndarray, np.ndarray]:
        """What each scanner sees at the exact displayed time, on the drawn (fine) mesh.

        in_view: inside the FOV cone (a) and facing the spacecraft.
        imaged:  also passes emission (b), resolution (d), sunlight (e) and occlusion (c).
        The cheap tests run on the fine mesh (crisp edges); occlusion is taken from the
        computation mesh, where the ray casting is done."""
        cfg, fine, comp = self.cfg, self.display, self.asteroid
        t = np.array([self.t])
        r_body = physics.to_body(physics.propagate(self.fleet, t, cfg.mu), t, cfg.omega_ast)[:, 0]
        c, n = self._fine_c32, self._fine_n32
        lit_fine = None
        lit_comp = physics.lit_mask(comp, cfg, t)
        self.shadow_fine = None
        if cfg.lighting:
            facing = n @ physics.sun_body(t, cfg)[0].astype(np.float32) >= np.cos(
                np.deg2rad(cfg.incidence_max_deg))
            lit_fine = facing & lit_comp[0][self.fine_map] if cfg.shadows else facing
            self.shadow_fine = facing & ~lit_fine          # facing the Sun but in a cast shadow
        cos_eta = np.cos(np.deg2rad(cfg.eta_max_deg))
        no_occ = copy.copy(cfg)
        no_occ.occlusion = False
        N, F = self.fleet.n, fine.n_facets
        imaged, in_view = np.zeros((N, F), bool), np.zeros((N, F), bool)
        for i, s in enumerate(self.fleet.spacecraft):
            r = r_body[i].astype(np.float32)
            rn = float(np.linalg.norm(r))
            cr = c @ r
            ell = np.sqrt(np.maximum(rn * rn - 2.0 * cr + self._fine_cc, 1e-12))   # |r - c|
            emission = (n @ r - self._fine_nc) / ell                               # n . q_hat
            off_axis = (rn - cr / rn) / ell                                        # boresight . (-q_hat)
            in_view[i] = (off_axis >= np.cos(cfg.fov(s.type) / 2)) & (emission > 0)
            ok = in_view[i] & (emission >= cos_eta) & (ell <= cfg.max_range(s.type))
            if lit_fine is not None:
                ok &= lit_fine
            if cfg.occlusion and ok.any():
                # occlusion on the computation mesh: facets that pass every other test but whose
                # line of sight is blocked
                geo = np.flatnonzero(physics.visibility_track(r[None], s.type, comp, no_occ, lit_comp)[0])
                if len(geo):
                    d = r - comp.centers[geo]
                    d /= np.linalg.norm(d, axis=1, keepdims=True)
                    hidden = comp.ray.intersects_any(comp.centers[geo] + 1e-4 * comp.r_max * comp.normals[geo], d)
                    blocked = np.zeros(comp.n_facets, bool)
                    blocked[geo[hidden]] = True
                    ok &= ~blocked[self.fine_map]
            imaged[i] = ok
        self.imaged_km2 = [float(fine.areas[m].sum()) for m in imaged]
        return imaged, in_view

    def _refresh_colors(self):
        live = self.highlight_current and self.color_mode != "Natural"
        fleet_key = tuple(o.key() for o in self.design.orbits)
        key = (self.color_mode, self.current_step(), live and self.t, id(self.result), self.stale, fleet_key)
        if key == self._color_key:
            return
        self._color_key = key
        base_key = key[:2] + key[3:]                       # coverage only changes once per time step
        if base_key != getattr(self, "_painted_key", None):
            self._painted_key = base_key
            k, seen = self.coverage_state()
            never = None
            if k is not None:
                k, seen = k[self.fine_map], seen[:, self.fine_map]
                never = ~self.result.illuminable[self.fine_map]
            self._painted = colors.paint(self.color_mode, k, seen, self._base_rgb, never).astype(np.float32)
        rgb = self._painted
        if live:
            imaged, in_view = self.footprints()
            if self.shadow_fine is not None and self.shadow_fine.any():
                rgb = rgb.copy()
                rgb[self.shadow_fine] *= 0.3                     # cast shadows of the terrain
            rgb = colors.footprint(rgb, imaged, in_view)
        else:
            self.imaged_km2 = []
        self.scene.set_facet_colors(rgb if rgb.dtype == np.uint8 else colors._u8(rgb))

    def set_color_mode(self, mode: str):
        self.color_mode = mode
        self.scene.set_legend(mode)
        self._color_key = None
        self.refresh()

    # ================================================================= numbers
    def mission_clock(self) -> str:
        day, rem = divmod(int(self.t), 86400)
        return f"Day {day + 1}  {rem // 3600:02d}:{(rem % 3600) // 60:02d}"

    def coverage_now(self) -> float | None:
        k, _ = self.coverage_state()
        if k is None:
            return None
        return float(self.asteroid.areas[k >= 1].sum() / self.result.a_star)

    def asteroid_dims(self) -> np.ndarray:
        v = self.display.vertices
        return v.max(axis=0) - v.min(axis=0)

    def banner_text(self) -> str:
        """The live coverage number shown big at the top of the 3D view."""
        c = self.coverage_now()
        clock = f"t = {self.t / 3600:.1f} h of T = {self.duration / 3600:.1f} h"
        if c is None:
            return f"evaluating mission ...   {clock}"
        r = self.result
        tail = f"(will reach {100 * r.C:.1f} % by the end, T = {self.design.T_days:.2f} d)"
        if self.evaluation is not None and not self.evaluation.feasible:
            tail += "   INFEASIBLE"
        return (f"IMAGED {100 * c:.1f} % of the sunlit area ({c * r.a_star:.0f} / {r.a_star:.0f} km2)"
                f"    {clock}    {tail}")

    def hud_text(self) -> str:
        ast, cfg = self.display, self.cfg
        dx, dy, dz = self.asteroid_dims()
        rev = self.t / (cfg.p_rot_h * 3600)
        lines = [f"{ast.name}  {ast.n_facets} plates, extent {dx:.1f} x {dy:.1f} x {dz:.1f} km, "
                 f"area {ast.total_area:.0f} km2   spin {cfg.p_rot_h:g} h   N = {self.fleet.n}   "
                 f"{self.frame} frame   (everything to true scale)",
                 f"{self.mission_clock()}   t = {self.t / 3600:5.1f} h of {self.duration / 3600:g} h   "
                 f"asteroid turned {rev:4.2f} rev"]
        r = self.result
        r_now = self._r_now
        k = self.current_step()
        sun = cfg.sun_inertial
        for i, s in enumerate(self.fleet.spacecraft):
            d = float(np.linalg.norm(r_now[i]))
            now = f"   imaging {self.imaged_km2[i]:5.1f} km2" if i < len(self.imaged_km2) else ""
            r_in = self._r_inertial_now[i]
            proj = float(r_in @ sun)
            dark = proj < 0 and np.linalg.norm(r_in - proj * sun) < cfg.eclipse_radius_km
            power = ""
            if r is not None and not self.stale and i < len(r.soc) and k < r.soc.shape[1]:
                power = f"   battery {100 * r.soc[i, k]:3.0f} %"
            power += "  ECLIPSE" if dark else ""
            sc_ = cfg.scanners[s.type]
            lines.append(f"SC{i + 1}  {sc_.name} {sc_.fov_deg:g} deg (usable to {cfg.max_range(s.type):.0f} km)"
                         f"   r {d:5.1f} km   v {1000 * s.speed(d, cfg.mu):4.2f} m/s"
                         f"   period {s.period(cfg.mu) / 3600:4.1f} h{now}{power}")
        lines.append(f"spacecraft are {SPACECRAFT_SPAN_M:g} m across - dots and labels are screen markers")
        return "\n".join(lines)

    def metrics_html(self) -> str:
        ok = "<span style='color:#5fd38a'>&#10003;</span>"
        bad = "<span style='color:#ff5c5c'>&#10007;</span>"

        def row(name, val, extra=""):
            return f"<tr><td>{name}</td><td align='right'><b>{val}</b></td><td>&nbsp;{extra}</td></tr>"

        r, cfg, ev = self.result, self.cfg, self.evaluation
        if r is None or self.stale:
            return "<i>evaluating mission ...</i>"
        c = r.cost
        w_c, w_z, w_e, w_t = cfg.weights
        verdict = ok + " feasible" if ev is None or ev.feasible else bad + " <b>INFEASIBLE</b> (shown for reference)"
        cost = c.J if ev is None or ev.feasible else ev.cost
        rows = [row("<b>Fitness J</b>", f"{cost:.4f}", f"min &nbsp; {verdict}"),
                row("&nbsp;missed 1 &minus; C", f"{1 - c.C:.3f}",
                    f"&times; {w_c:.2f} &nbsp; C = {100 * c.C:.1f} % (max): {r.S_a:.0f} of {r.a_star:.0f} km&sup2;"),
                row("&nbsp;fleet cost Z", f"{c.Z:.3f}", f"&times; {w_z:.2f} &nbsp; min: N = {r.N} of {cfg.n_max} slots"),
                row("&nbsp;energy E", f"{c.E:.3f}",
                    f"&times; {w_e:.2f} &nbsp; min: &frac12; &Delta;V {c.E_dv:.3f} + &frac12; elec. {c.E_elec:.3f}"),
                row("&nbsp;time T / T_max", f"{c.f_time:.3f}",
                    f"&times; {w_t:.2f} &nbsp; min: T = {r.T / 86400:.2f} d of {cfg.t_max_days:g}")]
        if ev is not None:
            for name in PRE_CHECKS + POST_CHECKS:
                v = ev.report.violations.get(name)
                if v is not None:
                    rows.append(row(LABELS.get(name, name), f"{v:+.3f}", ok if v <= 0 else bad))
            sep = ev.report.min_separation
            rows.append(row("min separation", f"{sep:.1f} km" if np.isfinite(sep) else "—", f"&ge; {cfg.d_sep:g}"))
        rows.append(row("lowest battery", f"{100 * r.soc.min():.0f} %", f"&ge; {100 * cfg.soc_min:.0f} %"))
        light = f"Sun lat {cfg.sun_lat_deg:g}&deg;" if cfg.lighting else "off"
        rows.append(row("camera", cfg.scanners[cfg.camera].name, f"frame every {cfg.dt_min:g} min, light {light}"))
        A = self.asteroid.areas
        v = r.first_seen < len(r.times)
        per = "".join(
            f"<tr><td style='color:{colors.sc_color_hex(i)}'>SC{i + 1}</td>"
            f"<td align='right'>{(A * v[i]).sum() / r.a_star:.3f}</td>"
            f"<td>&nbsp;r {o.a:.1f} km, &Delta;V {1000 * r.dv[i]:.2f} m/s, {r.n_img[i]} frames</td></tr>"
            for i, o in enumerate(self.design.orbits) if i < len(v))
        head = "<tr><td colspan=3><br><i>coverage alone &middot; radius &middot; &Delta;V &middot; frames</i></td></tr>"
        return f"<table width='100%'>{''.join(rows)}{head}{per}</table>"


def evaluate_design(pipe: Pipeline, design: Design, progress=None) -> tuple[Evaluation, EngineResult | None]:
    """Algorithm-free evaluation for display: Feasibility -> Engine, as the optimiser sees it.

    If feasibility rejects the design (so the engine is not run in the optimisation loop),
    the mission is still simulated here so the viewer can show what it would image."""
    x = pipe.space.encode(design) if design.N <= pipe.space.n_max else None
    if x is not None:
        ev = pipe.evaluate(x, progress=progress)
        if ev.result is not None:
            return ev, ev.result
    else:
        ev = None
    if design.N == 0:
        return ev, None
    return ev, pipe.engine.simulate(design, progress=progress)

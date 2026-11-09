"""SceneManager: owns every 3D actor and updates them from plain arrays.

Everything is drawn at TRUE scale in km: the asteroid mesh, the orbits and the
spacecraft models (~5.6 m across).  Because a 5.6 m spacecraft is far below one pixel
when the whole 34 km asteroid is in view, each spacecraft also gets a constant-size
screen marker and a label; zoom in (or ride / chase it) to see the real model.

The scene light is the Sun, coming from the same direction the illumination test
(e) uses, so the dark side in the picture is the side the model cannot image.

No physics and no Qt here, so it works with the interactive QtInteractor and with an
off-screen ``pv.Plotter`` (tools/snapshot.py) alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk

from spacecov.asteroid import Asteroid

from . import colors
from .spacecraft_model import cone_lines, cone_mesh, pose_matrix, spacecraft_parts

GROUPS = ("cones", "labels", "orbits", "trails", "tracks", "safety", "sun", "axes")
DEFAULT_VISIBLE = {"cones": True, "labels": True, "orbits": True, "trails": True, "tracks": False,
                   "safety": False, "sun": True, "axes": False}


@dataclass
class SpacecraftPose:
    position: np.ndarray          # (3,) km, in the displayed frame
    rotation: np.ndarray          # (3, 3) columns x, y, z(=boresight)
    fov: float                    # full FOV [rad]
    rim: np.ndarray               # (k, 3) km, where the cone's edge rays end (surface or usable range)
    label: str


class _SpacecraftActors:
    def __init__(self, plotter: pv.Plotter, index: int, fov: float, label: str, marker_size: float):
        self.plotter = plotter
        self.fov = fov
        col = colors.sc_color(index) / 255.0
        parts = spacecraft_parts()
        self.body = [
            plotter.add_mesh(parts["bus"], color=(0.78, 0.66, 0.35), smooth_shading=True, specular=0.5,
                             ambient=0.25),
            plotter.add_mesh(parts["scanner"], color=(0.75, 0.75, 0.78), specular=0.6, ambient=0.2),
            plotter.add_mesh(parts["lens"], color=(0.1, 0.1, 0.12), specular=1.0),
            plotter.add_mesh(parts["panels"], color=(0.12, 0.2, 0.45), specular=0.8, specular_power=40,
                             ambient=0.2),
        ]
        # constant-size screen marker (not to scale) so a 5 m spacecraft can be found from 100 km
        self.marker_mesh = pv.PolyData(np.zeros((1, 3)))
        self.marker = plotter.add_mesh(self.marker_mesh, color=col, point_size=marker_size,
                                       render_points_as_spheres=True, lighting=False)
        self.cone_mesh = cone_mesh()
        self.rim_mesh = cone_lines()
        self.cone = plotter.add_mesh(self.cone_mesh, color=col, opacity=0.05, ambient=0.6, diffuse=0.4,
                                     specular=0.0, show_scalar_bar=False)
        self.rim = plotter.add_mesh(self.rim_mesh, color=col, line_width=1.2, opacity=0.7, lighting=False)
        self.label = vtk.vtkBillboardTextActor3D()
        self.label.SetInput(label)
        tp = self.label.GetTextProperty()
        tp.SetFontSize(14)
        tp.SetBold(True)
        tp.SetColor(*col)
        tp.SetShadow(True)
        self.label.SetDisplayOffset(10, 6)
        plotter.renderer.AddActor(self.label)
        for a in self.body + [self.cone, self.rim, self.marker]:
            a.SetPickable(False)

    def update(self, pose: SpacecraftPose, selected: bool):
        T = pose_matrix(pose.position, pose.rotation)
        for a in self.body:
            a.user_matrix = T
        self.marker_mesh.points[0] = pose.position
        self.marker_mesh.Modified()
        pts = np.vstack([pose.position, pose.rim])
        for m in (self.cone_mesh, self.rim_mesh):
            m.points[:] = pts
            m.Modified()
        self.cone.prop.opacity = 0.12 if selected else 0.04
        self.rim.prop.line_width = 2.5 if selected else 1.2
        self.label.SetInput(pose.label)
        self.label.SetPosition(*pose.position)

    def set_marker_size(self, size: float):
        self.marker.prop.point_size = size

    def remove(self):
        for a in self.body + [self.cone, self.rim, self.marker]:
            self.plotter.remove_actor(a, render=False)
        self.plotter.renderer.RemoveActor(self.label)

    def actors(self, group: str):
        return {"cones": [self.cone, self.rim], "labels": [self.label]}.get(group, [])


class SceneManager:
    def __init__(self, plotter: pv.Plotter):
        self.p = plotter
        self.visible = dict(DEFAULT_VISIBLE)
        self.show_edges = False
        self.selected = -1
        self.marker_size = 11.0
        self._sc: list[_SpacecraftActors] = []
        self._asteroid_actor = None
        self._asteroid_mesh: pv.PolyData | None = None
        self._spin_axis = None
        self._lines: dict[str, list] = {"orbits": [], "trails": [], "tracks": []}
        self._trail_meshes: list[pv.PolyData] = []
        self._links = None
        self._safety: list = []
        self._sun_actor = None
        self._sun_label = None
        self._help_actor = None
        self.scene_radius = 60.0
        self.r_ast = 17.0
        self._setup_environment()

    # ------------------------------------------------------------------ environment
    def _setup_environment(self):
        p = self.p
        # Plain FreeType text only.  VTK's MathText backend (picked for strings such as
        # "a | b") boots an embedded matplotlib that hijacks sys.stdout/sys.stderr.
        renderer = vtk.vtkTextRenderer.GetInstance()
        if renderer is not None:
            renderer.SetDefaultBackend(vtk.vtkTextRenderer.FreeType)
        p.set_background("#000004", top="#05081a")
        p.renderer.SetNearClippingPlaneTolerance(1e-5)      # true-scale 5 m models next to a 34 km body
        try:
            p.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        p.remove_all_lights()
        self.sun_light = pv.Light(position=(1000, 0, 0), focal_point=(0, 0, 0), color="#fff6e8",
                                  intensity=1.0, light_type="scene light")
        self.sun_light.positional = False
        p.add_light(self.sun_light)
        # weak fill light so the night side is dark but still readable
        p.add_light(pv.Light(light_type="headlight", intensity=0.18))
        p.add_axes(line_width=2, labels_off=False)
        if not self.visible["axes"]:
            p.hide_axes()
        self.hud = p.add_text("", position="lower_left", font_size=9, color="#d8e4ff", name="hud")
        self.hint = p.add_text("H help   mouse rotate/zoom   WASD fly   1-7 ride   F chase   Space play",
                               position="lower_right", font_size=8, color="#7080a0", name="hint")
        self._setup_scale_bar()
        # big live coverage readout, top centre
        self.banner = vtk.vtkTextActor()
        tp = self.banner.GetTextProperty()
        tp.SetFontSize(21)
        tp.SetBold(True)
        tp.SetColor(0.24, 0.92, 0.45)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToTop()
        tp.SetBackgroundColor(0.0, 0.02, 0.05)
        tp.SetBackgroundOpacity(0.7)
        self.banner.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.banner.SetPosition(0.5, 0.985)
        p.renderer.AddActor2D(self.banner)

    # ------------------------------------------------------------------ km scale bar
    def _setup_scale_bar(self):
        """A 2D bar whose length is a round number of km at the distance of the focal point.

        Updated before every render, so true sizes stay readable at any zoom."""
        pts = vtk.vtkPoints()
        for _ in range(6):
            pts.InsertNextPoint(0, 0, 0)
        lines = vtk.vtkCellArray()
        for a, b in ((0, 1), (2, 3), (4, 5)):           # bar + two end ticks
            lines.InsertNextCell(2)
            lines.InsertCellPoint(a)
            lines.InsertCellPoint(b)
        self._bar_poly = vtk.vtkPolyData()
        self._bar_poly.SetPoints(pts)
        self._bar_poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._bar_poly)
        self._bar = vtk.vtkActor2D()
        self._bar.SetMapper(mapper)
        self._bar.GetProperty().SetColor(0.85, 0.9, 1.0)
        self._bar.GetProperty().SetLineWidth(2)
        self._bar_text = vtk.vtkTextActor()
        tp = self._bar_text.GetTextProperty()
        tp.SetFontSize(13)
        tp.SetColor(0.85, 0.9, 1.0)
        tp.SetJustificationToCentered()
        for a in (self._bar, self._bar_text):
            self.p.renderer.AddActor2D(a)
        self.p.renderer.AddObserver("StartEvent", lambda *_: self._update_scale_bar())

    def _update_scale_bar(self):
        ren = self.p.renderer
        w, h = ren.GetSize()
        if w < 10 or h < 10:
            return
        cam = ren.GetActiveCamera()
        d = cam.GetDistance()
        km_per_px = 2.0 * d * np.tan(np.deg2rad(cam.GetViewAngle()) / 2.0) / h
        target = 0.16 * w * km_per_px
        mag = 10 ** np.floor(np.log10(target))
        length = max(m * mag for m in (1, 2, 5) if m * mag <= target)
        px = length / km_per_px
        x0, y0 = w - 40 - px, 48
        P = self._bar_poly.GetPoints()
        for i, (x, y) in enumerate(((x0, y0), (x0 + px, y0), (x0, y0 - 5), (x0, y0 + 5),
                                    (x0 + px, y0 - 5), (x0 + px, y0 + 5))):
            P.SetPoint(i, x, y, 0)
        P.Modified()
        label = f"{length:g} km" if length >= 1 else f"{length * 1000:g} m"
        self._bar_text.SetInput(f"{label}  (at the view centre)")
        self._bar_text.SetPosition(x0 + px / 2, y0 + 8)

    # ------------------------------------------------------------------ asteroid
    def set_asteroid(self, ast: Asteroid):
        if self._asteroid_actor is not None:
            self.p.remove_actor(self._asteroid_actor, render=False)
            self.p.remove_actor(self._spin_axis, render=False)
        faces = np.hstack([np.full((ast.n_facets, 1), 3), ast.faces]).ravel()
        mesh = pv.PolyData(ast.vertices, faces)
        mesh.cell_data["rgb"] = np.clip(colors.natural(ast.n_facets), 0, 255).astype(np.uint8)
        self._asteroid_mesh = mesh
        self._asteroid_actor = self.p.add_mesh(mesh, scalars="rgb", rgb=True, show_scalar_bar=False,
                                               ambient=0.3, diffuse=0.8, specular=0.05,
                                               show_edges=self.show_edges, edge_color="#202020",
                                               line_width=0.5)
        self.r_ast = ast.r_max
        L = 1.3 * ast.r_max
        self._spin_axis = self.p.add_mesh(pv.Line((0, 0, -L), (0, 0, L)), color="#6070a0", line_width=1.5,
                                          opacity=0.7, lighting=False)
        self.scene_radius = max(self.scene_radius, 2.5 * ast.r_max)

    def set_facet_colors(self, rgb: np.ndarray):
        if self._asteroid_mesh is None:
            return
        self._asteroid_mesh.cell_data["rgb"][:] = rgb
        self._asteroid_mesh.Modified()

    def set_asteroid_rotation(self, R: np.ndarray | None):
        T = np.eye(4)
        if R is not None:
            T[:3, :3] = R
        if self._asteroid_actor is not None:
            self._asteroid_actor.user_matrix = T
            self._spin_axis.user_matrix = T

    def set_edges(self, on: bool):
        self.show_edges = on
        if self._asteroid_actor is not None:
            self._asteroid_actor.prop.show_edges = on

    # ------------------------------------------------------------------ Sun
    def set_sun(self, direction: np.ndarray, lit: bool = True):
        """Sun direction in the displayed frame: moves the light and the Sun arrow."""
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        self.sun_light.position = tuple(d * 1000.0)
        self.sun_light.intensity = 1.0 if lit else 0.0
        a, b = d * 1.25 * self.r_ast, d * 1.9 * self.r_ast
        if self._sun_actor is None:
            self._sun_mesh = pv.Line(a, b)
            self._sun_actor = self.p.add_mesh(self._sun_mesh, color="#ffd45a", line_width=3, lighting=False)
            self._sun_actor.SetPickable(False)
            self._sun_label = vtk.vtkBillboardTextActor3D()
            self._sun_label.SetInput("Sun")
            tp = self._sun_label.GetTextProperty()
            tp.SetFontSize(13)
            tp.SetColor(1.0, 0.83, 0.35)
            tp.SetBold(True)
            self.p.renderer.AddActor(self._sun_label)
            self._apply_group("sun")
        self._sun_mesh.points[:] = np.array([a, b])
        self._sun_mesh.Modified()
        self._sun_label.SetPosition(*b)

    # ------------------------------------------------------------------ spacecraft
    def set_spacecraft(self, poses: list[SpacecraftPose]):
        while len(self._sc) > len(poses):
            self._sc.pop().remove()
        while len(self._sc) < len(poses):
            i = len(self._sc)
            self._sc.append(_SpacecraftActors(self.p, i, poses[i].fov, poses[i].label, self.marker_size))
            self._apply_group("cones")
            self._apply_group("labels")
        for i, (a, pose) in enumerate(zip(self._sc, poses)):
            a.update(pose, i == self.selected)

    def set_marker_size(self, size: float):
        self.marker_size = size
        for a in self._sc:
            a.set_marker_size(size)

    # ------------------------------------------------------------------ lines
    def _set_polylines(self, group: str, curves: list[np.ndarray] | None, width: float, opacity: float):
        store = self._lines[group]
        for a in store:
            self.p.remove_actor(a, render=False)
        store.clear()
        for i, c in enumerate(curves or []):
            if len(c) < 2:
                continue
            a = self.p.add_mesh(pv.lines_from_points(c), color=colors.sc_color(i) / 255.0,
                                line_width=width, opacity=opacity, lighting=False)
            a.SetPickable(False)
            store.append(a)
        self._apply_group(group)

    def set_orbits(self, curves: list[np.ndarray] | None):
        """Exact Keplerian ellipses (inertial frame)."""
        self._set_polylines("orbits", curves, 1.6, 0.75)

    def set_tracks(self, curves: list[np.ndarray] | None):
        """Whole-mission paths in the displayed frame (spirals in the body frame)."""
        self._set_polylines("tracks", curves, 1.0, 0.35)

    def set_trails(self, curves: list[np.ndarray] | None):
        """Fixed-length fading trails, updated in place every animation frame."""
        curves = curves or []
        same = len(curves) == len(self._trail_meshes) and all(
            len(c) == m.n_points for c, m in zip(curves, self._trail_meshes))
        if same:
            for c, m in zip(curves, self._trail_meshes):
                m.points[:] = c
                m.Modified()
            return
        for a in self._lines["trails"]:
            self.p.remove_actor(a, render=False)
        self._lines["trails"].clear()
        self._trail_meshes = []
        for i, c in enumerate(curves):
            m = pv.lines_from_points(c)
            fade = np.linspace(0.1, 1.0, len(c))
            m["rgb"] = np.clip(colors.sc_color(i)[None, :] * fade[:, None], 0, 255).astype(np.uint8)
            a = self.p.add_mesh(m, scalars="rgb", rgb=True, line_width=2.5, lighting=False,
                                show_scalar_bar=False)
            a.SetPickable(False)
            self._lines["trails"].append(a)
            self._trail_meshes.append(m)
        self._apply_group("trails")

    def set_candidate(self, curve: np.ndarray | None, accepted: bool = True):
        """Orbit proposed by the current SA move: green if accepted, red if rejected."""
        if getattr(self, "_candidate", None) is not None:
            self.p.remove_actor(self._candidate, render=False)
            self._candidate = None
        if curve is not None and len(curve) > 1:
            self._candidate = self.p.add_mesh(pv.lines_from_points(curve), line_width=4.5, lighting=False,
                                              color="#4dff88" if accepted else "#ff4d4d", opacity=0.95)
            self._candidate.SetPickable(False)

    def set_links(self, segments: list[tuple[np.ndarray, np.ndarray]]):
        """Red lines between spacecraft that violate the separation constraint."""
        if self._links is not None:
            self.p.remove_actor(self._links, render=False)
            self._links = None
        if segments:
            pts = np.array([q for s in segments for q in s])
            lines = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(segments))])
            self._links = self.p.add_mesh(pv.PolyData(pts, lines=lines), color="#ff2a2a", line_width=4,
                                          lighting=False)

    def set_safety(self, r_min: float | None, r_max: float | None):
        """Periapsis safety sphere r_min (red) and apoapsis limit r_max (blue)."""
        for a in self._safety:
            self.p.remove_actor(a, render=False)
        self._safety = []
        for r, col, op, res in ((r_min, "#ff5050", 0.10, 36), (r_max, "#5080ff", 0.05, 24)):
            if r:
                s = pv.Sphere(radius=r, theta_resolution=res, phi_resolution=res // 2)
                a = self.p.add_mesh(s, style="wireframe", color=col, opacity=op, line_width=0.6,
                                    lighting=False)
                a.SetPickable(False)
                self._safety.append(a)
        self._apply_group("safety")

    # ------------------------------------------------------------------ overlays
    def set_banner(self, text: str, color=(0.24, 0.92, 0.45)):
        self.banner.SetInput(text)
        self.banner.GetTextProperty().SetColor(*color)

    def set_hud(self, text: str):
        self.hud.SetText(0, text)        # CornerAnnotation: 0 = lower left

    def set_legend(self, mode: str):
        """Colour key in the upper-right corner, one coloured text line per entry."""
        for a in getattr(self, "_legend_actors", []):
            self.p.renderer.RemoveActor2D(a)
        self._legend_actors = []
        entries = list(colors.LEGENDS.get(mode, [])) + (colors.FOOTPRINT_LEGEND if mode != "Natural" else [])
        for j, (text, col) in enumerate(entries):
            t = vtk.vtkTextActor()
            t.SetInput(text)
            tp = t.GetTextProperty()
            tp.SetFontSize(14)
            tp.SetColor(*(np.asarray(col) / 255.0))
            tp.SetJustificationToRight()
            tp.SetVerticalJustificationToTop()
            tp.SetBackgroundColor(0.02, 0.03, 0.08)
            tp.SetBackgroundOpacity(0.75)
            t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            t.SetPosition(0.99, 0.91 - 0.032 * j)
            self.p.renderer.AddActor2D(t)
            self._legend_actors.append(t)

    def toggle_help(self, text: str) -> bool:
        if self._help_actor is not None:
            self.p.remove_actor(self._help_actor, render=False)
            self._help_actor = None
            return False
        self._help_actor = self.p.add_text(text, position="upper_right", font_size=9, color="#e8eeff",
                                           name="help")
        self._help_actor.GetTextProperty().SetBackgroundColor(0.03, 0.04, 0.1)
        self._help_actor.GetTextProperty().SetBackgroundOpacity(0.85)
        return True

    # ------------------------------------------------------------------ groups & camera
    def set_group_visible(self, group: str, on: bool):
        self.visible[group] = on
        self._apply_group(group)

    def _apply_group(self, group: str):
        on = self.visible.get(group, True)
        if group == "axes":
            self.p.show_axes() if on else self.p.hide_axes()
            return
        if group in self._lines:
            actors = self._lines[group]
        elif group == "safety":
            actors = self._safety
        elif group == "sun":
            actors = [a for a in (self._sun_actor, self._sun_label) if a is not None]
        else:
            actors = [a for sc in self._sc for a in sc.actors(group)]
        for a in actors:
            a.SetVisibility(on)

    def fit(self, radius: float):
        """Radius the default view must show (largest apoapsis of the fleet)."""
        self.scene_radius = max(1.6 * self.r_ast, radius)

    def reset_camera(self):
        R = self.scene_radius
        cam = self.p.camera
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.position = tuple(np.array([1.0, -1.35, 0.75]) / np.linalg.norm([1.0, -1.35, 0.75]) * R * 2.9)
        cam.up = (0.0, 0.0, 1.0)
        cam.view_angle = 35.0
        self.p.renderer.ResetCameraClippingRange()

    def render(self):
        self.p.render()

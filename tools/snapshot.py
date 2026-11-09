"""Render a scenario off-screen to a PNG (quick visual check / figures for reports).

    python tools/snapshot.py scenarios/orbital_example.json out.png --hours 6
    python tools/snapshot.py scenarios/orbital_example.json out.png --hours 20 --frame body
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyvista as pv  # noqa: E402

from spacecov.scenario import Scenario  # noqa: E402
from viz.colors import COLOR_MODES  # noqa: E402
from viz.presenter import Presenter  # noqa: E402
from viz.scene import SceneManager  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("out")
    p.add_argument("--hours", type=float, default=0.0, help="mission time")
    p.add_argument("--frame", choices=["inertial", "body"], default="inertial")
    p.add_argument("--mode", choices=COLOR_MODES, default="Coverage")
    p.add_argument("--size", type=int, nargs=2, default=(1600, 1000))
    p.add_argument("--zoom", type=float, default=1.0)
    a = p.parse_args()

    plotter = pv.Plotter(off_screen=True, window_size=a.size)
    scene = SceneManager(plotter)
    if a.frame == "body":
        scene.visible["tracks"] = True
    pres = Presenter(scene)
    pres.frame = a.frame
    pres.color_mode = a.mode
    pres.load(Scenario.load(a.scenario))
    pres.evaluate()
    pres.set_time(a.hours * 3600)
    pres.refresh(full=True)
    plotter.camera.zoom(a.zoom)
    plotter.screenshot(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

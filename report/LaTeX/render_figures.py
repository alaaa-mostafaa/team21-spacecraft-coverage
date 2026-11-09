"""Off-screen 3D renders of the best designs found in the validation (uses the viewer's scene code).

    python report/LaTeX/render_figures.py

For each chosen case the best SA run of results/validation is rebuilt as a scenario, saved to
results/validation/best_<case>.json (open it with ./sim results/validation/best_<case>.json)
and rendered at four moments of its mission: Figures/progress_<case>.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

import validate as V  # noqa: E402
from spacecov.design import DesignSpace  # noqa: E402
from spacecov.scenario import Scenario  # noqa: E402
from viz.presenter import Presenter  # noqa: E402
from viz.scene import SceneManager  # noqa: E402

RES = ROOT / "results" / "validation"
FIG = HERE / "Figures"
CASES = {"A2": ("V4", "Eros 1 d"), "A1": ("V4", "Eros 10 d"), "B1": ("V4", "Kleopatra 3 d"), "A5": ("V4", "Eros 6 h")}
SHAPE = {"eros": {"source": "eros", "faces": 6000}, "kleopatra": V.SEARCH_SHAPE["kleopatra"]}


def load_raw():
    raw = json.loads((RES / "raw.json").read_text()) if (RES / "raw.json").exists() else []
    if not raw and (RES / "runs_partial.jsonl").exists():
        raw = [json.loads(x) for x in (RES / "runs_partial.jsonl").read_text().splitlines()]
    return raw


def best_scenario(raw, part, label) -> tuple[Scenario, dict] | None:
    rr = [r for r in raw if r["part"] == part and r["label"] == label and r["algo"] == "sa" and r["feasible"]]
    if not rr:
        return None
    r = min(rr, key=lambda r: r["J"])
    job = V.Job(part, label, r["body"], r["N_max"], r["T_max_days"], r["seed"])
    cfg = V.config_for(job)
    sp = DesignSpace(cfg.n_max, 1.0, 1e9, cfg.t_min_days, cfg.t_max_days)
    design = sp.decode(np.array(r["x_best"]))
    sc = Scenario(name=f"best SA design, {label} (seed {r['seed']}, J = {r['J']:.4f})",
                  asteroid=dict(SHAPE[r["body"]]), config=cfg, design=design, algorithm="sa",
                  algorithm_params={"max_evals": r["evals"], "seed": r["seed"]})
    return sc, r


def render(sc: Scenario, hours: float, out: Path, size=(1100, 900), zoom=1.0, frame="inertial"):
    pl = pv.Plotter(off_screen=True, window_size=size)
    scene = SceneManager(pl)
    if frame == "body":
        scene.visible["tracks"] = True
    pres = Presenter(scene)
    pres.hud_text = pres.banner_text = lambda: ""          # the figure caption says it instead
    pres.frame = frame
    pres.load(sc)
    pres.evaluate()
    pres.set_time(hours * 3600)
    pres.refresh(full=True)
    pl.camera.zoom(zoom)
    pl.screenshot(str(out))
    cov = pres.coverage_now()
    pl.close()
    return 100 * (cov or 0.0)


def main():
    raw = load_raw()
    tmp = RES / "renders"
    tmp.mkdir(parents=True, exist_ok=True)
    for case, (part, label) in CASES.items():
        got = best_scenario(raw, part, label)
        if got is None:
            print(f"{case}: no finished run yet")
            continue
        sc, r = got
        sc.save(RES / f"best_{case}.json")
        T_h = 24 * r["T_days"]
        moments = [T_h / 8, T_h / 4, T_h / 2, T_h]
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
        for ax, h in zip(axes, moments):
            png = tmp / f"{case}_{h:.1f}h.png"
            cov = render(sc, h, png, zoom=1.6 if r["body"] == "eros" else 1.8)
            ax.imshow(plt.imread(png))
            ax.set_axis_off()
            ax.set_title(f"t = {h:.1f} h: {cov:.0f} % imaged", fontsize=9)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.02)
        fig.savefig(FIG / f"progress_{case}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"{case}: {sc.name}, N = {sc.design.N}, T = {T_h:.1f} h")


if __name__ == "__main__":
    main()

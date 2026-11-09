"""Milestone 3 experiments: validation, case studies and SA parameter study.

Problem: within a fixed mission time T = 10 days, choose how many spacecraft to send (up to
N_max slots), each one's camera and orbit, to image as much of the sunlit surface of Eros as
possible at low cost Z and energy E.

    python -m spacecov.legacy.m3.experiments.run_experiments.py                  # everything (~1 h on 8+ cores)
    python -m spacecov.legacy.m3.experiments.run_experiments.py --quick          # smoke run: 2 seeds, small budgets
    python -m spacecov.legacy.m3.experiments.run_experiments.py --only cases --workers 6
    python -m spacecov.legacy.m3.experiments.run_experiments.py --figures-only   # redraw figures from results/raw_*.json

Outputs go to results/ :
    validation_saturation.csv   random fleets of N = 1, 3, 5: is T long enough to be interesting?
    validation_benchmarks.csv   SA on functions with a known optimum (0 at x = 0)
    validation_vs_random.csv    SA vs random search with the same evaluation budget
    validation_no_light.csv     every facet treated as lit (the Milestone 2 assumption)
    cases.csv / cases_summary.csv     every run / mean, std, best, worst per case
    params.csv / params_summary.csv   parameter study
    best_Nmax{n}_ds{d}.json     best fleet of each case - open it with  python -m viz <file>
    fig_*.png                   figures for the report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from spacecov.legacy.m3.algorithms import SAParams, random_search, solve  # noqa: E402
from spacecov.legacy.m3.algorithms.benchmarks import solve_benchmark  # noqa: E402
from spacecov.legacy.m3.design import OrbitalSolution  # noqa: E402
from spacecov.legacy.m3.engine import evaluate  # noqa: E402
from spacecov.legacy.m3.problem import OrbitalProblem  # noqa: E402
from spacecov.legacy.m3.asteroid import make_asteroid  # noqa: E402
from spacecov.legacy.m3.config import BODIES, OrbitalConfig  # noqa: E402
from spacecov.legacy.m3.scenario import Scenario  # noqa: E402

SEARCH_SHAPE = {"source": "eros", "faces": 1500}       # used during the search (fast)
CHECK_SHAPE = {"source": "eros", "faces": 6000}        # best designs are re-checked on a finer mesh
MISSION_DAYS = 10.0
FRAME_MIN = 10.0                                       # frame interval = time step [min]
CASES_NMAX = (5, 7, 10)
CASES_SUN = (0.0, 30.0, 60.0)                          # sub-solar latitude (season) [deg]
PARAM_CASE = (5, 0.0)


# --------------------------------------------------------------------------
# One SA run (executed in a worker process)
# --------------------------------------------------------------------------
@dataclass
class Job:
    group: str                       # cases | params | validation
    label: str                       # case / setting name
    n_max: int
    sun_lat: float
    seed: int
    params: dict
    lighting: bool = True
    random_search: bool = False
    extra: dict = field(default_factory=dict)
    days: float = MISSION_DAYS
    fixed_n: bool = False
    cfg_extra: dict = field(default_factory=dict)    # extra OrbitalConfig settings, e.g. {"dt_min": 15}
    body: str = "eros"                                # target body (spacecov.config.BODIES)


def mission_config(n_max: int, sun_lat: float, lighting: bool = True, days: float = MISSION_DAYS,
                   body: str = "eros", **extra) -> OrbitalConfig:
    return OrbitalConfig(duration_days=days, dt_min=FRAME_MIN, n_max=n_max, sun_lat_deg=sun_lat,
                         lighting=lighting, **{**BODIES[body], **extra})


def body_shapes(body: str) -> tuple[dict, dict]:
    """(search mesh, check mesh).  Kleopatra's radar model (4092 facets) is used for both."""
    return (SEARCH_SHAPE, CHECK_SHAPE) if body == "eros" else ({"source": body}, {"source": body})


def run_job(job: Job) -> dict:
    search_shape, check_shape = body_shapes(job.body)
    ast = make_asteroid(search_shape)
    cfg = mission_config(job.n_max, job.sun_lat, job.lighting, job.days, job.body, **job.cfg_extra)
    prob = OrbitalProblem(ast, job.n_max, cfg, fixed_n=job.fixed_n)
    t0 = time.perf_counter()
    if job.random_search:
        rs = random_search(prob, job.params["max_evals"], job.seed)
        x_best, curve, curve_C = rs["x_best"], rs["J_best"], rs["best_C"]
        curve_N = np.full(len(curve), np.nan)
        extra = {}
    else:
        res, _ = solve(prob, SAParams(**dict(job.params, seed=job.seed)))
        x_best, curve, curve_C = res.x_best, res.history["J_best"], res.history["best_C"]
        curve_N = res.history["best_N"]
        extra = {"T0": res.T0, "alpha": res.alpha, "L": res.chain_length, "n_evals": res.n_evals,
                 "accepted": float(res.history["accepted"].mean()), "stop": res.stop_reason}
    seconds = time.perf_counter() - t0
    best = prob.evaluate(x_best)
    sol = prob.decode(x_best).active()
    fine = evaluate(sol, make_asteroid(check_shape), cfg, keep_visible=False)
    return {"group": job.group, "label": job.label, "body": job.body, "N_max": job.n_max, "sun_lat": job.sun_lat, "days": job.days,
            "seed": job.seed, "J": best.J, "C": best.C, "Z": best.Z, "E": best.E, "E_dv": best.E_dv,
            "E_elec": best.E_elec, "S": best.S, "N": best.n_active,
            "n_narrow": int(np.sum(sol.types == 1)), "n_wide": int(np.sum(sol.types == 2)),
            "feasible": best.feasible, "min_soc": float(best.soc.min()), "dv_total_ms": 1000 * float(best.dv.sum()),
            "C_fine": fine.C, "J_fine": fine.J, "seconds": seconds,
            "evals_per_s": len(curve) / max(seconds, 1e-9), "x_best": np.asarray(x_best).tolist(),
            "curve_J": np.asarray(curve).tolist(), "curve_C": np.asarray(curve_C).tolist(),
            "curve_N": np.asarray(curve_N).tolist(), "coverage_t": best.coverage_curve()[0].tolist(),
            **extra, **job.extra}


def run_all(jobs: list[Job], workers: int, tag: str) -> list[dict]:
    """Run jobs in parallel; jobs lost to a crashed worker process are resubmitted (up to 3 rounds)."""
    out, t0, todo = [], time.perf_counter(), list(jobs)
    for _ in range(3):
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_job, j): j for j in todo}
                for f in as_completed(futs):
                    try:
                        r = f.result()
                    except BrokenProcessPool:
                        continue
                    out.append(r)
                    print(f"  [{tag} {len(out)}/{len(jobs)}] {r['label']:<20} seed {r['seed']}:  J {r['J']:+.4f}  "
                          f"C {100 * r['C']:5.1f} %  N {r['N']}  ({r['seconds']:.0f} s)   elapsed "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
        except BrokenProcessPool:
            pass
        done = {(r["label"], r["seed"]) for r in out}
        todo = [j for j in jobs if (j.label, j.seed) not in done]
        if not todo:
            break
        print(f"  {len(todo)} job(s) lost to a crashed worker - resubmitting", flush=True)
    return sorted(out, key=lambda r: (r["label"], r["seed"]))


def save_raw(out: Path, name: str, rows: list[dict]):
    """Every run including its convergence curve, so figures can be redrawn without re-running."""
    (out / f"raw_{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def load_raw(out: Path, name: str) -> list[dict]:
    p = out / f"raw_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


# --------------------------------------------------------------------------
# Validation studies without SA
# --------------------------------------------------------------------------
def saturation_check(out: Path, n_fleets: int = 50) -> list[dict]:
    """Random fleets of N = 1, 3, 5 over the mission: if most reached ~100 % of A*, the mission
    would be too long for the optimisation to matter."""
    ast = make_asteroid(SEARCH_SHAPE)
    prob = OrbitalProblem(ast, 5, mission_config(5, 0.0))
    rng = np.random.default_rng(0)
    rows = []
    for n in (1, 3, 5):
        for _ in range(n_fleets):
            X = prob.random_solution(rng).reshape(5, 7)
            X[:, 6] = 0
            X[:n, 6] = rng.integers(1, 3, n)
            r = prob.evaluate(X.ravel())
            rows.append({"N": n, "C": r.C, "J": r.J, "feasible": r.feasible})
        C = np.array([r["C"] for r in rows if r["N"] == n])
        print(f"  N = {n}: random fleets cover {100 * C.mean():.0f} % of A* on average "
              f"(min {100 * C.min():.0f}, max {100 * C.max():.0f}, >= 95 %: {np.mean(C >= 0.95):.0%})", flush=True)
    write_dicts(out / "validation_saturation.csv", rows)
    save_raw(out, "saturation", rows)
    return rows


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
SCALARS = ("group", "label", "N_max", "sun_lat", "days", "seed", "J", "C", "Z", "E", "E_dv", "E_elec", "S", "N",
           "n_narrow", "n_wide", "feasible", "min_soc", "dv_total_ms", "C_fine", "J_fine", "seconds",
           "evals_per_s", "T0", "alpha", "L", "n_evals", "accepted", "stop", "setting", "value")


def write_runs(path: Path, rows: list[dict]):
    keys = [k for k in SCALARS if any(k in r for r in rows)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    out = []
    for k, rs in groups.items():
        J, C, N = (np.array([r[x] for r in rs], float) for x in ("J", "C", "N"))
        sd = (lambda a: a.std(ddof=1) if len(a) > 1 else 0.0)
        out.append({"label": k, "N_max": rs[0]["N_max"], "sun_lat": rs[0]["sun_lat"], "runs": len(rs),
                    "J_best": J.min(), "J_mean": J.mean(), "J_std": sd(J), "J_worst": J.max(),
                    "C_mean": C.mean(), "C_std": sd(C), "C_best": C.max(),
                    "C_fine_mean": np.mean([r["C_fine"] for r in rs]),
                    "N_mean": N.mean(), "Z_mean": np.mean([r["Z"] for r in rs]),
                    "E_mean": np.mean([r["E"] for r in rs]),
                    "feasible": sum(bool(r["feasible"]) for r in rs),
                    "seconds_mean": np.mean([r["seconds"] for r in rs])})
    return out


def write_dicts(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def save_best(rows: list[dict], out: Path, n_max: int, sun_lat: float):
    r = min(rows, key=lambda r: (not r["feasible"], r["J"]))
    sc = Scenario(name=f"SA best of {len(rows)} runs: N_max = {n_max}, sub-solar latitude {sun_lat:g} deg "
                       f"(N = {r['N']}, C = {100 * r['C']:.1f} %, J = {r['J']:.4f})",
                  asteroid=dict(SEARCH_SHAPE), config=mission_config(n_max, sun_lat))
    sc.fleet = OrbitalSolution.from_vector(r["x_best"]).active()
    sc.save(out / f"best_Nmax{n_max}_ds{sun_lat:g}.json")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figures(out: Path, cases: list[dict], params: list[dict], vs_random: list[dict], saturation: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    colors = {5: "#1f77b4", 7: "#d62728", 10: "#2ca02c"}
    markers = {0.0: "o", 30.0: "s", 60.0: "^"}

    def mean_band(ax, runs, color, label, key="curve_J", scale=1.0, ls="-"):
        m = min(len(r[key]) for r in runs)
        Y = scale * np.array([r[key][:m] for r in runs], float)
        x = np.arange(1, m + 1)
        ax.plot(x, Y.mean(0), color=color, ls=ls, label=label, lw=1.5)
        ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=color, alpha=0.12, lw=0)

    def runs_of(n, d):
        return [r for r in cases if r["N_max"] == n and r["sun_lat"] == d]

    if cases:
        fig, axes = plt.subplots(2, len(CASES_SUN), figsize=(12, 6.4), sharex=True)
        for j, d in enumerate(CASES_SUN):
            for n in CASES_NMAX:
                if runs_of(n, d):
                    mean_band(axes[0, j], runs_of(n, d), colors[n], f"N_max = {n}")
                    mean_band(axes[1, j], runs_of(n, d), colors[n], f"N_max = {n}", key="curve_N")
            axes[0, j].set_title(f"sub-solar latitude {d:g}°")
            axes[0, j].set_ylabel("best fitness J (mean ± std)")
            axes[1, j].set_ylabel("spacecraft used by the best design")
            axes[1, j].set_xlabel("evaluation")
            axes[0, j].legend()
        fig.suptitle(f"SA convergence per case study (T = {MISSION_DAYS:g} days)")
        fig.tight_layout()
        fig.savefig(out / "fig_convergence.png")
        plt.close(fig)

        labels = [f"{n}\n{d:g}°" for d in CASES_SUN for n in CASES_NMAX]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, key, scale, ylab in ((axs[0], "J", 1, "final best J (lower is better)"),
                                     (axs[1], "C", 100, "coverage C of best design [% of A*]"),
                                     (axs[2], "N", 1, "spacecraft sent N")):
            ax.boxplot([[scale * r[key] for r in runs_of(n, d)] for d in CASES_SUN for n in CASES_NMAX],
                       tick_labels=labels)
            ax.set_ylabel(ylab)
            ax.set_xlabel("N_max / sub-solar latitude")
        fig.suptitle("Spread over independent runs (different seeds)")
        fig.tight_layout()
        fig.savefig(out / "fig_case_boxplots.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for r in cases:
            ax.scatter(r["Z"], 100 * r["C"], color=colors[r["N_max"]], marker=markers[r["sun_lat"]], s=28,
                       alpha=0.8, edgecolor="k", lw=0.3)
        for n in CASES_NMAX:
            ax.scatter([], [], color=colors[n], label=f"N_max = {n}")
        for d in CASES_SUN:
            ax.scatter([], [], color="grey", marker=markers[d], label=f"sub-solar lat. {d:g}°")
        ax.set_xlabel("cost Z")
        ax.set_ylabel("coverage C [% of A*]")
        ax.set_title("Coverage vs cost of every final design")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_tradeoff.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for d in CASES_SUN:
            for n in CASES_NMAX:
                rs = runs_of(n, d)
                if not rs:
                    continue
                b = min(rs, key=lambda r: (not r["feasible"], r["J"]))
                t = np.arange(len(b["coverage_t"])) * FRAME_MIN / 60 / 24
                ax.plot(t, 100 * np.array(b["coverage_t"]), color=colors[n],
                        ls={0.0: "-", 30.0: "--", 60.0: ":"}[d], label=f"N_max {n}, {d:g}° (N = {b['N']})")
        ax.set_xlabel("mission time [days]")
        ax.set_ylabel("covered [% of A*]")
        ax.set_title("Coverage vs time of the best design of each case")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(out / "fig_coverage_vs_time.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.6))
        x = np.arange(len(labels))
        nn = [np.mean([r["n_narrow"] for r in runs_of(n, d)] or [0]) for d in CASES_SUN for n in CASES_NMAX]
        nw = [np.mean([r["n_wide"] for r in runs_of(n, d)] or [0]) for d in CASES_SUN for n in CASES_NMAX]
        ax.bar(x, nn, label="narrow (6°)", color="#4da6ff")
        ax.bar(x, nw, bottom=nn, label="wide (12°)", color="#ff5c8a")
        ax.set_xticks(x, labels)
        ax.set_ylabel("spacecraft (mean over seeds)")
        ax.set_xlabel("N_max / sub-solar latitude")
        ax.set_title("Fleet composition chosen by SA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_fleet_composition.png")
        plt.close(fig)

    if params:
        settings = sorted({r["setting"] for r in params})
        fig, axes = plt.subplots(1, len(settings), figsize=(3.6 * len(settings), 3.6))
        for ax, s in zip(np.atleast_1d(axes), settings):
            runs = [r for r in params if r["setting"] == s]
            values = sorted({r["value"] for r in runs}, key=float)
            cmap = plt.get_cmap("viridis")
            for i, v in enumerate(values):
                mean_band(ax, [r for r in runs if r["value"] == v], cmap(i / max(len(values) - 1, 1)), f"{s} = {v}")
            ax.set_xlabel("evaluation")
            ax.set_ylabel("best J (mean ± std)")
            ax.set_title(f"effect of {s}")
            ax.legend(fontsize=7)
        fig.suptitle(f"SA parameter study (N_max = {PARAM_CASE[0]}, sub-solar latitude {PARAM_CASE[1]:g}°)")
        fig.tight_layout()
        fig.savefig(out / "fig_parameter_study.png")
        plt.close(fig)

    if vs_random:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        mean_band(ax, [r for r in vs_random if r["label"] == "SA"], "#d62728", "simulated annealing")
        mean_band(ax, [r for r in vs_random if r["label"] == "random search"], "#7f7f7f", "random search")
        ax.set_xlabel("evaluation")
        ax.set_ylabel("best J (mean ± std)")
        ax.set_title("SA vs random search, same budget")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_sa_vs_random.png")
        plt.close(fig)

    if saturation:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for n, col in ((1, "#7f7f7f"), (3, "#1f77b4"), (5, "#d62728")):
            ax.hist([100 * r["C"] for r in saturation if r["N"] == n], bins=np.linspace(0, 100, 21), alpha=0.55,
                    color=col, label=f"N = {n}")
        ax.set_xlabel(f"coverage after {MISSION_DAYS:g} days [% of A*]")
        ax.set_ylabel("random fleets")
        ax.set_title("Saturation check: random fleets rarely reach 100 %")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_saturation.png")
        plt.close(fig)


def mission_figures(out: Path, mission: list[dict], cases: list[dict]):
    """Mission time vs chosen fleet, coverage vs fixed fleet size, 2 days vs 10 days."""
    if not mission:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    ten = [r for r in cases if r["N_max"] == 7 and r["sun_lat"] == 0.0]
    sweep = [r for r in mission if r["label"].startswith("T=") and r["N_max"] == 7] +             [dict(r, days=MISSION_DAYS) for r in ten]
    days = sorted({r["days"] for r in sweep})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for d in days:
        rs = [r for r in sweep if r["days"] == d]
        a1.scatter([d] * len(rs), [r["N"] for r in rs], color="#4da6ff", alpha=0.6, s=30)
        a2.scatter([d] * len(rs), [100 * r["C"] for r in rs], color="#3ddc84", alpha=0.6, s=30)
    a1.plot(days, [np.mean([r["N"] for r in sweep if r["days"] == d]) for d in days], "o-", color="#1f5fbf",
            label="mean")
    a2.plot(days, [np.mean([100 * r["C"] for r in sweep if r["days"] == d]) for d in days], "o-",
            color="#1a8f50", label="mean")
    a1.set_xlabel("mission time T [days]")
    a1.set_ylim(0.5, 5.5)
    a1.set_yticks([1, 2, 3, 4, 5])
    a1.set_ylabel("spacecraft SA decides to send")
    a1.set_title("Spacecraft SA decides to send (N_max = 7)")
    a2.set_xlabel("mission time T [days]")
    a2.set_ylabel("coverage C [% of sunlit area]")
    a2.set_title("Coverage of the chosen fleet")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(days, [f"{d:g}" for d in days])
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_mission_time.png")
    plt.close(fig)

    fixed = [r for r in mission if r["label"].startswith("fixed N")]
    if fixed:
        ns = sorted({r["N"] for r in fixed})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        a1.plot(ns, [np.mean([100 * r["C"] for r in fixed if r["N"] == n]) for n in ns], "o-", color="#1a8f50")
        a1.set_xlabel("number of spacecraft N (fixed)")
        a1.set_ylabel("best coverage C [%]")
        a1.set_title("Coverage after 2 days vs fleet size")
        a2.plot(ns, [np.mean([r["J"] for r in fixed if r["N"] == n]) for n in ns], "o-", color="#d62728")
        a2.set_xlabel("number of spacecraft N (fixed)")
        a2.set_ylabel("best fitness J (lower is better)")
        a2.set_title("Fitness: where extra spacecraft stop paying off")
        fig.tight_layout()
        fig.savefig(out / "fig_fleet_size.png")
        plt.close(fig)


def draw_all(out: Path):
    figures(out, load_raw(out, "cases"), load_raw(out, "params"), load_raw(out, "vs_random"),
            load_raw(out, "saturation"))
    mission_figures(out, load_raw(out, "mission"), load_raw(out, "cases"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=3000, help="SA budget per run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", choices=["validation", "cases", "params", "mission"], action="append")
    ap.add_argument("--quick", action="store_true", help="2 seeds, 200 evaluations (checks the pipeline)")
    ap.add_argument("--params", nargs="*", default=None,
                    help="SA parameters to study (default: alpha chain_length p0 step0 p_type)")
    ap.add_argument("--sun", nargs="*", type=float, default=None, help="sub-solar latitudes of the case studies")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from results/raw_*.json")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.evals = 2, 200
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.figures_only:
        draw_all(out)
        print(f"figures redrawn in {out}")
        return
    parts = set(a.only or ["validation", "cases", "params", "mission"])
    base = asdict(SAParams(max_evals=a.evals))
    seeds = range(a.seeds)
    meta = {"evals": a.evals, "seeds": a.seeds, "search_shape": SEARCH_SHAPE, "check_shape": CHECK_SHAPE,
            "mission_days": MISSION_DAYS, "frame_min": FRAME_MIN, "sa_defaults": base,
            "started": time.strftime("%Y-%m-%d %H:%M")}

    if "validation" in parts:
        print("validation: saturation check (random fleets)", flush=True)
        saturation_check(out, 50 if not a.quick else 10)
        print("validation: benchmarks with known optimum", flush=True)
        rows = []
        for name, dim in (("sphere", 5), ("rastrigin", 2), ("rastrigin", 5), ("ackley", 2), ("ackley", 5)):
            vals = [solve_benchmark(name, dim, SAParams(max_evals=10000 if not a.quick else 2000, seed=s))
                    for s in range(10 if not a.quick else 3)]
            J = np.array([v.J_best for v in vals])
            rows.append({"function": name, "dim": dim, "known_optimum": 0.0, "best": J.min(), "mean": J.mean(),
                         "std": J.std(ddof=1), "worst": J.max(), "runs": len(J),
                         "success_rate (<1e-1)": float(np.mean(J < 0.1))})
            print(f"  {name:<9} d={dim}: best {J.min():.2e}  mean {J.mean():.2e}", flush=True)
        write_dicts(out / "validation_benchmarks.csv", rows)

        print("validation: SA vs random search and the all-lit model", flush=True)
        n, d = PARAM_CASE
        jobs = [Job("validation", "SA", n, d, s, base) for s in seeds]
        jobs += [Job("validation", "random search", n, d, s, base, random_search=True) for s in seeds]
        jobs += [Job("validation", "all lit (M2 model)", n, d, s, base, lighting=False) for s in seeds]
        res = run_all(jobs, a.workers, "validation")
        vs_random = [r for r in res if r["label"] in ("SA", "random search")]
        save_raw(out, "vs_random", vs_random)
        write_runs(out / "validation_vs_random.csv", vs_random)
        write_runs(out / "validation_no_light.csv", [r for r in res if r["label"].startswith("all lit")])
        sa_J = [r["J"] for r in vs_random if r["label"] == "SA"]
        rs_J = [r["J"] for r in vs_random if r["label"] == "random search"]
        try:
            from scipy.stats import mannwhitneyu
            meta["sa_vs_random_p"] = float(mannwhitneyu(sa_J, rs_J, alternative="less").pvalue)
        except Exception:
            pass

    if "cases" in parts:
        print("case studies", flush=True)
        suns = tuple(a.sun) if a.sun else CASES_SUN
        jobs = [Job("cases", f"Nmax={n}, ds={d:g}", n, d, s, base)
                for d in suns for n in CASES_NMAX for s in seeds]
        res = run_all(jobs, a.workers, "cases")
        save_raw(out, "cases", res)
        write_runs(out / "cases.csv", res)
        write_dicts(out / "cases_summary.csv", summarise(res))
        for d in suns:
            for n in CASES_NMAX:
                rs = [r for r in res if r["N_max"] == n and r["sun_lat"] == d]
                if rs:
                    save_best(rs, out, n, d)

    if "params" in parts:
        print("parameter study", flush=True)
        n, d = PARAM_CASE
        grid = {"alpha": [0.80, 0.90, 0.95, 0.98], "chain_length": [10, 50, 150],
                "p0": [0.5, 0.8, 0.95], "step0": [0.05, 0.25, 0.5], "p_type": [0.05, 0.15, 0.3]}
        if a.params:
            grid = {k: v for k, v in grid.items() if k in a.params}
        jobs = []
        for setting, values in grid.items():
            for v in values:
                p = dict(base, **{setting: v})
                if setting == "alpha":
                    p["chain_length"] = 50                 # fixed L so alpha alone sets the schedule
                for s in seeds:
                    jobs.append(Job("params", f"{setting}={v}", n, d, s, p, extra={"setting": setting, "value": v}))
        params = run_all(jobs, a.workers, "params")
        save_raw(out, "params", params)
        write_runs(out / "params.csv", params)
        write_dicts(out / "params_summary.csv", summarise(params))

    if "mission" in parts:
        # how the mission time changes the fleet SA sends, and coverage as a function of fleet size
        print("mission-time study", flush=True)
        jobs = [Job("mission", f"T={days:g}d, Nmax={n}", n, 0.0, s, base, days=days)
                for days in (1.0, 2.0, 3.0, 5.0) for n in ((5, 7, 10) if days == 2.0 else (7,))
                for s in (seeds if days == 2.0 else range(min(3, a.seeds)))]
        jobs += [Job("mission", f"fixed N={n}, T=2d", n, 0.0, s, dict(base, max_evals=max(200, a.evals * 2 // 3)),
                     days=2.0, fixed_n=True) for n in range(1, 7) for s in range(min(3, a.seeds))]
        res = run_all(jobs, a.workers, "mission")
        save_raw(out, "mission", res)
        write_runs(out / "mission.csv", res)
        write_dicts(out / "mission_summary.csv", summarise(res))
        for n in (5, 7, 10):
            rs = [r for r in res if r["label"] == f"T=2d, Nmax={n}"]
            if rs:
                r = min(rs, key=lambda r: (not r["feasible"], r["J"]))
                sc = Scenario(name=f"SA best of {len(rs)} runs: 2-day mission, N_max = {n} "
                                   f"(N = {r['N']}, C = {100 * r['C']:.1f} %)",
                              asteroid=dict(SEARCH_SHAPE), config=mission_config(n, 0.0, True, 2.0))
                sc.fleet = OrbitalSolution.from_vector(r["x_best"]).active()
                sc.save(out / f"best_T2d_Nmax{n}.json")

    (out / "run_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    draw_all(out)
    print(f"done - results in {out}")


if __name__ == "__main__":
    main()

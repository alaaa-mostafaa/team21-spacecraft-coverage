"""Validation of the current formulation, following the Milestone 3 methodology (report Sections 5 and 6.2).

    V1  physics checks                 the unit tests (tests/test_physics.py)            -> ./sim test
    V2  known optima                   SA on sphere / Rastrigin / Ackley (minimum 0), 10 seeds x 10 000 evals
    V3  saturation                     random designs of N = 1, 3, 5 flying for T_max: is there anything to optimise?
    V4  SA vs random search            same budget, 9 settings (Eros and Kleopatra, long and short missions),
                                       one-sided Mann-Whitney U test on J
    V5  all-lit model                  sunlight and shadows switched off, to compare with the literature
    V6  forced fleet size              N fixed to 1, 2, 3, ... (orbits and T optimised): is SA's own N the best?
    V7  finer mesh                     every best Eros design re-evaluated on the 6000-facet model
    P   SA parameter study             cooling factor alpha, start acceptance p0, start step, warm start
                                       (one parameter changed at a time from the defaults; Eros, T_max 2 d)

    Genetic Algorithm (Milestone 4, report chapter "Genetic Algorithm Implementation"):
    G2  known optima                   GA on sphere / Rastrigin / Ackley, as V2
    GV4 GA on the V4 settings          same 9 settings, seeds and budget as SA / random search in V4
    GV5 all-lit model                  GA on V5 (comparison with the literature)
    GP  GA parameter study             P, start, elite share, crossover fraction, tournament size, alignment,
                                       blend probability / weight, mutation step (one at a time; Eros, T_max 2 d)
    H   hard cases                     the large body (Kleopatra) with very short missions, up to N_max = 20
                                       (101 genes): SA, GA and random search
    B   budget sweep                   SA and GA with 5000 evaluations on three V4 settings (same seeds):
                                       does the GA catch up with a larger budget?
    SA vs GA (summary "SA_GA"): built from V4 (SA, random) + GV4 (GA) whenever both are present.
    M   matched comparison             SA and GA with an equal budget of 1200 evaluations, seeds 0-2, on
                                       Eros, N_max = 10, T_max = 10 d, 3000-facet search mesh
    NI  equal native iterations        the M case with 20 GA generations (1143 evaluations) against an SA
                                       budget of 20 evaluations; those 20 are all used by SA's warm start
                                       (n_init = 20), so the SA side is the best of 20 random designs

Adaptation to the current problem: the mission time T is a decision variable, so the Milestone 3 mission
lengths become T_max (T is searched in [T_min, T_max], T_min = min(6 h, T_max / 4)).

    python experiments/validate.py                    # everything (~30-40 min on 12 cores)
    python experiments/validate.py --quick            # 2 seeds, small budgets: a pipeline check
    python experiments/validate.py --only V4 V6 --seeds 5 --evals 2000 --workers 11
    python experiments/validate.py --only G2 GP GV4 GV5 --seeds 5 --evals 2000     # the GA parts
    python experiments/validate.py --only M NI          # fixed seeds and budgets (--seeds / --evals ignored)

Outputs in results/validation/: <part>.csv, raw.json, summary.json and fig_*.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from importlib import metadata as package_metadata

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spacecov.algorithms import make  # noqa: E402
from spacecov.algorithms.benchmarks import BenchmarkProblem  # noqa: E402
from spacecov.asteroid import PRESETS, make_asteroid  # noqa: E402
from spacecov.config import for_body  # noqa: E402
from spacecov.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "results" / "validation"
SEARCH_SHAPE = {"eros": {"source": "eros", "faces": 1500}, "kleopatra": dict(PRESETS["kleopatra"])}
CHECK_SHAPE = {"source": "eros", "faces": 6000}            # V7: Kleopatra's radar model is used as is
H = 1.0 / 24.0

# V4 (label, body, N_max, T_max in days) - the nine Milestone 3 settings
VS_RANDOM = [("Eros 10 d", "eros", 5, 10.0), ("Eros 1 d", "eros", 5, 1.0), ("Eros 1 d, N_max 10", "eros", 10, 1.0),
             ("Eros 12 h", "eros", 10, 12 * H), ("Eros 6 h", "eros", 10, 6 * H), ("Eros 2 h", "eros", 10, 2 * H),
             ("Kleopatra 3 d", "kleopatra", 5, 3.0), ("Kleopatra 3 d, N_max 10", "kleopatra", 10, 3.0),
             ("Kleopatra 12 h", "kleopatra", 10, 12 * H)]
# V6 (label, body, T_max in days, forced N values, seeds) + the free-N setting SA is compared with
FIXED_N = [("Eros 2 d", "eros", 2.0, range(1, 7), None), ("Eros 6 h", "eros", 6 * H, range(1, 9), None),
           ("Kleopatra 3 d", "kleopatra", 3.0, range(1, 6), 3)]
# P (setting label, SA parameters that differ from the defaults); Eros, N_max 5, T_max 2 days
PARAM_CASE = ("eros", 5, 2.0)
PARAM_STUDY = [("default", {}),
               ("alpha 0.80", {"alpha": 0.80}), ("alpha 0.95", {"alpha": 0.95}), ("alpha 0.98", {"alpha": 0.98}),
               ("p0 0.30", {"p0": 0.30}), ("p0 0.80", {"p0": 0.80}), ("p0 0.95", {"p0": 0.95}),
               ("step0 0.05", {"step0": 0.05}), ("step0 0.10", {"step0": 0.10}), ("step0 0.50", {"step0": 0.50}),
               ("no warm start", {"n_init": 1})]
# GP (setting label, GA parameters that differ from the defaults); same case as P
GA_PARAM_STUDY = [("default", {}),
                  ("P 10", {"pop_size": 10}), ("P 20", {"pop_size": 20}), ("P 80", {"pop_size": 80}),
                  ("P 120", {"pop_size": 120}),
                  ("uniform start", {"stratified": False}),
                  ("elites 0 %", {"elite_frac": 0.0}), ("elites 2.5 %", {"elite_frac": 0.025}),
                  ("elites 10 %", {"elite_frac": 0.10}), ("elites 20 %", {"elite_frac": 0.20}),
                  ("crossover 0", {"crossover_frac": 0.0}), ("crossover 0.6", {"crossover_frac": 0.6}),
                  ("crossover 0.95", {"crossover_frac": 0.95}), ("crossover 1", {"crossover_frac": 1.0}),
                  ("tournament 3", {"tournament_k": 3}), ("tournament 5", {"tournament_k": 5}),
                  ("no alignment", {"align": False}),
                  ("blend 0", {"p_blend": 0.0}), ("blend 0.5", {"p_blend": 0.5}), ("blend 1", {"p_blend": 1.0}),
                  ("blend 1, lambda 1/2", {"p_blend": 1.0, "blend_weight": 0.5}),
                  ("step0 0.10", {"step0": 0.10}), ("step0 0.50", {"step0": 0.50})]
# H (label, body, N_max, T_max in days): the big body with very little time
HARD = [("Kleopatra 6 h", "kleopatra", 10, 6 * H), ("Kleopatra 2 h", "kleopatra", 10, 2 * H),
        ("Kleopatra 6 h, N_max 20", "kleopatra", 20, 6 * H)]
# B (label of a V4 setting, budget): the same setting with a larger budget
BUDGET = [("Eros 1 d", 5000), ("Eros 2 h", 5000), ("Kleopatra 12 h", 5000)]
# M / NI (label, body, N_max, T_max in days, search mesh, seeds): the matched GA-SA comparison
MATCHED = ("Eros 10 d, N_max 10, 3000 facets", "eros", 10, 10.0, {"source": "eros", "faces": 3000}, 3)
MATCHED_EVALS = 1200
NATIVE_ITERATIONS = 20          # GA generations (initial one included) and SA evaluations
FREE_FOR_FIXED = {"Eros 2 d": ("eros", 10, 2.0), "Eros 6 h": ("eros", 10, 6 * H), "Kleopatra 3 d": ("kleopatra", 10, 3.0)}


@dataclass
class Job:
    part: str
    label: str
    body: str
    n_max: int
    t_max_days: float
    seed: int
    algo: str = "sa"
    evals: int = 2000
    lighting: bool = True
    fixed_n: bool = False
    params: dict = field(default_factory=dict)           # algorithm settings that differ from the defaults (P, GP)
    mesh: dict | None = None                             # search mesh; None -> SEARCH_SHAPE[body]
    extra: dict = field(default_factory=dict)


def config_for(job: Job):
    cfg = for_body(job.body)
    cfg.n_max, cfg.t_max_days, cfg.fixed_n, cfg.lighting = job.n_max, job.t_max_days, job.fixed_n, job.lighting
    cfg.t_min_days = min(0.25, job.t_max_days / 4.0)
    return cfg


def run_job(job: Job) -> dict:
    cfg = config_for(job)
    shape = job.mesh or SEARCH_SHAPE[job.body]
    pipe = Pipeline(make_asteroid(shape), cfg)
    t0 = time.perf_counter()
    res = make(job.algo, max_evals=job.evals, seed=job.seed, **job.params).run(pipe)
    seconds = time.perf_counter() - t0
    b = res.best
    row = {"part": job.part, "label": job.label, "algo": job.algo, "body": job.body, "N_max": job.n_max,
           "T_max_days": job.t_max_days, "seed": job.seed, "lighting": job.lighting, "fixed_n": job.fixed_n,
           "mesh_source": shape, "mesh_facets": pipe.asteroid.n_facets,
           "weights": list(cfg.weights), "camera": cfg.camera, "dt_min": cfg.dt_min,
           "occlusion": cfg.occlusion, "shadows": cfg.shadows, "n_ref": cfg.n_ref,
           "feasible": bool(b.feasible), "J": b.cost, "N": b.design.N, "T_days": b.design.T_days,
           "seconds": seconds, "evals": res.n_evals, "x_best": np.asarray(res.x_best).tolist(),
           # an SA run whose whole budget goes to its warm start (part NI) reports no step: no curve
           "curve_J": res.history.get("best_cost", np.array([])).tolist(), "params": job.params,
           "T0": res.extra.get("T0"), "alpha_used": res.extra.get("alpha"), "stop": res.stop_reason,
           # computational cost independent of the machine: spacecraft actually simulated (cache misses)
           "tracks": pipe.engine.n_tracks, "engine_runs": pipe.engine.n_simulations, "rejected": pipe.n_rejected,
           **job.extra}
    if job.algo == "sa":
        row["sa_params"] = job.params                                         # name used by Milestone 3 rows
    if job.algo == "ga":
        e = res.extra
        row.update({"pop_size": e["pop_size"], "n_elite": e["n_elite"], "n_crossover": e["n_crossover"],
                    "n_mutation": e["n_mutation"], "generations": e["generations_run"],
                    "improvements_by_operator": e["improvements_by_operator"],
                    "gen": {k: v.tolist() for k, v in e["generations"].items()}})
    if b.result is not None:
        c = b.result.cost
        row.update({"C": c.C, "Z": c.Z, "E": c.E, "f_time": c.f_time})
    if job.body == "eros":                                                    # V7: finer mesh
        fine = Pipeline(make_asteroid(CHECK_SHAPE), cfg).evaluate(res.x_best)
        if fine.result is not None:
            row.update({"J_fine": fine.cost, "C_fine": fine.result.C})
    return row


def run_all(jobs: list[Job], workers: int) -> list[dict]:
    """Run the jobs in parallel; every finished run is appended to runs_partial.jsonl at once,
    so an interrupted validation keeps what it has done."""
    rows, t0 = [], time.perf_counter()
    partial = OUT / "runs_partial.jsonl"
    with ProcessPoolExecutor(max_workers=workers) as ex, partial.open("w", encoding="utf-8") as log:
        futs = {ex.submit(run_job, j): j for j in jobs}
        failed = []
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:                      # keep the other runs; report and fail at the end
                j = futs[f]
                failed.append(f"{j.part} {j.label} {j.algo} seed {j.seed}: {exc!r}")
                print(f"  [{k}/{len(jobs)}] FAILED {failed[-1]}", flush=True)
                continue
            rows.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(f"  [{k}/{len(jobs)} {time.perf_counter() - t0:5.0f}s] {r['part']} {r['label']} {r['algo']} "
                  f"seed {r['seed']}: J {r['J']:.4f}  N {r['N']}  T {r['T_days']:.2f} d  "
                  f"C {100 * r.get('C', float('nan')):.1f} %", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed (finished runs are in {partial}):\n" + "\n".join(failed))
    return rows


# --------------------------------------------------------------------------
# V2, V3 (fast, in this process)
# --------------------------------------------------------------------------
def benchmarks(seeds: int, evals: int, algo: str = "sa") -> list[dict]:
    rows = []
    for name, dim in (("sphere", 5), ("rastrigin", 2), ("rastrigin", 5), ("ackley", 2), ("ackley", 5)):
        J = np.array([make(algo, max_evals=evals, seed=s).run(BenchmarkProblem(name, dim)).cost_best
                      for s in range(seeds)])
        rows.append({"algorithm": algo, "function": name, "dim": dim, "known_optimum": 0.0, "best": J.min(), "mean": J.mean(),
                     "std": J.std(ddof=1), "worst": J.max(), "runs": len(J), "below_0.1": float(np.mean(J < 0.1))})
        print(f"  {name:<9} d={dim}: best {J.min():.1e}  mean {J.mean():.1e}  below 0.1: {np.mean(J < 0.1):.0%}",
              flush=True)
    return rows


def saturation(n_designs: int) -> list[dict]:
    """Random designs of N = 1, 3, 5 flying for the whole T_max = 10 days on Eros."""
    cfg = for_body("eros")
    cfg.n_max = 5
    pipe = Pipeline(make_asteroid(SEARCH_SHAPE["eros"]), cfg)
    rng = np.random.default_rng(0)
    rows = []
    for n in (1, 3, 5):
        for _ in range(n_designs):
            x = pipe.random_solution(rng)
            x[0] = cfg.t_max_days
            for i in range(cfg.n_max):
                x[pipe.space.slot_index(i, "on")] = 1.0 if i < n else 0.0
            ev = pipe.evaluate(x)
            r = ev.result if ev.result is not None else pipe.engine.simulate(ev.design)
            rows.append({"N": n, "C": r.C, "J": r.J, "feasible": ev.feasible})
        C = np.array([r["C"] for r in rows if r["N"] == n])
        print(f"  N = {n}: random designs image {100 * C.mean():.0f} % of A* on average (min {100 * C.min():.0f}, "
              f"max {100 * C.max():.0f}, >= 95 %: {np.mean(C >= 0.95):.0%})", flush=True)
    return rows


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def mean(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(v)) if v else float("nan")


def summarise_vs_random(rows: list[dict]) -> list[dict]:
    from scipy.stats import mannwhitneyu
    out = []
    for label, body, n_max, t_max in VS_RANDOM:
        sa = [r for r in rows if r["label"] == label and r["algo"] == "sa"]
        rs = [r for r in rows if r["label"] == label and r["algo"] == "random"]
        if not sa or not rs:
            continue
        p = float(mannwhitneyu([r["J"] for r in sa], [r["J"] for r in rs], alternative="less").pvalue)
        out.append({"setting": label, "N_max": n_max, "T_max_days": t_max, "runs": len(sa),
                    **{f"{k}_SA": mean(sa, k) for k in ("J", "C", "Z", "E", "N", "T_days")},
                    **{f"{k}_random": mean(rs, k) for k in ("J", "C", "Z", "E", "N", "T_days")},
                    "worst_SA_J": max(r["J"] for r in sa), "best_random_J": min(r["J"] for r in rs),
                    "SA_feasible": sum(r["feasible"] for r in sa), "p_SA_less": p})
    return out


def summarise_fixed(rows: list[dict]) -> list[dict]:
    out = []
    for label, *_ in FIXED_N:
        for n in sorted({r["N_max"] for r in rows if r["label"] == label and r["fixed_n"]}):
            rr = [r for r in rows if r["label"] == label and r["fixed_n"] and r["N_max"] == n]
            out.append({"setting": label, "N": n, "runs": len(rr), "best_J": min(r["J"] for r in rr),
                        "mean_J": mean(rr, "J"), "mean_C": mean(rr, "C"), "mean_T_days": mean(rr, "T_days")})
        free = [r for r in rows if r["label"] == label and not r["fixed_n"]]
        if free:
            out.append({"setting": label, "N": "SA chooses", "runs": len(free), "best_J": min(r["J"] for r in free),
                        "mean_J": mean(free, "J"), "mean_C": mean(free, "C"), "mean_T_days": mean(free, "T_days"),
                        "N_chosen": [r["N"] for r in free]})
    return out


def summarise_params(rows: list[dict]) -> list[dict]:
    out = []
    for label, params in PARAM_STUDY:
        rr = [r for r in rows if r["label"] == label]
        if rr:
            J = np.array([r["J"] for r in rr])
            out.append({"setting": label, **params, "runs": len(rr), "mean_J": float(J.mean()),
                        "std_J": float(J.std(ddof=1)) if len(J) > 1 else 0.0, "best_J": float(J.min()),
                        "worst_J": float(J.max()), "mean_C": mean(rr, "C"), "mean_N": mean(rr, "N"),
                        "mean_T_days": mean(rr, "T_days"), "mean_evals": mean(rr, "evals"),
                        "mean_alpha": mean(rr, "alpha_used")})
    return out


def summarise_ga_params(rows: list[dict]) -> list[dict]:
    out = []
    for label, params in GA_PARAM_STUDY:
        rr = [r for r in rows if r["label"] == label]
        if rr:
            J = np.array([r["J"] for r in rr])
            out.append({"setting": label, **params, "runs": len(rr), "mean_J": float(J.mean()),
                        "std_J": float(J.std(ddof=1)) if len(J) > 1 else 0.0, "best_J": float(J.min()),
                        "worst_J": float(J.max()), "mean_C": mean(rr, "C"), "mean_N": mean(rr, "N"),
                        "mean_T_days": mean(rr, "T_days"), "mean_tracks": mean(rr, "tracks"),
                        "mean_seconds": mean(rr, "seconds"), "pop_size": rr[0].get("pop_size"),
                        "generations": mean(rr, "generations")})
    return out


def evals_to_reach(row: dict, target: float) -> float:
    """Evaluations until the best-so-far J first reaches the target (nan if never).  SA does not
    record its warm-start samples, so its curve starts after them: the offset puts both on the
    same axis of evaluations actually spent."""
    c = np.asarray(row["curve_J"], float)
    hit = np.flatnonzero(c <= target)
    return float(hit[0] + 1 + row["evals"] - len(c)) if len(hit) else float("nan")


def summarise_sa_vs_ga(rows: list[dict]) -> list[dict]:
    """Per V4 setting: SA (V4), GA (GV4) and random search (V4) with the performance indices."""
    from scipy.stats import mannwhitneyu
    out = []
    for label, body, n_max, t_max in VS_RANDOM:
        sa = [r for r in rows if r["label"] == label and r["algo"] == "sa" and r["part"] == "V4"]
        ga = [r for r in rows if r["label"] == label and r["algo"] == "ga" and r["part"] == "GV4"]
        rs = [r for r in rows if r["label"] == label and r["algo"] == "random" and r["part"] == "V4"]
        if not sa or not ga:
            continue
        # convergence index: evaluations to reach the worse of the two median final J values
        target = max(float(np.median([r["J"] for r in sa])), float(np.median([r["J"] for r in ga])))
        row = {"setting": label, "N_max": n_max, "T_max_days": t_max, "runs_SA": len(sa), "runs_GA": len(ga),
               "target_J": target}
        for tag, rr in (("SA", sa), ("GA", ga), ("random", rs)):
            if not rr:
                continue
            row.update({f"{k}_{tag}": mean(rr, k) for k in ("J", "C", "Z", "E", "N", "T_days", "seconds", "tracks")})
            row[f"std_J_{tag}"] = float(np.std([r["J"] for r in rr], ddof=1)) if len(rr) > 1 else 0.0
            row[f"worst_J_{tag}"] = max(r["J"] for r in rr)
            row[f"NT_{tag}"] = float(np.mean([r["N"] * r["T_days"] for r in rr]))
            row[f"evals_to_target_{tag}"] = float(np.nanmedian([evals_to_reach(r, target) for r in rr]))
        Js, Jg = [r["J"] for r in sa], [r["J"] for r in ga]
        row["p_GA_less"] = float(mannwhitneyu(Jg, Js, alternative="less").pvalue)
        row["p_SA_less"] = float(mannwhitneyu(Js, Jg, alternative="less").pvalue)
        out.append(row)
    return out


def summarise_algos(rows: list[dict], labels: list[str]) -> list[dict]:
    """Per setting: every algorithm's mean J / C / N / T / seconds / tracks, and the one-sided
    Mann-Whitney p-values of GA < SA and SA < GA."""
    from scipy.stats import mannwhitneyu
    out = []
    for label in labels:
        rr = [r for r in rows if r["label"] == label]
        if not rr:
            continue
        row = {"setting": label, "N_max": rr[0]["N_max"], "T_max_days": rr[0]["T_max_days"], "evals": rr[0]["evals"]}
        for algo in ("sa", "ga", "random"):
            a = [r for r in rr if r["algo"] == algo]
            if a:
                row.update({f"{k}_{algo}": mean(a, k) for k in ("J", "C", "N", "T_days", "seconds", "tracks")})
                row[f"std_J_{algo}"] = float(np.std([r["J"] for r in a], ddof=1)) if len(a) > 1 else 0.0
                row[f"runs_{algo}"] = len(a)
        Js = [r["J"] for r in rr if r["algo"] == "sa"]
        Jg = [r["J"] for r in rr if r["algo"] == "ga"]
        if Js and Jg:
            row["p_GA_less"] = float(mannwhitneyu(Jg, Js, alternative="less").pvalue)
            row["p_SA_less"] = float(mannwhitneyu(Js, Jg, alternative="less").pvalue)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    skip = ("x_best", "curve_J", "gen")
    keys = [k for k in rows[0] if k not in skip]
    for r in rows[1:]:
        keys += [k for k in r if k not in keys and k not in skip]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def figures(raw: list[dict], sat: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vs = [r for r in raw if r["part"] == "V4"]
    if vs:
        labels = [s[0] for s in VS_RANDOM if any(r["label"] == s[0] for r in vs)]
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, (algo, col, off) in enumerate((("sa", "#d9534f", -0.18), ("random", "#8a8f9f", 0.18))):
            for i, lab in enumerate(labels):
                J = [r["J"] for r in vs if r["label"] == lab and r["algo"] == algo]
                ax.bar(i + off, np.mean(J), width=0.34, color=col, alpha=0.35,
                       label=("simulated annealing" if algo == "sa" else "random search") if i == 0 else None)
                ax.scatter(np.full(len(J), i + off), J, color=col, s=12, zorder=3)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel("best fitness J (lower is better)")
        ax.set_title("SA against random search with the same budget")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "fig_sa_vs_random.png", dpi=150)
        plt.close(fig)
    fx = [r for r in raw if r["part"] == "V6"]
    if fx:
        fig, axes = plt.subplots(1, len(FIXED_N), figsize=(4 * len(FIXED_N), 3.4))
        for ax, (label, *_r) in zip(np.atleast_1d(axes), FIXED_N):
            rr = [r for r in fx if r["label"] == label and r["fixed_n"]]
            ns = sorted({r["N_max"] for r in rr})
            best = [min(r["J"] for r in rr if r["N_max"] == n) for n in ns]
            ax.plot(ns, best, "o-", color="#d9534f")
            free = [r for r in fx if r["label"] == label and not r["fixed_n"]]
            if free:
                for r in free:
                    ax.scatter(r["N"], r["J"], marker="*", s=90, color="#1f77b4", zorder=3)
            ax.set_title(f"{label}: best J vs forced N")
            ax.set_xlabel("spacecraft N (forced)")
            ax.set_ylabel("best J")
        fig.tight_layout()
        fig.savefig(OUT / "fig_fixed_n.png", dpi=150)
        plt.close(fig)
    if sat:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for n, col in ((1, "#8a8f9f"), (3, "#1f77b4"), (5, "#d9534f")):
            ax.hist([100 * r["C"] for r in sat if r["N"] == n], bins=np.linspace(0, 100, 21), alpha=0.55,
                    color=col, label=f"N = {n}")
        ax.set_xlabel("coverage after 10 days [% of A*]")
        ax.set_ylabel("random designs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "fig_saturation.png", dpi=150)
        plt.close(fig)


def run_info(a, parts) -> dict:
    """What was run, on what code and machine (for the report)."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    packages = {}
    for name in ("numpy", "scipy", "trimesh", "rtree", "embreex", "pyvista", "pyvistaqt", "PyQt5", "matplotlib", "pytest"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    return {"command": " ".join(sys.argv), "parts": parts, "seeds": a.seeds, "evals": a.evals,
            "workers": a.workers, "quick": a.quick, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_t0": time.time(), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "machine": platform.platform(), "cpus": os.cpu_count(),
            "packages": packages,
            "threads_per_worker": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                  "MKL_NUM_THREADS")}}


def main():
    global OUT                                   # --out redirects every output file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*",
                    choices=["V2", "V3", "V4", "V5", "V6", "P", "G2", "GV4", "GV5", "GP", "H", "B", "M", "NI"],
                    help="parts to run (default: the SA parts V2-V6)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: results/validation)")
    ap.add_argument("--skip-random", action="store_true",
                    help="V4: rerun SA only and keep the random-search rows already in raw.json")
    a = ap.parse_args()
    OUT = a.out
    parts = a.only or ["V2", "V3", "V4", "V5", "V6"]
    if a.quick:
        a.seeds, a.evals = 2, 150
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = OUT / "raw.json", OUT / "summary.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else []
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    raw = [r for r in raw if r["part"] not in parts                        # re-running a part replaces it
           or (a.skip_random and r["part"] == "V4" and r["algo"] == "random")]
    info = run_info(a, parts)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    sat = []

    if "V2" in parts:
        print("V2 known optima (benchmark functions)", flush=True)
        summary["V2"] = benchmarks(3 if a.quick else 10, 2000 if a.quick else 10000)
        write_csv(OUT / "V2_benchmarks.csv", summary["V2"])
    if "G2" in parts:
        print("G2 known optima (benchmark functions), GA", flush=True)
        summary["G2"] = benchmarks(3 if a.quick else 10, 2000 if a.quick else 10000, algo="ga")
        write_csv(OUT / "G2_benchmarks.csv", summary["G2"])
    if "V3" in parts:
        print("V3 saturation (random designs)", flush=True)
        sat = saturation(10 if a.quick else 50)
        write_csv(OUT / "V3_saturation.csv", sat)
        summary["V3"] = [{"N": n, "mean_C": mean([r for r in sat if r["N"] == n], "C"),
                          "share_C_95": float(np.mean([r["C"] >= 0.95 for r in sat if r["N"] == n]))}
                         for n in (1, 3, 5)]
    jobs: list[Job] = []
    if "V4" in parts:
        for label, body, n_max, t_max in VS_RANDOM:
            for s in range(a.seeds):
                for algo in ("sa",) if a.skip_random else ("sa", "random"):
                    jobs.append(Job("V4", label, body, n_max, t_max, s, algo, a.evals))
    if "V5" in parts:
        for s in range(a.seeds):
            jobs.append(Job("V5", "Eros 10 d, all lit", "eros", 5, 10.0, s, "sa", a.evals, lighting=False))
    if "V6" in parts:
        for label, body, t_max, ns, seeds in FIXED_N:
            k = min(seeds or a.seeds, a.seeds)
            for n in ns:
                for s in range(k):
                    jobs.append(Job("V6", label, body, n, t_max, s, "sa", a.evals, fixed_n=True))
            fb, fn, ft = FREE_FOR_FIXED[label]
            for s in range(k):
                jobs.append(Job("V6", label, fb, fn, ft, s, "sa", a.evals))
    if "P" in parts:
        body, n_max, t_max = PARAM_CASE
        for label, params in PARAM_STUDY:
            for s in range(a.seeds):
                jobs.append(Job("P", label, body, n_max, t_max, s, "sa", a.evals, params=params))
    if "GV4" in parts:
        for label, body, n_max, t_max in VS_RANDOM:
            for s in range(a.seeds):
                jobs.append(Job("GV4", label, body, n_max, t_max, s, "ga", a.evals))
    if "GV5" in parts:
        for s in range(a.seeds):
            jobs.append(Job("GV5", "Eros 10 d, all lit", "eros", 5, 10.0, s, "ga", a.evals, lighting=False))
    if "H" in parts:
        for label, body, n_max, t_max in HARD:
            for s in range(a.seeds):
                for algo in ("sa", "ga", "random"):
                    jobs.append(Job("H", label, body, n_max, t_max, s, algo, a.evals))
    if "B" in parts:
        cases = {lab: (body, n_max, t_max) for lab, body, n_max, t_max in VS_RANDOM}
        for label, budget in BUDGET:
            body, n_max, t_max = cases[label]
            for s in range(a.seeds):
                for algo in ("sa", "ga"):
                    jobs.append(Job("B", label, body, n_max, t_max, s, algo, budget))
    if "M" in parts or "NI" in parts:
        label, body, n_max, t_max, mesh, n_seeds = MATCHED
        if "M" in parts:
            for s in range(n_seeds):
                for algo in ("sa", "ga"):
                    jobs.append(Job("M", label, body, n_max, t_max, s, algo, MATCHED_EVALS, mesh=mesh))
        if "NI" in parts:
            from spacecov.algorithms.ga import auto_pop_size, split
            P = auto_pop_size(1 + 5 * n_max)
            n_e = split(P, 0.05, 0.8)[0]
            ga_evals = P + (NATIVE_ITERATIONS - 1) * (P - n_e)          # 20 generations: 60 + 19 x 57 = 1143
            for s in range(n_seeds):
                jobs.append(Job("NI", label, body, n_max, t_max, s, "ga", ga_evals, mesh=mesh))
                jobs.append(Job("NI", label, body, n_max, t_max, s, "sa", NATIVE_ITERATIONS, mesh=mesh))
    if "GP" in parts:
        body, n_max, t_max = PARAM_CASE
        for label, params in GA_PARAM_STUDY:
            for s in range(a.seeds):
                jobs.append(Job("GP", label, body, n_max, t_max, s, "ga", a.evals, params=params))
    if jobs:
        jobs.sort(key=lambda j: -(j.t_max_days * (4 if j.body == "kleopatra" else 1)))   # long jobs first
        print(f"running {len(jobs)} optimisation runs on {a.workers} workers", flush=True)
        raw += run_all(jobs, a.workers)
    raw_path.write_text(json.dumps(raw))

    if any(r["part"] == "V4" for r in raw):
        summary["V4"] = summarise_vs_random([r for r in raw if r["part"] == "V4"])
        write_csv(OUT / "V4_sa_vs_random.csv", summary["V4"])
        write_csv(OUT / "V4_runs.csv", [r for r in raw if r["part"] == "V4"])
    if any(r["part"] == "V5" for r in raw):
        v5 = [r for r in raw if r["part"] == "V5"]
        summary["V5"] = {"runs": len(v5), "mean_C": mean(v5, "C"), "best_C": max(r.get("C", 0) for r in v5),
                         "mean_J": mean(v5, "J"), "mean_N": mean(v5, "N"), "mean_T_days": mean(v5, "T_days")}
        write_csv(OUT / "V5_all_lit.csv", v5)
    if any(r["part"] == "V6" for r in raw):
        summary["V6"] = summarise_fixed([r for r in raw if r["part"] == "V6"])
        write_csv(OUT / "V6_fixed_n.csv", summary["V6"])
    if any(r["part"] == "P" for r in raw):
        summary["P"] = summarise_params([r for r in raw if r["part"] == "P"])
        write_csv(OUT / "P_parameters.csv", summary["P"])
    if any(r["part"] == "GV5" for r in raw):
        g5 = [r for r in raw if r["part"] == "GV5"]
        summary["GV5"] = {"runs": len(g5), "mean_C": mean(g5, "C"), "best_C": max(r.get("C", 0) for r in g5),
                          "mean_J": mean(g5, "J"), "mean_N": mean(g5, "N"), "mean_T_days": mean(g5, "T_days")}
        write_csv(OUT / "GV5_all_lit.csv", g5)
    if any(r["part"] == "GP" for r in raw):
        summary["GP"] = summarise_ga_params([r for r in raw if r["part"] == "GP"])
        write_csv(OUT / "GP_parameters.csv", summary["GP"])
    if any(r["part"] == "H" for r in raw):
        summary["H"] = summarise_algos([r for r in raw if r["part"] == "H"], [h[0] for h in HARD])
        write_csv(OUT / "H_hard.csv", summary["H"])
    if any(r["part"] == "B" for r in raw):
        summary["B"] = summarise_algos([r for r in raw if r["part"] == "B"], [b[0] for b in BUDGET])
        write_csv(OUT / "B_budget.csv", summary["B"])
    for part, name in (("M", "M_matched.csv"), ("NI", "NI_native_iterations.csv")):
        if any(r["part"] == part for r in raw):
            summary[part] = summarise_algos([r for r in raw if r["part"] == part], [MATCHED[0]])
            write_csv(OUT / name, sorted([r for r in raw if r["part"] == part], key=lambda r: (r["seed"], r["algo"])))
    if any(r["part"] == "GV4" for r in raw):
        write_csv(OUT / "GV4_runs.csv", [r for r in raw if r["part"] == "GV4"])
        if any(r["part"] == "V4" for r in raw):
            summary["SA_GA"] = summarise_sa_vs_ga([r for r in raw if r["part"] in ("V4", "GV4")])
            write_csv(OUT / "SA_vs_GA.csv", summary["SA_GA"])
    fine = [r for r in raw if "C_fine" in r]
    if fine:
        d = np.array([abs(r["C_fine"] - r["C"]) for r in fine if "C" in r])
        summary["V7"] = {"designs": len(d), "mean_abs_dC_points": 100 * float(d.mean()),
                         "max_abs_dC_points": 100 * float(d.max())}
    summary["all_final_designs_feasible"] = all(r["feasible"] for r in raw)
    summary["settings"] = {"seeds": a.seeds, "evals": a.evals, "quick": a.quick,
                           "t_min_rule": "min(0.25 d, T_max / 4)", "check_mesh": CHECK_SHAPE,
                           "objective_weights": list(for_body("eros").weights), "n_ref": for_body("eros").n_ref,
                           "normalization": "fixed n_ref; Z and E may exceed 1 when N > n_ref"}
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["wall_seconds"] = round(time.time() - info["_t0"])
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    (OUT / f"run_info_{'_'.join(parts)}.json").write_text(json.dumps(info, indent=2))   # kept per set of parts
    figures(raw, sat)
    print(json.dumps({k: v for k, v in summary.items() if k in ("V5", "V7", "all_final_designs_feasible")},
                     indent=2), flush=True)
    for row in summary.get("V4", []):
        print(f"  {row['setting']:<24} J SA {row['J_SA']:.3f} / random {row['J_random']:.3f}   "
              f"C {100 * row['C_SA']:.1f} / {100 * row['C_random']:.1f} %   p = {row['p_SA_less']:.3g}")
    for row in summary.get("V6", []):
        print(f"  {row['setting']:<14} N = {row['N']!s:<11} best J {row['best_J']:.4f}  mean C "
              f"{100 * row['mean_C']:.1f} %  {row.get('N_chosen', '')}")


if __name__ == "__main__":
    main()

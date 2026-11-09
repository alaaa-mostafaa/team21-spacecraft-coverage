"""Build the tables (generated/*.tex) and charts (Figures/*.png) of the Milestone 3 report
from the validation results in results/validation/ (experiments/validate.py).

    python report/LaTeX/make_assets.py

Every number in the generated tables comes from results/validation/raw.json and
summary.json, so re-running the validation and this script updates the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import validate as V  # noqa: E402

RES = ROOT / "results" / "validation"
GEN = HERE / "generated"
FIG = HERE / "Figures"
SA_COL, RS_COL, ACC = "#2563a8", "#9aa0aa", "#c8553d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "figure.dpi": 100})


def load():
    """raw.json / summary.json of a finished validation; while it is still running, the runs
    finished so far (runs_partial.jsonl) with the summaries recomputed from them."""
    raw = json.loads((RES / "raw.json").read_text()) if (RES / "raw.json").exists() else []
    summary = json.loads((RES / "summary.json").read_text()) if (RES / "summary.json").exists() else {}
    part = RES / "runs_partial.jsonl"
    if part.exists():
        done = {(r["part"], r["label"], r["algo"], r["seed"], r["N_max"], json.dumps(r.get("sa_params", {}))) for r in raw}
        for line in part.read_text().splitlines():
            r = json.loads(line)
            if (r["part"], r["label"], r["algo"], r["seed"], r["N_max"], json.dumps(r.get("sa_params", {}))) not in done:
                raw.append(r)
    for key, part_name, fn in (("V4", "V4", V.summarise_vs_random), ("V6", "V6", V.summarise_fixed),
                               ("P", "P", V.summarise_params)):
        rr = [r for r in raw if r["part"] == part_name]
        if rr:
            try:
                summary[key] = fn(rr)
            except Exception as exc:                                  # incomplete groups while running
                print(f"summary {key}: {exc}")
    return raw, summary


def rows(raw, part, **kw):
    return [r for r in raw if r["part"] == part and all(r.get(k) == v for k, v in kw.items())]


def ms(v, fmt="{:.3f}"):
    v = np.asarray(v, float)
    return (fmt + r" $\pm$ " + fmt).format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\,\\%"


def tmax_label(t):
    return f"{t:g} d" if t >= 1 else f"{round(24 * t):d} h"


def dur(days):
    return f"{days:.2f} d" if days >= 1 else f"{24 * days:.1f} h"


def table(name, caption, label, header, body, colspec, size="footnotesize", note=None):
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\{size}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"]
    lines += [(" & ".join(map(str, r)) + r" \\") if r != "MID" else r"\midrule" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip", rf"\parbox{{0.95\textwidth}}{{\scriptsize {note}}}"]
    lines += [r"\end{table}", ""]
    (GEN / f"{name}.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------- tables
def tab_v2(summary):
    body = [[r["function"].capitalize(), r["dim"], f"{r['best']:.1e}", f"{r['mean']:.1e}", f"{r['worst']:.1e}",
             f"{100 * r['below_0.1']:.0f}\\,\\%"] for r in summary.get("V2", [])]
    table("tab_v2", "V2: \\ac{SA} on test functions with known minimum 0 (10 runs of 10\\,000 evaluations)",
          "tab:v2", ["Function", "Dim.", "Best", "Mean", "Worst", "Runs with $f<0.1$"], body, "@{}lccccc@{}")


def tab_v3(summary):
    import csv
    sat = list(csv.DictReader((RES / "V3_saturation.csv").open()))
    body = []
    for n in (1, 3, 5):
        C = np.array([float(r["C"]) for r in sat if int(r["N"]) == n])
        body.append([n, len(C), pct(C.mean(), 0), pct(C.min(), 0), pct(C.max(), 0), pct(np.mean(C >= 0.95), 0)])
    table("tab_v3", "V3: coverage of random designs flown for ten days around Eros", "tab:v3",
          ["$N$", "Designs", "Mean $C$", "Min", "Max", "Share with $C\\ge95\\,\\%$"], body, "@{}cccccc@{}")
    return sat


def tab_v4(summary):
    body = []
    for r in summary.get("V4", []):
        p = r["p_SA_less"]
        body.append([r["setting"].replace("N_max 10", "$N_{\\max}$ 10").replace(", $N", " ($N").replace(" 10", " 10)")
                     if "N_max" in r["setting"] else r["setting"], r["N_max"],
                     f"\\textbf{{{r['J_SA']:.3f}}} / {r['J_random']:.3f}",
                     f"\\textbf{{{100 * r['C_SA']:.1f}}} / {100 * r['C_random']:.1f}",
                     f"\\textbf{{{r['N_SA']:.1f}}} / {r['N_random']:.1f}",
                     f"\\textbf{{{r['T_days_SA']:.2f}}} / {r['T_days_random']:.2f}",
                     f"{r['worst_SA_J']:.3f} / {r['best_random_J']:.3f}", f"{p:.3f}"])
    table("tab_v4", "V4: \\ac{SA} (bold) against random search with the same budget; means over 5 runs each",
          "tab:v4", ["Setting", "$N_{\\max}$", "$J$ $\\downarrow$", "$C$ [\\%] $\\uparrow$", "$N$", "$T$ [d]",
                     "worst SA / best RS $J$", "$p$"], body, "@{}lccccccc@{}", size="scriptsize",
          note="$p$: one-sided Mann--Whitney test that \\ac{SA}'s $J$ is lower. "
               "``worst SA / best RS'': if the first number is lower, every \\ac{SA} run beat every random-search run.")


def tab_cases(raw):
    """Results of every case study (SA runs of V4, V5 and the free-N runs of V6)."""
    body = []
    groups = [(lab, b, n, t, rows(raw, "V4", label=lab, algo="sa")) for lab, b, n, t in V.VS_RANDOM]
    groups.append(("Eros 10 d, all lit (D)", "eros", 5, 10.0, rows(raw, "V5")))
    case_id = {"Eros 10 d": "A1", "Eros 1 d": "A2", "Eros 1 d, N_max 10": "A3", "Eros 12 h": "A4", "Eros 6 h": "A5",
               "Eros 2 h": "A6", "Kleopatra 3 d": "B1", "Kleopatra 3 d, N_max 10": "B2", "Kleopatra 12 h": "B3",
               "Eros 10 d, all lit (D)": "D"}
    for lab, body_name, n_max, t_max, rr in groups:
        if not rr:
            continue
        rr = sorted(rr, key=lambda r: r["seed"])
        body.append([case_id.get(lab, ""), body_name.capitalize(), tmax_label(t_max), n_max,
                     ", ".join(str(r["N"]) for r in rr), dur(np.mean([r["T_days"] for r in rr])),
                     ms([r["J"] for r in rr]), pct(np.mean([r["C"] for r in rr])),
                     f"{np.mean([r['Z'] for r in rr]):.2f}", f"{np.mean([r['E'] for r in rr]):.3f}",
                     f"{np.mean([r['f_time'] for r in rr]):.2f}",
                     f"{sum(r['feasible'] for r in rr)}/{len(rr)}"])
        if lab in ("Eros 2 h", "Kleopatra 12 h"):
            body.append("MID")
    table("tab_cases", "Fleets designed by \\ac{SA} in every case study (mean over runs; $N$ is listed per run)",
          "tab:caseresults", ["Case", "Body", "$T_{\\max}$", "$N_{\\max}$", "$N$ per run", "$T$", "$J$ (mean $\\pm$ std)",
                              "$C$", "$Z$", "$E$", "$T/T_{\\max}$", "Feas."], body, "@{}lllcllcccccc@{}",
          size="scriptsize")


def tab_v5(raw):
    lit = rows(raw, "V5")
    ref = rows(raw, "V4", label="Eros 10 d", algo="sa")
    body = [["Nallapu \\& Thangavelautham~\\cite{nallapu2019}", "one flyby (10 min)", "ignored", "8",
             "100\\,\\% (1\\,m)", "GA"]]
    for name, rr, sun in (("This work, case D", lit, "ignored"), ("This work, case A1", ref, "Sun + shadows")):
        if rr:
            body.append([name, "orbits, $T\\le10$\\,d", sun, ", ".join(str(r["N"]) for r in rr),
                         f"{100 * np.mean([r['C'] for r in rr]):.1f}\\,\\% (3\\,m)", "SA"])
    table("tab_v5", "V5: comparison with the closest prior work on Eros", "tab:v5",
          ["Work", "Trajectories", "Lighting", "Spacecraft", "Coverage (resolution)", "Algorithm"], body,
          "@{}llllll@{}", size="scriptsize")


def tab_v6(summary):
    body = []
    for r in summary.get("V6", []):
        n = r["N"] if r["N"] != "SA chooses" else "free ($N_{\\max}$ 10)"
        chosen = ", ".join(map(str, r.get("N_chosen", []))) if r.get("N_chosen") else ""
        body.append([r["setting"], n, r["runs"], f"{r['best_J']:.4f}", f"{r['mean_J']:.4f}", pct(r["mean_C"]),
                     dur(r["mean_T_days"]), chosen])
    table("tab_v6", "V6: forced fleet size against the fleet size \\ac{SA} chooses itself", "tab:v6",
          ["Mission", "$N$", "Runs", "Best $J$", "Mean $J$", "Mean $C$", "Mean $T$", "$N$ chosen"], body,
          "@{}llcccccl@{}", size="scriptsize")


def tab_runtime(raw):
    body = []
    for lab, b, n_max, t_max in V.VS_RANDOM:
        rr = rows(raw, "V4", label=lab, algo="sa")
        if rr:
            s = np.array([r["seconds"] for r in rr])
            body.append([lab.replace(", N_max 10", ""), b.capitalize(), tmax_label(t_max), n_max,
                         1 + 5 * n_max, int(round(t_max * 144)) + 1, f"{s.mean() / 60:.1f}",
                         f"{np.mean([r['evals'] for r in rr]) / s.mean():.1f}"])
    table("tab_runtime", "Size of the case studies and the cost of one \\ac{SA} run (2000 evaluations, one CPU core)",
          "tab:cost", ["Setting", "Body", "$T_{\\max}$", "$N_{\\max}$", "Variables", "Steps $N_t$", "Minutes per run",
                       "Evaluations / s"], body, "@{}lllccccc@{}", size="scriptsize")


def tab_params(summary):
    body = []
    for r in summary.get("P", []):
        name = (r["setting"].replace("alpha ", "$\\alpha=$ ").replace("p0 ", "$p_0=$ ")
                .replace("step0 ", "$\\sigma_0=$ ").replace("no warm start", "one random start"))
        body.append([name, f"{r['mean_alpha']:.3f}", r["runs"], ms([r["mean_J"]], "{:.4f}").split(" ")[0]
                     + f" $\\pm$ {r['std_J']:.4f}", f"{r['best_J']:.4f}", f"{r['worst_J']:.4f}", pct(r["mean_C"]),
                     f"{r['mean_N']:.1f}", f"{r['mean_evals']:.0f}"])
    table("tab_params", "Case E: effect of the \\ac{SA} parameters (Eros, $T_{\\max}=2$\\,d, $N_{\\max}=5$; one "
          "parameter changed at a time from the defaults)", "tab:params",
          ["Setting", "$\\alpha$ used", "Runs", "$J$ (mean $\\pm$ std)", "Best", "Worst", "Mean $C$", "Mean $N$",
           "Evaluations"], body, "@{}lcccccccc@{}", size="scriptsize")


# ---------------------------------------------------------------------------- charts
def fig_sa_vs_random(raw):
    vs = rows(raw, "V4")
    labels = [s[0] for s in V.VS_RANDOM if any(r["label"] == s[0] for r in vs)]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for algo, col, off, name in (("sa", SA_COL, -0.19, "simulated annealing"), ("random", RS_COL, 0.19, "random search")):
        for i, lab in enumerate(labels):
            J = [r["J"] for r in vs if r["label"] == lab and r["algo"] == algo]
            ax.bar(i + off, np.mean(J), width=0.36, color=col, alpha=0.45, label=name if i == 0 else None)
            ax.scatter(np.full(len(J), i + off), J, color=col, s=14, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(labels)), [lab.replace(", N_max 10", "\n$N_{max}$ 10") for lab in labels], fontsize=9)
    ax.set_ylabel("best fitness $J$ (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sa_vs_random.png")
    plt.close(fig)


def fig_fixed_n(raw):
    fx = rows(raw, "V6")
    if not fx:
        return
    fig, axes = plt.subplots(1, len(V.FIXED_N), figsize=(11, 3.3))
    for ax, (label, *_r) in zip(np.atleast_1d(axes), V.FIXED_N):
        rr = [r for r in fx if r["label"] == label and r["fixed_n"]]
        ns = sorted({r["N_max"] for r in rr})
        if not ns:
            continue
        best = [min(r["J"] for r in rr if r["N_max"] == n) for n in ns]
        meanJ = [np.mean([r["J"] for r in rr if r["N_max"] == n]) for n in ns]
        ax.plot(ns, best, "o-", color=ACC, label="best of runs (forced $N$)")
        ax.plot(ns, meanJ, "o--", color=ACC, alpha=0.4, label="mean of runs")
        k = int(np.argmin(best))
        for n, r in zip(ns, [r for r in rr]):
            pass
        cov = [np.mean([r["C"] for r in rr if r["N_max"] == n]) for n in ns]
        for n, c, b in zip(ns, cov, best):
            ax.annotate(f"{100 * c:.0f}%", (n, b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7,
                        color="#555")
        free = [r for r in fx if r["label"] == label and not r["fixed_n"]]
        for j, r in enumerate(free):
            ax.scatter(r["N"] + 0.08 * (j - len(free) / 2), r["J"], marker="*", s=80, color=SA_COL, zorder=4,
                       label="free $N$ (SA's choice)" if j == 0 else None)
        ax.scatter(ns[k], best[k], s=160, facecolor="none", edgecolor="black", zorder=5)
        ax.set_title(label)
        ax.set_xlabel("number of spacecraft $N$")
        ax.set_ylabel("$J$")
        ax.set_xticks(ns)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fixed_n.png")
    plt.close(fig)


def fig_saturation(sat):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for n, col in ((1, RS_COL), (3, SA_COL), (5, ACC)):
        ax.hist([100 * float(r["C"]) for r in sat if int(r["N"]) == n], bins=np.linspace(0, 100, 21), alpha=0.6,
                color=col, label=f"N = {n}")
    ax.set_xlabel("coverage after 10 days [% of the sunlit area]")
    ax.set_ylabel("random designs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_saturation.png")
    plt.close(fig)


def fig_convergence(raw):
    """Best-so-far J against evaluations, mean and spread over seeds, for every case (SA and RS)."""
    labs = [s for s in V.VS_RANDOM if rows(raw, "V4", label=s[0])]
    if not labs:
        return
    ncol = 3
    nrow = int(np.ceil(len(labs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), squeeze=False)
    for ax, (lab, *_r) in zip(axes.ravel(), labs):
        for algo, col, name in (("sa", SA_COL, "SA"), ("random", RS_COL, "random search")):
            rr = rows(raw, "V4", label=lab, algo=algo)
            if not rr:
                continue
            L = min(len(r["curve_J"]) for r in rr)
            C = np.array([r["curve_J"][:L] for r in rr])
            C = np.where(C > 10, np.nan, C)                         # infeasible start: not drawn
            x = np.arange(L) + (rr[0]["evals"] - L)
            m, s = np.nanmean(C, 0), np.nanstd(C, 0)
            ax.plot(x, m, color=col, label=name)
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.2, lw=0)
        ax.set_title(lab.replace("N_max", "$N_{max}$"), fontsize=9)
        ax.set_xlabel("evaluations", fontsize=8)
        ax.set_ylabel("best $J$", fontsize=8)
    for ax in axes.ravel()[len(labs):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_convergence.png")
    plt.close(fig)


def fig_fleet_vs_time(raw):
    """N and T chosen by SA against T_max (Eros, N_max 10 where available)."""
    eros = [s for s in V.VS_RANDOM if s[1] == "eros"]
    if not any(rows(raw, "V4", label=s[0], algo="sa") for s in eros):
        return
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.2))
    rng = np.random.default_rng(1)
    for lab, b, n_max, t_max in eros:
        rr = rows(raw, "V4", label=lab, algo="sa")
        if not rr:
            continue
        mk = "o" if n_max == 10 else "s"
        x = np.full(len(rr), 24 * t_max) * np.exp(rng.normal(0, 0.04, len(rr)))
        a1.scatter(x, [r["N"] for r in rr], marker=mk, color=SA_COL, alpha=0.7, s=22)
        a2.scatter(x, [24 * r["T_days"] for r in rr], marker=mk, color=SA_COL, alpha=0.7, s=22)
        a3.scatter(x, [100 * r["C"] for r in rr], marker=mk, color=SA_COL, alpha=0.7, s=22)
    tt = np.array([2, 1000])
    a2.plot(tt, tt, color="#888", lw=0.8, ls="--", label="$T=T_{max}$")
    a2.legend(frameon=False, fontsize=8)
    for ax, yl in ((a1, "spacecraft sent $N$"), (a2, "mission duration $T$ [h]"), (a3, "coverage $C$ [%]")):
        ax.set_xscale("log")
        ax.set_xticks([2, 6, 12, 24, 240], ["2 h", "6 h", "12 h", "1 d", "10 d"])
        ax.set_xlabel("longest allowed mission $T_{max}$")
        ax.set_ylabel(yl)
    a2.set_yscale("log")
    a1.scatter([], [], marker="o", color=SA_COL, label="$N_{max}=10$")
    a1.scatter([], [], marker="s", color=SA_COL, label="$N_{max}=5$")
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fleet_vs_time.png")
    plt.close(fig)


def orbit_stats(raw):
    """Radii (in units of R_max) and inclinations of every spacecraft in the best SA designs."""
    from spacecov.design import DesignSpace
    r_max = {"eros": 17.63, "kleopatra": 114.17}
    out = {}
    for body in ("eros", "kleopatra"):
        a, inc = [], []
        for r in raw:
            if r["body"] != body or r["algo"] != "sa" or r["part"] not in ("V4", "V6") or not r["feasible"]:
                continue
            sp = DesignSpace(r["N_max"], 1, 1e9, 0, 1e9, fixed_n=r["fixed_n"])
            d = sp.decode(np.array(r["x_best"]))
            a += [o.a / r_max[body] for o in d.orbits]
            inc += [o.inc for o in d.orbits]
        if a:
            out[body] = {"a_over_R": np.array(a), "inc": np.array(inc)}
    if out:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3))
        for (body, d), col in zip(out.items(), (SA_COL, ACC)):
            a1.hist(d["a_over_R"], bins=np.linspace(1.5, 5.7, 22), alpha=0.6, color=col, label=body.capitalize(),
                    density=True)
            a2.hist(d["inc"], bins=np.linspace(0, 180, 19), alpha=0.6, color=col, label=body.capitalize(),
                    density=True)
        a1.set_xlabel("orbit radius $a$ / farthest surface point $R_{max}$")
        a2.set_xlabel("inclination $\\iota$ [deg]")
        a1.axvline(1.5, color="#555", ls=":", lw=1)
        for ax in (a1, a2):
            ax.set_ylabel("share of spacecraft")
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "fig_orbits.png")
        plt.close(fig)
    return out


def fig_params(raw):
    pr = rows(raw, "P")
    if not pr:
        return
    groups = [("cooling factor $\\alpha$", ["default", "alpha 0.80", "alpha 0.95", "alpha 0.98"]),
              ("initial acceptance $p_0$", ["p0 0.30", "default", "p0 0.80", "p0 0.95"]),
              ("initial step $\\sigma_0$", ["step0 0.05", "step0 0.10", "default", "step0 0.50"]),
              ("warm start", ["default", "no warm start"])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (title, labs) in zip(axes, groups):
        for i, lab in enumerate(labs):
            J = [r["J"] for r in pr if r["label"] == lab]
            if not J:
                continue
            col = SA_COL if lab == "default" else ACC
            ax.bar(i, np.mean(J), color=col, alpha=0.4, width=0.6)
            ax.errorbar(i, np.mean(J), yerr=np.std(J, ddof=1) if len(J) > 1 else 0, color=col, capsize=3)
            ax.scatter(np.full(len(J), i), J, color=col, s=10, zorder=3)
        names = [("default\n(auto, " + f"{np.mean([r['alpha_used'] for r in pr if r['label'] == 'default']):.2f})"
                  if (lab == "default" and "alpha" in title) else lab.split(" ", 1)[-1] if lab != "default" else
                  "default") for lab in labs]
        names = ["no warm start" if lab == "no warm start" else n for lab, n in zip(labs, names)]
        ax.set_xticks(range(len(labs)), names, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("final best $J$")
    lo = min(r["J"] for r in pr)
    hi = max(r["J"] for r in pr)
    axes[0].set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.tight_layout()
    fig.savefig(FIG / "fig_params.png")
    plt.close(fig)
    # convergence per setting
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, (title, labs) in zip(axes, groups):
        for lab in labs:
            rr = [r for r in pr if r["label"] == lab]
            if not rr:
                continue
            L = min(len(r["curve_J"]) for r in rr)
            C = np.array([r["curve_J"][:L] for r in rr])
            C = np.where(C > 10, np.nan, C)
            x = np.arange(L) + (int(np.mean([r["evals"] for r in rr])) - L)
            ax.plot(x, np.nanmean(C, 0), lw=1.5 if lab == "default" else 1.0,
                    color="black" if lab == "default" else None, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("evaluations")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("best $J$ (mean of runs)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_params_convergence.png")
    plt.close(fig)


# ---------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, ec="#333", fs=9, style="round,pad=0.02,rounding_size=0.08", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text="", col="#333", tx=0.0, ty=0.0, fs=8, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, fontsize=fs, ha="center", va="center", color=col)


def polyarrow(ax, pts, col="#333"):
    """A right-angled connector through ``pts`` with an arrow head at the end."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=col, lw=1.2)
    arrow(ax, *pts[-2], *pts[-1], col=col)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    box(ax, 1.6, 1.9, 2.6, 1.6, "ALGORITHM\n(SA, random search;\nGA, PSO later)\n\nproposes $\\mathbf{x}$, reads $J$",
        "#dbe7f5", bold=False)
    box(ax, 5.5, 1.9, 2.8, 1.6, "FEASIBILITY\n\nrepair; checks $N$, $T$, $g_1$, $g_2$,\n"
        "inclination, $g_3$ separation", "#fbe3d6")
    box(ax, 9.4, 1.9, 2.8, 1.6, "ENGINE\n\norbits $\\rightarrow$ body frame $\\rightarrow$\nvisibility, light, battery"
        "\n$\\rightarrow$ $C, Z, E, T$ $\\rightarrow$ $J$", "#dff0d8")
    arrow(ax, 2.9, 2.2, 4.1, 2.2, "decision vector $\\mathbf{x}$", ty=0.18)
    arrow(ax, 6.9, 2.2, 8.0, 2.2, "feasible design", ty=0.18)
    arrow(ax, 4.1, 1.45, 2.9, 1.45, "infeasible:\n$J_0+\\rho\\sum g^2$", col=ACC, ty=-0.35)
    ax.plot([9.4, 9.4], [1.1, 0.35], color=SA_COL, lw=1.3)
    ax.plot([9.4, 1.6], [0.35, 0.35], color=SA_COL, lw=1.3)
    arrow(ax, 1.6, 0.35, 1.6, 1.1, col=SA_COL)
    ax.text(5.5, 0.2, "fitness $J$ (+ $g_4$ battery check) fed back to the algorithm", color=SA_COL, ha="center",
            fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png")
    plt.close(fig)


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    X = 3.3
    steps = [
        (10.1, "Start: draw 20 random feasible designs,\nevaluate them, $\\mathbf{x}\\leftarrow$ best", "#e8e8e8"),
        (9.0, "20 trial moves $\\rightarrow$ $\\Theta_0=-\\mathrm{median}(\\Delta J^+)/\\ln p_0$\n"
              "fit $\\alpha$ so that $\\Theta$ reaches $10^{-3}\\Theta_0$ at the budget", "#e8e8e8"),
        (7.85, "step $\\sigma=\\max(0.01,\\,0.25\\sqrt{\\Theta/\\Theta_0})$\n"
               "neighbour: Gaussian step on 1-2 active coordinates\n(switch flip adds/removes a spacecraft)", "#dbe7f5"),
        (6.65, "repair: wrap angles, reflect $a,\\iota,T$, keep $N\\geq1$", "#dbe7f5"),
        (5.6, "Feasibility $\\rightarrow$ Engine: $J(\\mathbf{y})$,  $\\Delta J=J(\\mathbf{y})-J(\\mathbf{x})$",
         "#dff0d8"),
    ]
    for y, t, c in steps:
        box(ax, X, y, 5.6, 0.85 if "\n" in t else 0.6, t, c, fs=8.5)
    half = [0.425 if "\n" in t else 0.3 for _, t, _ in steps]
    for k in range(len(steps) - 1):
        arrow(ax, X, steps[k][0] - half[k], X, steps[k + 1][0] + half[k + 1])
    # decision
    from matplotlib.patches import Polygon
    yd = 4.35
    ax.add_patch(Polygon([[X, yd + 0.55], [X + 1.9, yd], [X, yd - 0.55], [X - 1.9, yd]], fc="#fff3cd", ec="#333"))
    ax.text(X, yd, "$\\Delta J\\leq0$ or\nrand $<e^{-\\Delta J/\\Theta}$ ?", ha="center", va="center", fontsize=8.5)
    arrow(ax, X, 5.6 - 0.3, X, yd + 0.55)
    box(ax, 6.35, yd, 1.9, 0.55, "accept: $\\mathbf{x}\\leftarrow\\mathbf{y}$", "#dff0d8", fs=8.5)
    arrow(ax, X + 1.9, yd, 6.35 - 0.95, yd, "yes", ty=0.15)
    box(ax, X, 3.15, 4.4, 0.55, "update best (feasible first, then lowest $J$)", "#e8e8e8", fs=8.5)
    arrow(ax, X, yd - 0.55, X, 3.15 + 0.28, "no", tx=0.25)
    arrow(ax, 6.35, yd - 0.28, X + 2.2, 3.15 + 0.1)
    yd2 = 2.05
    ax.add_patch(Polygon([[X, yd2 + 0.5], [X + 1.7, yd2], [X, yd2 - 0.5], [X - 1.7, yd2]], fc="#fff3cd", ec="#333"))
    ax.text(X, yd2, "$L$ moves done\nat this $\\Theta$ ?", ha="center", va="center", fontsize=8.5)
    arrow(ax, X, 3.15 - 0.28, X, yd2 + 0.5)
    polyarrow(ax, [(X - 1.7, yd2), (X - 2.95, yd2), (X - 2.95, 7.85), (X - 2.8, 7.85)])
    ax.text(X - 2.55, 2.2, "no", fontsize=8)
    box(ax, X, 0.95, 3.6, 0.5, "$\\Theta\\leftarrow\\alpha\\,\\Theta$", "#dbe7f5", fs=8.5)
    arrow(ax, X, yd2 - 0.5, X, 0.95 + 0.25, "yes", tx=0.3)
    yd3 = 0.95
    box(ax, 6.55, 0.3, 1.6, 0.45, "return best", "#e8e8e8", fs=8.5)
    arrow(ax, X + 1.8, yd3 - 0.1, 6.55 - 0.5, 0.3 + 0.2, "budget used", fs=7.5, ty=0.18)
    polyarrow(ax, [(X - 1.8, 0.95), (X - 3.15, 0.95), (X - 3.15, 7.65), (X - 2.8, 7.65)])
    ax.text(X - 2.6, 0.75, "budget left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "sa_flowchart.png")
    plt.close(fig)


def numbers(raw, summary, stats):
    """LaTeX macros for numbers quoted in the text."""
    m = {}
    if raw:
        m["nRuns"] = str(len(raw))
        m["nFeasible"] = str(sum(r["feasible"] for r in raw))
    v7 = summary.get("V7")
    if v7:
        m["fineDesigns"] = str(v7["designs"])
        m["fineMeanDC"] = f"{v7['mean_abs_dC_points']:.2f}"
        m["fineMaxDC"] = f"{v7['max_abs_dC_points']:.2f}"
    for body, d in stats.items():
        b = body.capitalize()
        m[f"aMedian{b}"] = f"{np.median(d['a_over_R']):.1f}"
        m[f"incMedian{b}"] = f"{np.median(d['inc']):.0f}"
        m[f"incPolar{b}"] = f"{100 * np.mean((d['inc'] > 45) & (d['inc'] < 135)):.0f}"
        m[f"nOrbits{b}"] = str(len(d["inc"]))
    info = json.loads((RES / "run_info.json").read_text()) if (RES / "run_info.json").exists() else {}
    if info.get("wall_seconds"):
        m["valHours"] = f"{info['wall_seconds'] / 3600:.1f}"
    (GEN / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    return m


def main():
    GEN.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    fig_architecture()
    fig_flowchart()
    raw, summary = load()
    tab_v2(summary)
    sat = tab_v3(summary) if (RES / "V3_saturation.csv").exists() else []
    if sat:
        fig_saturation(sat)
    tab_v4(summary)
    tab_cases(raw)
    tab_v5(raw)
    tab_v6(summary)
    tab_runtime(raw)
    tab_params(summary)
    fig_sa_vs_random(raw)
    fig_fixed_n(raw)
    fig_convergence(raw)
    fig_fleet_vs_time(raw)
    fig_params(raw)
    stats = orbit_stats(raw)
    print(json.dumps(numbers(raw, summary, stats), indent=1))


if __name__ == "__main__":
    main()

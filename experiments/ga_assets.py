"""Figures and LaTeX tables of the Genetic Algorithm milestone, from results/validation/raw.json.

    python experiments/ga_assets.py                              # results/validation -> report/LaTeX
    python experiments/ga_assets.py --results DIR --report DIR

Reads the rows written by validate.py (parts V2, V4, V5 for SA / random search; G2, GP, GV4, GV5 for
the GA) and writes
    Figures/fig_ga_params.png          GA parameter study (part GP)
    Figures/fig_sa_vs_ga.png           best J of every run, SA / GA / random search, per case study
    Figures/fig_convergence_sa_ga.png  best J against evaluations, median and interquartile band
    Figures/fig_ga_generations.png     one GA run per generation: fitness, diversity, fleet size, operators
    Figures/fig_hard_budget.png        hard cases (part H) and the 5000-evaluation budget (part B)
    generated/tab_hard.tex             hard cases, SA / GA / random search
    generated/tab_budget.tex           2000 vs 5000 evaluations, SA and GA
    generated/tab_g2.tex               benchmark functions, SA (V2) and GA (G2)
    generated/tab_gp.tex               parameter study table
    generated/tab_sa_ga.tex            SA vs GA performance indices per case study
    generated/numbers_ga.tex           numbers quoted in the text (\\newcommand)
Parts that are missing are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import validate as v  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-2, neutral grey for the baseline
COL = {"sa": "#2a78d6", "ga": "#eb6834", "random": "#8a8f9f"}
NAME = {"sa": "simulated annealing", "ga": "genetic algorithm", "random": "random search"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight"})


def runs(raw, part, algo=None, label=None):
    return [r for r in raw if r["part"] == part and (algo is None or r["algo"] == algo)
            and (label is None or r["label"] == label)]


def curve(r) -> tuple[np.ndarray, np.ndarray]:
    """(evaluation index, best-so-far J).  SA does not report its warm-start samples, so its
    curve starts later; it is aligned on the last evaluation."""
    c = np.asarray(r["curve_J"], float)
    return np.arange(r["evals"] - len(c) + 1, r["evals"] + 1), c


# ------------------------------------------------------------------------------------- figures
def fig_params(raw, out, plt):
    rows = v.summarise_ga_params(runs(raw, "GP"))
    if not rows:
        return
    labels = [r["setting"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    default = next(r["mean_J"] for r in rows if r["setting"] == "default")
    ax.axvline(default, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for yi, r in zip(y, rows):
        J = [x["J"] for x in runs(raw, "GP", label=r["setting"])]
        ax.scatter(J, np.full(len(J), yi), s=10, color=COL["ga"], alpha=0.45, lw=0, zorder=2)
        ax.plot([r["mean_J"]] * 2, [yi - 0.32, yi + 0.32], color=INK, lw=2, solid_capstyle="round", zorder=3)
    ax.set_yticks(y, labels)
    for t in ax.get_yticklabels():
        if t.get_text() == "default":
            t.set_fontweight("bold")
    ax.set_xlabel("best fitness J of each run (lower is better); bar = mean of the seeds")
    ax.set_title("GA parameter study: Eros, $N_{\\max}=5$, $T_{\\max}$ = 2 d, one setting changed at a time",
                 loc="left")
    ax.text(default, len(rows) - 0.35, " default", color=MUTED, fontsize=8, va="bottom")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig_ga_params.png")
    plt.close(fig)


def fig_sa_vs_ga(raw, out, plt):
    labels = [s[0] for s in v.VS_RANDOM if runs(raw, "GV4", "ga", s[0]) and runs(raw, "V4", "sa", s[0])]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    groups = (("sa", "V4", -0.25), ("ga", "GV4", 0.0), ("random", "V4", 0.25))
    for algo, part, off in groups:
        for i, lab in enumerate(labels):
            J = [r["J"] for r in runs(raw, part, algo, lab)]
            if not J:
                continue
            ax.scatter(np.full(len(J), i + off), J, s=16, color=COL[algo], lw=0, alpha=0.8, zorder=3,
                       label=NAME[algo] if i == 0 else None)
            ax.plot([i + off - 0.09, i + off + 0.09], [np.mean(J)] * 2, color=COL[algo], lw=2.2,
                    solid_capstyle="round", zorder=4)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("best fitness J (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)       # above the plot, clear of the dots
    fig.savefig(out / "fig_sa_vs_ga.png")
    plt.close(fig)


def fig_convergence(raw, out, plt):
    labels = [s[0] for s in v.VS_RANDOM if runs(raw, "GV4", "ga", s[0]) and runs(raw, "V4", "sa", s[0])]
    if not labels:
        return
    n = len(labels)
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(10, 2.6 * np.ceil(n / cols)), squeeze=False)
    for ax, lab in zip(axes.ravel(), labels):
        for algo, part in (("random", "V4"), ("sa", "V4"), ("ga", "GV4")):
            rr = runs(raw, part, algo, lab)
            if not rr:
                continue
            xs = [curve(r) for r in rr]
            start = max(x[0][0] for x in xs)
            grid = np.arange(start, min(x[0][-1] for x in xs) + 1)
            Y = np.array([c[np.searchsorted(e, grid)] for e, c in xs])
            lo, med, hi = np.percentile(Y, [25, 50, 75], axis=0)
            ax.fill_between(grid, lo, hi, color=COL[algo], alpha=0.15, lw=0)
            ax.plot(grid, med, color=COL[algo], lw=2 if algo != "random" else 1.4, label=NAME[algo])
        ax.set_title(lab, loc="left")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best J so far")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes.ravel()[0].legend(loc="upper right")
    fig.suptitle("Convergence: median best J over the seeds, band = interquartile range", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence_sa_ga.png")
    plt.close(fig)


def fig_generations(raw, out, plt, label="Eros 10 d"):
    rr = [r for r in runs(raw, "GV4", "ga", label) if "gen" in r]
    if not rr:
        return
    r = sorted(rr, key=lambda x: x["J"])[len(rr) // 2]                     # the median run
    g = {k: np.asarray(val, float) for k, val in r["gen"].items()}
    gen = np.arange(len(g["best"]))
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 2.9))
    ax[0].plot(gen, g["mean"], color=COL["ga"], lw=1.4, alpha=0.45)
    ax[0].plot(gen, g["best"], color=COL["ga"], lw=2)
    ax[0].set_ylabel("fitness J")
    ax[0].set_title("best (dark), population mean (light)", loc="left", fontsize=8.5)
    ax[1].plot(gen, g["diversity"], color=INK, lw=2)
    ax[1].set_ylabel("diversity")
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("spread of the genes", loc="left", fontsize=8.5)
    ax[2].fill_between(gen, g["N_min"], g["N_max"], color=COL["ga"], alpha=0.15, lw=0)
    ax[2].plot(gen, g["N_mean"], color=COL["ga"], lw=2)
    ax[2].set_ylabel("fleet size N")
    ax[2].set_title("mean (line), min to max (band)", loc="left", fontsize=8.5)
    for a in ax[:3]:
        a.set_xlabel("generation")
    # where the improvements came from, over every GA run of the case studies
    tot: dict[str, int] = {}
    for x in runs(raw, "GV4", "ga"):
        for k, c in x.get("improvements_by_operator", {}).items():
            tot[k] = tot.get(k, 0) + c
    tot.pop("initial", None)
    if tot:
        names = list(tot)
        vals = np.array([tot[k] for k in names], float)
        ax[3].barh(range(len(names)), 100 * vals / vals.sum(), color=COL["ga"], height=0.55)
        ax[3].set_yticks(range(len(names)), names)
        ax[3].set_xlabel("share of new best designs [%]")
        ax[3].grid(axis="y", visible=False)
        for i, val in enumerate(100 * vals / vals.sum()):
            ax[3].text(val + 1, i, f"{val:.0f} %", va="center", fontsize=8, color=INK)
        ax[3].set_xlim(0, 100)
        ax[3].set_title("source of new best designs (all runs)", loc="left", fontsize=8.5)
    fig.suptitle(f"GA run with the median result, {label} (seed {r['seed']}, J = {r['J']:.4f})",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_ga_generations.png")
    plt.close(fig)


def fig_hard_budget(raw, out, plt):
    hard = [h[0] for h in v.HARD if runs(raw, "H", "ga", h[0])]
    bud = [b[0] for b in v.BUDGET if runs(raw, "B", "ga", b[0])]
    if not hard and not bud:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for algo, off in (("sa", -0.25), ("ga", 0.0), ("random", 0.25)):
        for i, lab in enumerate(hard):
            J = [r["J"] for r in runs(raw, "H", algo, lab)]
            ax.scatter(np.full(len(J), i + off), J, s=16, color=COL[algo], lw=0, alpha=0.8, zorder=3,
                       label=NAME[algo] if i == 0 else None)
            ax.plot([i + off - 0.09, i + off + 0.09], [np.mean(J)] * 2, color=COL[algo], lw=2.2,
                    solid_capstyle="round", zorder=4)
    ax.set_xticks(range(len(hard)), [h.replace(", ", "\n") for h in hard])
    ax.set_ylabel("best fitness J (lower is better)")
    ax.set_title("Hard cases: Kleopatra, short missions (2000 evaluations)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=3)
    ax = axes[1]
    for i, lab in enumerate(bud):
        for algo, part2, off in (("sa", "V4", -0.2), ("ga", "GV4", 0.2)):
            J2 = [r["J"] for r in runs(raw, part2, algo, lab)]
            J5 = [r["J"] for r in runs(raw, "B", algo, lab)]
            x2, x5 = i + off - 0.08, i + off + 0.08
            ax.scatter(np.full(len(J2), x2), J2, s=12, facecolor="none", edgecolor=COL[algo], lw=1, zorder=3)
            ax.scatter(np.full(len(J5), x5), J5, s=16, color=COL[algo], lw=0, zorder=3)
            ax.annotate("", xy=(x5, np.mean(J5)), xytext=(x2, np.mean(J2)),
                        arrowprops=dict(arrowstyle="-|>", color=COL[algo], lw=1.6), zorder=4)
    ax.set_xticks(range(len(bud)), bud)
    ax.set_ylabel("best fitness J")
    ax.set_title("Budget: 2000 (open) to 5000 evaluations (filled), arrow = mean", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_hard_budget.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------- tables
def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def tab_g2(summary, gen):
    sa = {(r["function"], r["dim"]): r for r in summary.get("V2", [])}
    ga = {(r["function"], r["dim"]): r for r in summary.get("G2", [])}
    if not ga:
        return
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{mean best value} & \multicolumn{2}{c}{worst run} & runs below 0.1 \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Function & $n$ & SA & GA & SA & GA & SA / GA \\", r"\midrule"]
    for k, g in ga.items():
        s = sa.get(k, {})
        lines.append(f"{k[0].capitalize()} & {k[1]} & {s.get('mean', float('nan')):.1e} & {g['mean']:.1e} & "
                     f"{s.get('worst', float('nan')):.1e} & {g['worst']:.1e} & "
                     f"{100 * s.get('below_0.1', float('nan')):.0f} / {100 * g['below_0.1']:.0f}\\,\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_g2.tex").write_text("\n".join(lines) + "\n")


def tab_gp(raw, gen):
    rows = v.summarise_ga_params(runs(raw, "GP"))
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Setting & mean $J$ & std $J$ & best $J$ & mean $C$ [\%] & mean $N$ & tracks / run \\", r"\midrule"]
    best = min(r["mean_J"] for r in rows)
    for r in rows:
        mJ = f(r["mean_J"], 4)
        if r["mean_J"] == best:
            mJ = r"\textbf{" + mJ + "}"
        name = r["setting"].replace("%", r"\%").replace("lambda 1/2", r"$\lambda=\frac12$")
        if r["setting"] == "default":
            name = r"\textbf{default}"
        lines.append(f"{name} & {mJ} & {f(r['std_J'], 4)} & {f(r['best_J'], 4)} & {100 * r['mean_C']:.1f} & "
                     f"{r['mean_N']:.1f} & {r['mean_tracks']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_gp.tex").write_text("\n".join(lines) + "\n")


def tab_sa_ga(raw, gen):
    rows = v.summarise_sa_vs_ga([r for r in raw if r["part"] in ("V4", "GV4")])
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{mean $J$} & \multicolumn{2}{c}{mean $C$ [\%]} & \multicolumn{2}{c}{$N\cdot T$ [d]}"
             r" & \multicolumn{2}{c}{evals to target} & \multicolumn{2}{c}{tracks} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Case study & SA & GA & SA & GA & SA & GA & SA & GA & SA & GA \\", r"\midrule"]
    for r in rows:
        js, jg = r["J_SA"], r["J_GA"]
        J1, J2 = f(js, 4), f(jg, 4)
        if r["p_GA_less"] < 0.05:
            J2 = r"\textbf{" + J2 + "}"
        elif r["p_SA_less"] < 0.05:
            J1 = r"\textbf{" + J1 + "}"
        name = r["setting"].replace("N_max", r"$N_{\max}=$")
        lines.append(f"{name} & {J1} & {J2} & {100 * r['C_SA']:.1f} & {100 * r['C_GA']:.1f} & "
                     f"{r['NT_SA']:.2f} & {r['NT_GA']:.2f} & {f(r['evals_to_target_SA'], 0)} & "
                     f"{f(r['evals_to_target_GA'], 0)} & {f(r.get('tracks_SA'), 0)} & {f(r.get('tracks_GA'), 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_sa_ga.tex").write_text("\n".join(lines) + "\n")


def tab_hard(summary, gen):
    rows = summary.get("H", [])
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{mean $J$} & \multicolumn{3}{c}{mean $C$ [\%]} & \multicolumn{2}{c}{mean $N$} & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"Case & SA & GA & random & SA & GA & random & SA & GA & $p$ (GA $<$ SA) \\", r"\midrule"]
    for r in rows:
        name = r["setting"].replace("N_max", r"$N_{\max}=$")
        lines.append(f"{name} & {r['J_sa']:.4f} & {r['J_ga']:.4f} & {r['J_random']:.4f} & "
                     f"{100 * r['C_sa']:.1f} & {100 * r['C_ga']:.1f} & {100 * r['C_random']:.1f} & "
                     f"{r['N_sa']:.1f} & {r['N_ga']:.1f} & {r['p_GA_less']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_hard.tex").write_text("\n".join(lines) + "\n")


def tab_budget(raw, summary, gen):
    rows = summary.get("B", [])
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{SA mean $J$} & \multicolumn{2}{c}{GA mean $J$} & \multicolumn{2}{c}{$p$ at 5000} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Case & 2000 & 5000 & 2000 & 5000 & GA $<$ SA & SA $<$ GA \\", r"\midrule"]
    for r in rows:
        lab = r["setting"]
        s2 = np.mean([x["J"] for x in runs(raw, "V4", "sa", lab)])
        g2 = np.mean([x["J"] for x in runs(raw, "GV4", "ga", lab)])
        pg = f"{r['p_GA_less']:.3f}"
        if r["p_GA_less"] < 0.05:
            pg = r"\textbf{" + pg + "}"
        lines.append(f"{lab} & {s2:.4f} & {r['J_sa']:.4f} & {g2:.4f} & {r['J_ga']:.4f} & {pg} & {r['p_SA_less']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_budget.tex").write_text("\n".join(lines) + "\n")


def numbers(raw, summary, gen):
    """\\newcommand macros for numbers quoted in the text."""
    m = {}
    rows = v.summarise_sa_vs_ga([r for r in raw if r["part"] in ("V4", "GV4")])
    if rows:
        m["gaWins"] = sum(r["p_GA_less"] < 0.05 for r in rows)
        m["saWins"] = sum(r["p_SA_less"] < 0.05 for r in rows)
        m["gaTies"] = len(rows) - m["gaWins"] - m["saWins"]
        m["gaSettings"] = len(rows)
        m["gaMeanJ"] = f"{np.mean([r['J_GA'] for r in rows]):.4f}"
        m["saMeanJ"] = f"{np.mean([r['J_SA'] for r in rows]):.4f}"
    if "GV5" in summary:
        m["gaAllLitC"] = f"{100 * summary['GV5']['mean_C']:.1f}"
        m["gaAllLitBestC"] = f"{100 * summary['GV5']['best_C']:.1f}"
    g = runs(raw, "GV4", "ga")
    if g:
        m["gaFeasible"] = sum(r["feasible"] for r in g)
        m["gaRuns"] = len(g)
    lines = [f"\\newcommand{{\\{k}}}{{{val}}}" for k, val in m.items()]
    (gen / "numbers_ga.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "validation")
    ap.add_argument("--report", type=Path, default=ROOT / "report" / "LaTeX")
    a = ap.parse_args()
    raw = json.loads((a.results / "raw.json").read_text())
    summary_path = a.results / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    figs, gen = a.report / "Figures", a.report / "generated"
    figs.mkdir(parents=True, exist_ok=True)
    gen.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    for fn in (fig_params, fig_sa_vs_ga, fig_convergence, fig_generations, fig_hard_budget):
        fn(raw, figs, plt)
    tab_g2(summary, gen)
    tab_gp(raw, gen)
    tab_sa_ga(raw, gen)
    tab_hard(summary, gen)
    tab_budget(raw, summary, gen)
    numbers(raw, summary, gen)
    print("written:", *sorted(p.name for p in figs.glob("fig_*ga*.png")), *sorted(p.name for p in gen.glob("*g*.tex")))


if __name__ == "__main__":
    main()

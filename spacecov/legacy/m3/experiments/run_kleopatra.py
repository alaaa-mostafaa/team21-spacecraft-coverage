"""Second target: 216 Kleopatra (radar shape model), a 3-day mission.

Same problem, same SA, same settings as for Eros; only the body (shape, mass, spin) and the
size-scaled distances change (spacecov.config.BODIES).  Kleopatra is 6.5x longer than Eros, has 46x
its surface area, and fills only 64 % of its convex hull (Eros 84 %): a much more non-convex target.

    python -m spacecov.legacy.m3.experiments.run_kleopatra.py                 # everything (~45 min on 11 workers)
    python -m spacecov.legacy.m3.experiments.run_kleopatra.py --seeds 2 --evals 600   # quick look
Results: results/kleopatra/  (raw_all.json, all.csv, best_kleopatra_Nmax*.json)
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from run_experiments import Job, mission_config, run_all, write_runs  # noqa: E402
from spacecov.legacy.m3.algorithms import SAParams  # noqa: E402
from spacecov.legacy.m3.design import OrbitalSolution  # noqa: E402
from spacecov.legacy.m3.scenario import Scenario  # noqa: E402

BODY, DAYS = "kleopatra", 3.0


def jobs(seeds, evals):
    base = asdict(SAParams(max_evals=evals))
    kw = dict(days=DAYS, body=BODY)
    out = []
    for n in (5, 10):
        for s in range(seeds):
            out.append(Job("cases", "SA", n, 0.0, s, base, extra={"setting": f"N_max={n}"}, **kw))
            out.append(Job("vs_random", "random search", n, 0.0, s, base, random_search=True,
                           extra={"setting": f"N_max={n}"}, **kw))
    fixed = dict(base, max_evals=max(200, evals * 2 // 3))
    for n in range(1, 6):
        for s in range(min(3, seeds)):
            out.append(Job("fixed", "SA", n, 0.0, s, fixed, fixed_n=True, extra={"setting": f"N={n}"}, **kw))
    return out


def save_best(rows, out, n_max):
    r = min(rows, key=lambda r: (not r["feasible"], r["J"]))
    sc = Scenario(name=f"Kleopatra, SA best of {len(rows)} runs: N_max = {n_max}, {DAYS:g} days "
                       f"(N = {r['N']}, C = {100 * r['C']:.1f} %, J = {r['J']:.4f})",
                  asteroid={"source": BODY}, config=mission_config(n_max, 0.0, True, DAYS, BODY))
    sc.fleet = OrbitalSolution.from_vector(r["x_best"]).active()
    sc.save(out / f"best_kleopatra_Nmax{n_max}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    out = ROOT / "results" / "kleopatra"
    out.mkdir(parents=True, exist_ok=True)
    res = run_all(jobs(a.seeds, a.evals), a.workers, "kleopatra")
    (out / "raw_all.json").write_text(json.dumps(res), encoding="utf-8")
    write_runs(out / "all.csv", [dict(r, label=f"{r['group']} | {r['setting']} | {r['label']}") for r in res])
    for n in (5, 10):
        save_best([r for r in res if r["group"] == "cases" and r["N_max"] == n], out, n)

    print(f"\n{'group':10s} {'setting':10s} {'method':14s} {'J':>8s} {'C %':>6s} {'N':>4s}")
    for g, s, l in sorted({(r["group"], r["setting"], r["label"]) for r in res}):
        rr = [r for r in res if (r["group"], r["setting"], r["label"]) == (g, s, l)]
        print(f"{g:10s} {s:10s} {l:14s} {np.mean([r['J'] for r in rr]):8.4f} {100 * np.mean([r['C'] for r in rr]):6.1f} "
              f"{np.mean([r['N'] for r in rr]):4.1f}  ({', '.join(str(r['N']) for r in rr)})")
    for n in (5, 10):
        A = [r["J"] for r in res if r["group"] == "cases" and r["N_max"] == n]
        B = [r["J"] for r in res if r["group"] == "vs_random" and r["N_max"] == n]
        if len(A) > 1:
            print(f"N_max={n}: SA vs random p = {mannwhitneyu(A, B, alternative='less').pvalue:.4f}")


if __name__ == "__main__":
    main()

"""Very short missions (a few hours): does SA send more spacecraft when there is less time?

Eros spins once every 5.27 h, so a 2 h mission sees less than half a turn and a 12 h mission about
two turns.  Same problem, cameras (one frame every 10 min), weights and SA settings as everywhere.

    python -m spacecov.legacy.m3.experiments.run_short.py                 # everything
Results: results/short/  (raw_all.json, all.csv)
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

from run_experiments import Job, run_all, write_runs  # noqa: E402
from spacecov.legacy.m3.algorithms import SAParams  # noqa: E402

# (body, hours) ; N_max = 10 so SA is free to send a large fleet
SETTINGS = [("eros", 2), ("eros", 6), ("eros", 12), ("kleopatra", 12)]
FIXED = ("eros", 6, range(1, 9))


def jobs(seeds, evals):
    base = asdict(SAParams(max_evals=evals))
    out = []
    for body, hours in SETTINGS:
        kw = dict(days=hours / 24.0, body=body, extra={"setting": f"{body} {hours} h"})
        for s in range(seeds):
            out.append(Job("cases", "SA", 10, 0.0, s, base, **kw))
            out.append(Job("vs_random", "random search", 10, 0.0, s, base, random_search=True, **kw))
    body, hours, Ns = FIXED
    fixed = dict(base, max_evals=max(200, evals * 2 // 3))
    for n in Ns:
        for s in range(min(3, seeds)):
            out.append(Job("fixed", "SA", n, 0.0, s, fixed, fixed_n=True, days=hours / 24.0, body=body,
                           extra={"setting": f"{body} {hours} h, N={n}"}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    out = ROOT / "results" / "short"
    out.mkdir(parents=True, exist_ok=True)
    res = run_all(jobs(a.seeds, a.evals), a.workers, "short")
    (out / "raw_all.json").write_text(json.dumps(res), encoding="utf-8")
    write_runs(out / "all.csv", [dict(r, label=f"{r['group']} | {r['setting']} | {r['label']}") for r in res])

    print(f"\n{'group':10s} {'setting':22s} {'method':14s} {'J':>8s} {'C %':>6s} {'N':>4s}")
    for g, s, l in sorted({(r["group"], r["setting"], r["label"]) for r in res}):
        rr = [r for r in res if (r["group"], r["setting"], r["label"]) == (g, s, l)]
        print(f"{g:10s} {s:22s} {l:14s} {np.mean([r['J'] for r in rr]):8.4f} {100 * np.mean([r['C'] for r in rr]):6.1f} "
              f"{np.mean([r['N'] for r in rr]):4.1f}  ({', '.join(str(r['N']) for r in rr)})")
    for body, hours in SETTINGS:
        s = f"{body} {hours} h"
        A = [r["J"] for r in res if r["group"] == "cases" and r["setting"] == s]
        B = [r["J"] for r in res if r["group"] == "vs_random" and r["setting"] == s]
        if len(A) > 1:
            print(f"{s}: SA vs random p = {mannwhitneyu(A, B, alternative='less').pvalue:.4f}")


if __name__ == "__main__":
    main()

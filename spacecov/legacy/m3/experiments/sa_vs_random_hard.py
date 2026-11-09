"""SA vs random search on a 1-day Eros mission (harder than the 10-day validation case).

With T = 10 days coverage saturates (almost any 2-spacecraft fleet images ~90 %), so random search
looks good. Here the mission is short and/or the search space large, where placement matters more.

    python -m spacecov.legacy.m3.experiments.sa_vs_random_hard.py --pilot          # 2 seeds, 600 evals: quick look
    python -m spacecov.legacy.m3.experiments.sa_vs_random_hard.py                  # 5 seeds, 3000 evals -> results/hard_vs_random.csv
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

from run_experiments import Job, run_all, write_runs  # noqa: E402
from spacecov.legacy.m3.algorithms import SAParams  # noqa: E402

# (label, N_max, mission days)
SETTINGS = [("T=1d, Nmax=5", 5, 1.0), ("T=1d, Nmax=10", 10, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", help="setting labels to run (default: all)")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (to add seeds to earlier runs)")
    ap.add_argument("--append", action="store_true", help="merge with the runs already in results/")
    a = ap.parse_args()
    if a.pilot:
        a.seeds, a.evals = 2, 600
    base = asdict(SAParams(max_evals=a.evals))
    settings = [s for s in SETTINGS if not a.only or s[0] in a.only]

    jobs = []
    for name, n, days in settings:
        for s in range(a.seed_start, a.seed_start + a.seeds):
            jobs.append(Job("hard", "SA", n, 0.0, s, base, days=days, extra={"setting": name}))
            jobs.append(Job("hard", "random search", n, 0.0, s, base, days=days, random_search=True,
                            extra={"setting": name}))
    res = run_all(jobs, a.workers, "hard")

    out = ROOT / "results"
    tag = "pilot_" if a.pilot else ""
    raw = out / f"raw_{tag}hard_vs_random.json"
    if a.append and raw.exists():
        res = json.loads(raw.read_text(encoding="utf-8")) + res
    raw.write_text(json.dumps(res), encoding="utf-8")
    write_runs(out / f"{tag}hard_vs_random.csv", [dict(r, label=f"{r['setting']} | {r['label']}") for r in res])

    print(f"\n{'setting':16s} {'method':14s} {'J':>8s} {'C %':>7s} {'N':>5s}")
    for name, _, _ in settings:
        for label in ("SA", "random search"):
            r = [x for x in res if x["setting"] == name and x["label"] == label]
            print(f"{name:16s} {label:14s} {np.mean([x['J'] for x in r]):8.3f} "
                  f"{100 * np.mean([x['C'] for x in r]):7.1f} {np.mean([x['N'] for x in r]):5.1f}")


if __name__ == "__main__":
    main()

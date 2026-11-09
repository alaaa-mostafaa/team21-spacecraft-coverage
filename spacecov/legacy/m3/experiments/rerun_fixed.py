"""Re-run only the fixed-fleet-size sweep of the mission study (2 days, N = 1..6) with a full budget,
and replace those rows in results/raw_mission.json (+ mission.csv, mission_summary.csv).

    python -m spacecov.legacy.m3.experiments.rerun_fixed.py --seeds 5 --evals 3000
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiments import Job, run_all, summarise, write_dicts, write_runs  # noqa: E402
from spacecov.legacy.m3.algorithms import SAParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    base = asdict(SAParams(max_evals=a.evals))
    jobs = [Job("mission", f"fixed N={n}, T=2d", n, 0.0, s, base, days=2.0, fixed_n=True)
            for n in range(1, 7) for s in range(a.seeds)]
    new = run_all(jobs, a.workers, "fixed")
    out = ROOT / "results"
    raw = json.loads((out / "raw_mission.json").read_text(encoding="utf-8"))
    res = [r for r in raw if not r["label"].startswith("fixed N=")] + new
    (out / "raw_mission.json").write_text(json.dumps(res), encoding="utf-8")
    write_runs(out / "mission.csv", res)
    write_dicts(out / "mission_summary.csv", summarise(res))
    for n in range(1, 7):
        rr = [r for r in new if r["N_max"] == n]
        print(f"N={n}: C mean {100 * sum(r['C'] for r in rr) / len(rr):.1f} %, best J {min(r['J'] for r in rr):.4f}, "
              f"mean J {sum(r['J'] for r in rr) / len(rr):.4f}")


if __name__ == "__main__":
    main()

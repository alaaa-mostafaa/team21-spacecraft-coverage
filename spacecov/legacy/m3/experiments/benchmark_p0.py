"""Effect of the initial acceptance p0 and the warm start on the hardest benchmark (Rastrigin, 5 dimensions).

10 seeds x 10,000 evaluations per setting, as in the benchmark validation.

    python -m spacecov.legacy.m3.experiments.benchmark_p0.py   -> results/benchmark_p0.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from spacecov.legacy.m3.algorithms import SAParams  # noqa: E402
from spacecov.legacy.m3.algorithms.benchmarks import solve_benchmark  # noqa: E402


def main():
    rows = []
    for p0, n_init in ((0.8, 1), (0.5, 1), (0.5, 20)):        # old default, p0 only, new default
        vals = np.array([solve_benchmark("rastrigin", 5, SAParams(max_evals=10000, seed=s, p0=p0, n_init=n_init)).J_best
                         for s in range(10)])
        rows.append({"function": "rastrigin", "dim": 5, "p0": p0, "n_init": n_init, "best": vals.min(), "mean": vals.mean(),
                     "success_rate (<1e-1)": float(np.mean(vals < 0.1))})
        print(rows[-1])
    with open(ROOT / "results" / "benchmark_p0.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

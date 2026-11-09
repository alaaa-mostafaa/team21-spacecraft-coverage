"""Command line: evaluate a scenario's fleet, or design one with simulated annealing.

    python -m spacecov scenarios/orbital_example.json
    python -m spacecov scenarios/orbital_example.json --days 3
    python -m spacecov scenarios/orbital_example.json --optimise --n 7 --evals 3000 --seed 1 --save best.json

When optimising, --n is N_max: the number of slots.  SA decides how many of them to use.
"""
import argparse
import copy

import numpy as np

from .algorithms import solve
from .design import N_VARS
from .engine import evaluate
from .problem import OrbitalProblem
from .scenario import Scenario


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario")
    p.add_argument("--days", type=float, help="mission duration T [days]")
    p.add_argument("--dt", type=float, help="time step = frame interval [min] (>= 8.4 min for the downlink)")
    p.add_argument("--sun-lat", type=float, help="sub-solar latitude [deg] (season)")
    p.add_argument("--no-occlusion", action="store_true", help="skip ray casting")
    p.add_argument("--no-lighting", action="store_true", help="treat every facet as sunlit (report M2)")
    p.add_argument("--optimise", action="store_true", help="run simulated annealing from this fleet")
    p.add_argument("--n", type=int, help="N_max slots when optimising (default: config n_max)")
    p.add_argument("--evals", type=int, help="SA evaluation budget")
    p.add_argument("--seed", type=int, help="SA random seed")
    p.add_argument("--save", help="write the scenario with the optimised fleet to this file")
    a = p.parse_args()

    sc = Scenario.load(a.scenario)
    cfg = sc.config
    if a.days:
        cfg.duration_days = a.days
    if a.dt:
        cfg.dt_min = a.dt
    if a.no_occlusion:
        cfg.occlusion = False
    if a.no_lighting:
        cfg.lighting = False
    if a.sun_lat is not None:
        cfg.sun_lat_deg = a.sun_lat
    ast = sc.build_asteroid()
    print(f"{sc.name}\n{ast.name}: {ast.n_facets} facets, R_max = {ast.r_max:.2f} km")
    print(f"r_min = {cfg.resolve_r_min(ast.r_max):.2f} km, T = {cfg.duration_days:g} d, "
          f"N_t = {cfg.n_steps}, dt = {cfg.dt_min:g} min, lighting {'on' if cfg.lighting else 'off'}")

    if not a.optimise:
        if sc.fleet is None:
            raise SystemExit("scenario has no spacecraft")
        res = evaluate(sc.fleet, ast, cfg)
        print(res.summary())
        print(f"min separation = {res.min_separation:.2f} km, lowest battery charge = {100 * res.soc.min():.0f} %, "
              f"Delta-V = {', '.join(f'{1000 * v:.2f}' for v in res.dv)} m/s, "
              f"evaluation took {res.eval_seconds:.2f} s")
        return

    params = copy.deepcopy(sc.sa)
    if a.evals:
        params.max_evals = a.evals
    if a.seed is not None:
        params.seed = a.seed
    n = a.n or cfg.n_max
    prob = OrbitalProblem(ast, n, cfg)
    x0 = None
    if sc.fleet and sc.fleet.n_active <= n:                     # start from the scenario's fleet,
        x0 = np.zeros((n, N_VARS))                # padded with unused slots
        x0[:, 0] = prob.r_min + 10.0
        rows = sc.fleet.active().to_vector().reshape(-1, N_VARS)
        x0[:len(rows)] = rows
        x0 = x0.ravel()

    def report(step):
        if step.improved or step.iteration % 100 == 0:
            print(f"  it {step.iteration:5d}  T {step.T:.2e}  J {step.J:+.4f}  best {step.J_best:+.4f}"
                  f"  C {step.info_best.get('C', float('nan')):.3f}  N {step.info_best.get('N', 0):.0f}")

    res, best = solve(prob, params, x0, report)
    print(f"done: {res.stop_reason}, {res.n_evals} evaluations, {res.seconds:.1f} s, "
          f"T0 = {res.T0:.3g}, alpha = {res.alpha:.4f}, L = {res.chain_length}")
    print("best:", best.summary())
    if a.save:
        sc.fleet = prob.decode(res.x_best).active()
        sc.config.n_max = n
        sc.save(a.save)
        print(f"saved {a.save}")


if __name__ == "__main__":
    main()

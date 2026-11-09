"""Command line: evaluate a scenario's design, or optimise one.

    python -m spacecov scenarios/orbital_example.json                      # evaluate the design
    python -m spacecov scenarios/orbital_example.json --optimise           # run the scenario's algorithm
    python -m spacecov scenarios/orbital_example.json --optimise --algorithm sa --evals 3000 \\
        --seed 1 --n-max 7 --save best.json

Every evaluation goes  Algorithm -> Feasibility -> Engine -> cost.
"""
import argparse
import copy

from . import algorithms
from .asteroid import PRESETS
from .config import for_body
from .pipeline import Pipeline
from .scenario import Scenario


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario")
    p.add_argument("--optimise", action="store_true", help="search for the best design")
    p.add_argument("--algorithm", choices=algorithms.available(), help="default: the scenario's algorithm")
    p.add_argument("--evals", type=int, help="evaluation budget")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--n-max", type=int, help="number of spacecraft slots N_max")
    p.add_argument("--body", choices=["eros", "kleopatra"], help="target body (sets its constants and mesh)")
    p.add_argument("--t-max", type=float, help="longest mission T_max [days] (simulation horizon)")
    p.add_argument("--from-design", action="store_true", help="start the search from the scenario's design")
    p.add_argument("--dt", type=float, help="time step = frame interval [min] (>= 8.4 min for the downlink)")
    p.add_argument("--sun-lat", type=float, help="sub-solar latitude [deg] (season)")
    p.add_argument("--no-occlusion", action="store_true", help="skip ray casting")
    p.add_argument("--no-lighting", action="store_true", help="treat every facet as sunlit")
    p.add_argument("--save", help="write the scenario with the best design to this file")
    a = p.parse_args()

    sc = Scenario.load(a.scenario)
    if a.body:
        sc.config = for_body(a.body, sc.config)
        sc.asteroid = dict(PRESETS[a.body])
    cfg = sc.config
    for flag, attr in ((a.n_max, "n_max"), (a.t_max, "t_max_days"), (a.dt, "dt_min"), (a.sun_lat, "sun_lat_deg")):
        if flag is not None:
            setattr(cfg, attr, flag)
    if a.no_occlusion:
        cfg.occlusion = False
    if a.no_lighting:
        cfg.lighting = False
    ast = sc.build_asteroid()
    pipe = Pipeline(ast, cfg)
    print(f"{sc.name}\n{ast.name}: {ast.n_facets} facets, R_max = {ast.r_max:.2f} km, A* = {pipe.engine.a_star:.0f} km2")
    print(f"N in 1..{cfg.n_max}, T in {cfg.t_min_days:g}..{cfg.t_max_days:g} d, a in {pipe.space.r_min:.1f}.."
          f"{cfg.r_max:g} km, dt = {cfg.dt_min:g} min, camera {cfg.scanners[cfg.camera].name}, "
          f"weights (coverage, cost, energy, time) = {tuple(round(w, 3) for w in cfg.weights)}")

    if not a.optimise:
        if sc.design is None:
            raise SystemExit("scenario has no design")
        ev = pipe.evaluate(pipe.space.encode(sc.design))
        print(ev.summary())
        if ev.result is not None:
            r = ev.result
            print(f"min separation = {ev.report.min_separation:.2f} km, lowest battery = {100 * ev.report.min_soc:.0f} %, "
                  f"frames = {', '.join(map(str, r.n_img))}, evaluation {r.eval_seconds:.2f} s")
        return

    name = a.algorithm or sc.algorithm
    params = dict(sc.algorithm_params)
    if a.evals:
        params["max_evals"] = a.evals
    if a.seed is not None:
        params["seed"] = a.seed
    algo = algorithms.make(name, **params)
    x0 = None
    if a.from_design and sc.design is not None and sc.design.N <= cfg.n_max:
        x0 = pipe.space.encode(sc.design)

    def report(step):
        if step.improved or step.iteration % 100 == 0:
            b = step.best
            c = f"C {100 * b.result.C:5.1f} %" if b.result is not None else "rejected"
            print(f"  eval {step.iteration:5d}  T {step.extra.get('T', float('nan')):.2e}  "
                  f"cost {step.cost:8.4f}  best {b.cost:.4f}  ({c}, {pipe.describe(step.best_x).split(' [')[0]})")

    print(f"running {name} with {algo.params}")
    res = algo.run(pipe, x0, report)
    print(res.summary())
    print(f"engine runs {pipe.engine.n_simulations}, rejected by feasibility {pipe.n_rejected}, "
          f"spacecraft simulated {pipe.engine.n_tracks}")
    print("best:", pipe.describe(res.x_best))
    print(res.best.summary())
    if a.save:
        out = copy.deepcopy(sc)
        out.design = pipe.space.decode(res.x_best)
        out.algorithm, out.algorithm_params = name, params
        out.name = f"{name} best: {out.design.summary()}, J = {res.cost_best:.4f}"
        out.save(a.save)
        print(f"saved {a.save}")


if __name__ == "__main__":
    main()

# Cooperative Spacecraft Configurations Optimization

Team 21, MCTR 1021 (GUC): design a fleet of spacecraft that images an irregular asteroid
(**433 Eros**, or the 6.5× larger **216 Kleopatra**). The optimiser chooses **how long the mission
lasts (T), which spacecraft slots fly (so how many spacecraft, N) and the circular orbit of each**.
It minimises one fitness J that **maximises coverage** while **minimising fleet cost, energy and
mission time**.

```
             x = [T, (a, ι, Ω, M₀, t) × N_max]
  Algorithm ──────────────────────────────▶ Feasibility ──(feasible design)──▶ Engine
  (SA, GA, random)  ◀────── fitness J ─────── (rejected: penalty cost) ◀───── J(x)
```

This merges our modular structure with the final Milestone 3 formulation (coverage / cost / energy
fitness, on/off slots, Kleopatra), with the mission time made a variable that is minimised.

* **`spacecov/algorithms/`**: the optimisation algorithms. Each one only proposes decision vectors and
  reads back a cost (`base.Algorithm`), so it can be validated on functions with a known optimum
  (`benchmarks.py`). Implemented: **Simulated Annealing** (`sa.py`), the **Genetic Algorithm**
  (`ga.py`) and **random search** (`random_search.py`, the baseline).
* **`spacecov/design.py`**: the decision variables and their bounds.
* **`spacecov/feasibility.py`**: the constraints. A design that breaks one is rejected **before** the
  engine runs.
* **`spacecov/engine.py`**: simulates the mission (`physics.py`) and computes the fitness J.
* **`spacecov/pipeline.py`**: chains Feasibility and Engine into the single `evaluate(x)` the
  algorithms call.
* **`viz/`**: an interactive true-scale 3D viewer (PyVista + Qt) with live charts of the search.
* **`spacecov/legacy/m3/`**: the final Milestone 3 code (frozen), which produced the original `results/`
  and `report/`; its SA-only experiment scripts are retained for historical reproducibility.
* **`tests/`**: physics checks, the pipeline, the algorithms on benchmarks, the viewer logic, and the
  Milestone 3 tests (`tests/legacy/`).

The current Milestone 4 code is everything under `spacecov/` and `viz/` outside `spacecov/legacy/m3/`.
Generated validation tables and figures live under `results/validation/`; they should be regenerated
with `experiments/validate.py` rather than edited by hand.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip install -r requirements.txt
# For the validated report/benchmark environment:
.venv/bin/pip install -r requirements-lock.txt
```

`embreex` (installed automatically on 64-bit Windows/Linux) makes the ray casting 10–50× faster.
The lock file records the versions used by the validated macOS environment; use it when reproducing
benchmark numbers, and record the git commit shown in `results/validation/run_info.json`.

## Run

Windows (`sim.bat`) and Linux/macOS (`./sim`) take the same commands:

```
sim                                       viewer on the example design
sim my_scenario.json                      viewer on any scenario
sim eval scenarios/orbital_example.json   cost and constraints only, no window
sim optimise scenarios/orbital_example.json --algorithm ga --evals 3000 --seed 1 --save best.json
sim test                                  unit tests
sim experiments                           Milestone 3 experiments (legacy)
```

For the desktop integration smoke test, use a real OpenGL-capable desktop session:
`python tools/gui_smoke.py .work/gui-smoke`. PyVista/VTK can segfault in Qt's headless
`offscreen` platform, so the regular unit suite remains the headless-safe validation path.

`optimise` also takes `--algorithm sa|ga|random`, `--body eros|kleopatra`, `--n-max 7`, `--t-max 10` (days), `--from-design` (start from
the scenario's design), `--dt`, `--sun-lat`, `--no-occlusion` and `--no-lighting`. The best design
written by `--save` opens in the viewer.

From Python:

```python
from spacecov import Pipeline, Scenario
from spacecov.algorithms import make

sc = Scenario.load("scenarios/orbital_example.json")
pipe = Pipeline(sc.build_asteroid(), sc.config)        # Feasibility + Engine
result = make("sa", max_evals=1500, seed=0).run(pipe)  # Algorithm
print(pipe.describe(result.x_best), result.best.summary())
```

### The viewer

* **3D view (top).** Everything is at true scale in km: the full 49,152-plate Gaskell model of Eros,
  the orbits and the spacecraft models (5.6 m across). A 5.6 m spacecraft is far below one pixel when
  the 34 km asteroid is in view, so each spacecraft also has a coloured screen marker and label (not
  to scale); the km scale bar reads distances at the view centre. Press **F** to chase a spacecraft
  from 30 m and see its real size, or **1–7** to look through its camera. The scene light *is* the
  Sun, from the same direction the sunlight test uses, and terrain cast shadows are darkened.
* **Scanner cones** end where their edge rays hit Eros, or at the camera's usable range. On the
  surface, **bright** = imaged right now, **faint** = in the field of view but not counted (too far,
  too oblique, unlit or hidden). The HUD shows each spacecraft's speed (vis-viva), imaged area,
  battery charge and eclipses.
* **Orbits** are the exact circles; positions come from Kepler's equation at the displayed time and
  Eros turns by ω·t on the same clock, so the spin and the orbital speeds are in their real ratio.
  **B** switches to the asteroid-fixed frame, where each path winds around the body.
* **Coverage vs mission time (bottom left):** C(t) of the fleet and of each spacecraft alone, as % of
  the area that is ever sunlit. Click the chart to jump to that time.
* **Search charts:** fitness J (candidates, current, best), coverage against mission time T of every
  design tried, and an algorithm-specific progress chart. SA shows temperature and accepted/rejected
  neighbors; GA shows generation/population spread and crossover or mutation children. While the
  search runs, the 3D view shows the current candidate or best design.

| Input | Action |
|---|---|
| Mouse left / right / middle drag, wheel | rotate / zoom / pan |
| **W A S D**, **Q E**, Shift | fly, ×4 speed |
| **1 … 7**, **F**, **0**/Esc, **R** | ride along, chase, free camera, reset |
| Space, `,` `.`, `[` `]` | play/pause, step, slower/faster (×1 real time up to ×28800) |
| **M**, **B**, **C O L T**, **H**, **P** | colouring, frame, cones/orbits/labels/trails, help, screenshot |

Side tabs: **Mission** (target body, N_max, T_min / T_max, camera, fitness weights, frame interval, sunlight/season/shadows),
**Optimiser** (algorithm and its settings, start from random designs or the current one, Run/Stop/Pause,
watch pace, move log), **Design** (mission time T and the orbit of each spacecraft), **View**.
The metrics panel shows the cost terms and every feasibility check of the displayed design; an
infeasible design is still simulated and drawn, marked INFEASIBLE. *File → Export search history*
writes the run to CSV.

## The optimisation problem

**Decision variables** (`design.py`). One fixed-length vector of 1 + 5·N_max numbers (26 for N_max = 5):

| Variable | Meaning | Kind | Range (Eros) |
|---|---|---|---|
| T | mission duration | real | T_min … T_max = 0.25 … 10 days |
| a_i | orbit radius (circular orbits) | real | r_min … r_max = 26.5 … 100 km |
| ι_i | inclination: tilt of the orbit to the asteroid's equator | real | 0 … 180° |
| Ω_i | longitude of the ascending node: where the orbit crosses the equator | real, wraps | 0 … 360° |
| M₀,i | mean anomaly at t = 0: where the spacecraft starts | real, wraps | 0 … 360° |
| t_i | on/off switch: 1 = slot i flies a spacecraft | integer | 0 or 1 |

The number of spacecraft is N = Σ t_i (1 … N_max, default N_max = 5). A switched-off slot keeps its
orbit numbers unused, so every algorithm sees one box of a fixed size. Orbits are circular (e = 0, so
ω has no meaning), and every spacecraft carries the same 6° camera. Kleopatra uses the same variables
with distances scaled to the body (orbits 171 … 650 km).

**Fitness** (`engine.py`), minimised:

  J = w_C · (1 − C) + w_Z · Z + w_E · E + w_T · T / T_max,   weights 0.5 / 0.15 / 0.15 / 0.2

| Term | Meaning | Goal |
|---|---|---|
| C | coverage: area imaged within T ÷ A* (area sunlit at some time within T_max) | **maximise** (enters as 1 − C, the share missed) |
| Z | fleet cost: N × (bus 1.0 + camera 0.2) ÷ cost of 10 spacecraft = N / 10 | minimise |
| E | energy: ½ ΣΔV ÷ (10 · 3v₀) + ½ ΣE_elec ÷ (10 · E_ref); E_elec = 50 W · T + ~15 kJ per frame | minimise |
| T / T_max | mission duration | minimise |

* The weights add to 1. Coverage enters as 1 − C, not 1/C: 1/C is infinite for an empty map and makes
  the last percentages of coverage almost worthless. Fleet cost and energy use the fixed reference
  fleet `n_ref = 10`, so they can exceed 1 when more than 10 spacecraft are selected; this deliberately
  keeps larger fleets penalized instead of saturating their cost.
* E already grows with T (the bus draws 50 W for the whole mission), so time is partly charged there too.
* A facet counts as imaged at a step if (a) it is inside the camera cone, (b) the emission angle is
  ≤ 65°, (c) no terrain blocks the line of sight (ray casting), (d) the pixel on the ground is
  ≤ 3 m (20 m on Kleopatra), and (e) it is sunlit (Sun ≥ 20° above its horizon, no cast shadow).
* Each spacecraft is simulated once over T_max and cached by its four orbit variables. A design of
  duration T uses the steps t ≤ T, so changing T or switching a slot costs no new simulation, and
  moving one orbit costs one.

**Constraints** (`feasibility.py`). Each is normalised so that a value ≤ 0 is satisfied:

| | Constraint | Checked |
|---|---|---|
| N | at least one switch on; every switch 0 or 1 | before the engine |
| T | T_min ≤ T ≤ T_max | before the engine |
| g1 | a_i ≥ r_min = 1.5 R_max = 26.5 km (periapsis safety; circular orbit) | before the engine |
| g2 | a_i ≤ r_max = 100 km (apoapsis limit) | before the engine |
| inc | 0 ≤ ι_i ≤ 180° | before the engine |
| g3 | every pair of flying spacecraft ≥ 2 km apart at every step t ≤ T (10 km on Kleopatra) | before the engine |
| g4 | battery charge ≥ 20 % at every step t ≤ T | after the engine (needs the imaging) |

A design that fails feasibility costs `1000 + 1000 · Σ violation²`: worse than any feasible design,
but infeasible designs can still be ranked (useful for GA / PSO). `repair(x)` maps a vector back into
the box (a / ι / T reflected at their bounds, angles wrapped, switches rounded with at least one on).

**Frame interval = time step.** One frame is taken per step, and each 6.3 MB frame needs 503 s to
downlink, so Δt ≥ 8.4 min; 10 min is used.

## Cameras

Modelled on real small-body mapping cameras (2048 × 2048 px), with GSD_max = 3 m (similar to NEAR's
mapping resolution at Eros). Usable range ℓ_max = GSD_max · p / (2 tan(ψ/2)).

| Type | FOV | Real analogue | Usable range | Footprint at 25 km |
|---|---|---|---|---|
| 1 narrow (default) | 6° | Hayabusa2 ONC-T (6.35°), Dawn Framing Camera (5.5°) | 58.6 km | 2.6 km |
| 2 wide | 12° | Rosetta OSIRIS wide-angle camera (12°) | 29.2 km | 5.3 km |

All spacecraft carry the same camera (`config.camera`). The narrow camera works from most of the
orbit range; the wide one images 4× the area per frame but only within ~29 km, i.e. from low orbits.

## Simulated Annealing (spacecov/algorithms/sa.py)

```
x ← best of n_init = 20 random designs that pass feasibility         (warm start)
T0 ← −median(uphill ΔJ of 20 trial moves around x) / ln p0           (p0 = 0.5)
for each temperature level T = T0, αT0, α²T0, …   (α fitted so the budget ends at 10⁻³·T0, or given)
    repeat L times (default L = max(10, dim)):
        y ← neighbour(x): change 1–2 of the coordinates that matter now (T, a switch, or a / ι / Ω /
            M₀ of a flying spacecraft) by N(0, (step·range)²), step = step0·√(T/T0); a switch flips
            on/off; repair wraps the angles and reflects the rest at their bounds
        ΔJ = J(y) − J(x)                          Algorithm → Feasibility → Engine
        accept y if ΔJ ≤ 0, else with probability exp(−ΔJ/T)        (Metropolis)
        keep the best design (a feasible design always beats an infeasible one)
until the evaluation budget is used
```

* **Why the warm start:** the narrow camera resolves 3 m only within 58.6 km, so high orbits and short
  missions image almost nothing and the fitness is flat and bad there. T0 measured from such a start
  was far too hot and the budget turned into a random walk.
* **Why p0 = 0.5:** it beat 0.8 on both 1-day and 10-day missions, in agreement with the Milestone 3
  parameter study.
* **Validation** (`tests/test_algorithms.py`): SA reaches the known minimum 0 of the sphere, Ackley
  and 2-D Rastrigin functions. On the fleet problem (Eros, N_max = 5, T_max = 10 days, 800
  evaluations, 3 seeds) it reaches a mean J of 0.109 against 0.170 for random search with the same
  budget, winning on every seed. It chooses 3 spacecraft and T ≈ 1.2–1.6 days, imaging 95–96 % of the
  sunlit area. The Milestone 3 report (fixed 10-day mission) has the full SA study.

## Genetic Algorithm (spacecov/algorithms/ga.py)

```
population ← P designs stratified over the fleet size N (P = 40 for N_max = 5, 60 for N_max = 10)
repeat until the evaluation budget is used:
    n_e = 5 % of P best distinct designs survive unchanged                        (elitism)
    80 % of the rest: 2 tournament parents → align their slots → swap or blend    (crossover)
    20 % of the rest: 1 tournament parent → change 1–2 active genes (as SA's move) (mutation)
    repair, re-mutate duplicates, evaluate the children (Algorithm → Feasibility → Engine)
```

* **Chromosome** = the decision vector: a gene is one number, a slot (5 genes) is one spacecraft.
* **Slot alignment:** a fleet can be stored in N_max! slot orders, so B's slots are matched to A's
  (nearest orbit, angles on the circle) before crossover.
* **Swap or blend** each aligned pair (and T): a pair in which both slots fly (and T) is blended with
  BLX-α with probability p_b = 0.3 (≈ one new orbit per child; angles along the shorter arc);
  otherwise each child takes the whole slot from one parent, chosen with probability ½ (slot swap).
  A pair with only one flying slot is never blended.
* **Generic:** without `slot_groups()` (benchmark functions) every gene is its own group, i.e.
  uniform crossover + BLX-α. Validated in `tests/test_ga.py` (known optima, every operator alone).
* `sim optimise scenarios/orbital_example.json --algorithm ga --evals 2000`, or pick `ga` in the
  viewer's Optimiser tab. Report chapter: `report/LaTeX/chapters/ga.tex`.
* Experiments: `python experiments/validate.py --only G2 GP GV4 GV5` (GA benchmarks, parameter study,
  the V4 settings, all-lit Eros); with V4 present, `SA_vs_GA.csv` compares both algorithms.
  `--only H B` adds the hard cases and the 5000-evaluation budget sweep; `--only M NI` the matched
  GA–SA comparison (Eros, N_max = 10, T_max = 10 d, 3000-facet mesh, seeds 0–2, 1200 evaluations) and
  the equal-native-iterations run. `python experiments/ga_assets.py` rebuilds the GA tables and figures.

Adding an algorithm: subclass `algorithms.base.Algorithm` (a `Params` dataclass and `run(problem, x0,
callback)`), use `Recorder` for the budget, best design and history, and register it in
`algorithms/__init__.py`. The viewer's Optimiser tab builds its settings form from `Params`.

## Milestone 3 (legacy)

The final Milestone 3 code (7 numbers per slot including eccentricity and argument of periapsis, a fixed
10-day mission, J = 0.6 (1 − C) + 0.2 Z + 0.2 E + penalty) is frozen in `spacecov/legacy/m3/`. It
shares physics, geometry and the meshes with the current code. Its experiments produced `results/`
and the submitted report `report/Team21_Milestone3_Final.pdf`:
`python -m spacecov.legacy.m3.experiments.run_experiments [--quick]` (also `rerun_fixed`,
`sa_vs_random_hard`, `run_short`, `run_kleopatra`, `benchmark_p0`). The `results/best_*.json` files use
the M3 scenario format (`spacecov.legacy.m3.scenario`). The figure scripts in `report/` need the
Milestone 3 viewer: branch `milestone3-final`.

## Modelling notes and assumptions

* **Eros:** Gaskell shape model (NASA PDS, `ver64q.tab`, 49,152 plates). The viewer always draws the
  full model; the objectives are computed on a copy decimated to 1500/3000/6000 facets (cached in
  `data/eros/cache/`) and mapped back onto it. GM = 4.463×10⁵ m³/s², rotation period 5.27 h,
  R_max = 17.63 km, mean radius 8.4 km.
* **Kleopatra:** radar shape model (Ostro et al. 2000, NASA PDS `216kleopatra.tab`, 4,092 facets),
  centred on its centre of mass; GM = 1.98×10⁸ m³/s² (Brož et al. 2021), rotation period 5.385 h.
  Distances scale with the body (`config.BODIES`, `for_body`): orbits 171–650 km, 20 m pixels,
  10 km separation keep the same ratios to its size as for Eros.
* **Sunlight:** a facet is imaged only if the Sun is at least 20° above its local horizon (incidence
  ≤ 70°) and not hidden by terrain. The Sun direction is fixed in inertial space over the mission
  (Eros moves ~0.6°/day around the Sun); the sub-solar latitude (season) is a setting. Switch it off
  with `"lighting": false` (or `--no-lighting`) to recover the Milestone 2 assumption.
* **Two-body dynamics:** orbits are fixed circular Kepler orbits around a point mass. The bodies' irregular
  gravity field (which makes real orbits precess) and solar radiation pressure are not modelled; this
  is the main accuracy limitation. Uniform rotation about z.
* **Power values** are illustrative small-spacecraft numbers; spacecraft are nadir-pointing
  points with a circular field of view; a facet counts as imaged if its centre passes every test.
* **Spacecraft size** (for drawing only): 1.6 m bus, 5.6 m across the solar panels.

## Layout

```
spacecov/design.py         decision vector x = [T, (a, inc, RAAN, M0, on) x N_max], bounds, encode/decode
spacecov/feasibility.py    constraints N, T, g1-g4, repair
spacecov/engine.py         mission simulation per spacecraft (cached), objectives C Z E and the fitness J
spacecov/pipeline.py       Feasibility + Engine = the Problem the algorithms see
spacecov/algorithms/       base (Algorithm, Recorder, local_move), sa, ga, random_search, benchmarks
spacecov/physics.py        Kepler, frames, visibility (a)-(e), shadows, eclipses, battery, Delta-V
spacecov/config.py         MissionConfig, BODIES (Eros, Kleopatra), cameras, ranges, constraints, weights
spacecov/asteroid.py       meshes: Gaskell Eros and Kleopatra (PDS) loaders + decimation, ellipsoid fallback
spacecov/scenario.py       JSON scenarios (asteroid, config, design, algorithm settings)
spacecov/legacy/m3/        final Milestone 3 code and experiments (frozen)
viz/                       scene (3D actors), presenter (state -> scene), plots, panels, app (window)
results/, report/          Milestone 3 results and the submitted report
data/                      shape models (eros/, kleopatra/)
```

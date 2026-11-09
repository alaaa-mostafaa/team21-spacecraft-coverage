"""spacecov - cooperative spacecraft asteroid-coverage optimisation (Team 21, MCTR 1021).

    Algorithm  --x-->  Feasibility  --design-->  Engine  --cost F-->  Algorithm

    design.py        decision vector x = [N, T, (a, inc, RAAN, M0) x N_max]
    feasibility.py   constraints; rejects a design before the engine runs
    engine.py        mission simulation (physics.py) and the cost function F
    pipeline.py      Feasibility + Engine as one Problem for the algorithms
    algorithms/      optimisation algorithms (simulated annealing, ...) and benchmarks
    legacy/          the frozen Milestone 3 formulation and experiments

GUI-free: the ``viz`` package renders results.
"""
from .asteroid import Asteroid, make_asteroid
from .config import MissionConfig
from .design import CircularOrbit, Design, DesignSpace
from .engine import Engine, EngineResult
from .feasibility import Feasibility, FeasibilityReport
from .pipeline import Evaluation, Pipeline
from .scenario import Scenario

__all__ = ["Asteroid", "make_asteroid", "MissionConfig", "CircularOrbit", "Design", "DesignSpace", "Engine",
           "EngineResult", "Feasibility", "FeasibilityReport", "Evaluation", "Pipeline", "Scenario"]

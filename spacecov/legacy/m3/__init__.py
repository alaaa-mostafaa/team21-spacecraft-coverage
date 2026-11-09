"""spacecov - cooperative spacecraft asteroid-coverage simulation core (Team 21, MCTR 1021).

GUI-free: optimisers import this package; the ``viz`` package renders its results.
"""
from .asteroid import Asteroid, make_asteroid
from .config import OrbitalConfig
from .algorithms import ALGORITHMS, SAParams, get_algorithm
from .design import DesignSpace, OrbitalSolution, OrbitalSpacecraft
from .problem import OrbitalProblem
from .scenario import Scenario

__all__ = ["ALGORITHMS", "Asteroid", "DesignSpace", "make_asteroid", "OrbitalConfig", "OrbitalProblem",
           "OrbitalSolution", "OrbitalSpacecraft", "SAParams", "Scenario", "get_algorithm"]

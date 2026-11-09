"""Interactive true-scale 3D viewer with live simulated annealing (PyVista + Qt).

    python -m viz scenarios/orbital_example.json

From code (e.g. after an optimiser run)::

    from viz import show
    show(scenario)                       # a spacecov.Scenario
"""


def show(scenario=None, **kwargs):
    from .app import run
    return run(scenario, **kwargs)

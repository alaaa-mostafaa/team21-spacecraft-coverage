"""Milestone 3, final version (frozen) so its experiments and results stay reproducible.

    spacecov.legacy.m3            the submitted M3 formulation: N_max slots of
                                  [a, e, inc, RAAN, argp, M0, on/off], fixed 10-day mission,
                                  J = 0.6 (1 - C) + 0.2 Z + 0.2 E + penalty (Eros and Kleopatra)
    spacecov.legacy.m3.experiments   the scripts that produced results/ and report/
        python -m spacecov.legacy.m3.experiments.run_experiments --quick
    python -m spacecov.legacy.m3 spacecov/legacy/m3/example.json      evaluate / optimise (M3 CLI)

It shares physics, geometry and the asteroid meshes with the current code.  New work uses
spacecov.design / feasibility / engine / pipeline / algorithms.
"""

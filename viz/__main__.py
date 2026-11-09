"""python -m viz [scenario.json]"""
import argparse
import sys

from .app import run


def main():
    p = argparse.ArgumentParser(description="3D viewer and live simulated annealing for asteroid coverage")
    p.add_argument("scenario", nargs="?", help="scenario JSON (default: scenarios/orbital_example.json)")
    a = p.parse_args()
    sys.exit(run(a.scenario))


if __name__ == "__main__":
    main()

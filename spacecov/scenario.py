"""Scenario = asteroid + mission config + a design (N, T, orbits) + algorithm settings, as JSON.

    {
      "name": "...",
      "asteroid": {"source": "eros", "faces": 1500},
      "config":   {"n_max": 5, "t_max_days": 10, ...},            (spacecov.config.MissionConfig)
      "design":   {"N": 2, "T_days": 3.0,
                   "spacecraft": [{"a": 45, "inc": 60, "raan": 0, "M0": 0}, ...]},
      "algorithm": {"name": "sa", "params": {"max_evals": 1500, "seed": 0}}
    }

Milestone 3 files (7 variables per spacecraft, results/best_*.json) use another format:
open them with spacecov.legacy.m3_scenario.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .asteroid import PRESETS, Asteroid, make_asteroid
from .config import MissionConfig
from .design import CircularOrbit, Design


@dataclass
class Scenario:
    name: str = "untitled"
    asteroid: dict[str, Any] = field(default_factory=lambda: dict(PRESETS["eros"]))
    config: MissionConfig = field(default_factory=MissionConfig)
    design: Design | None = None
    algorithm: str = "sa"
    algorithm_params: dict[str, Any] = field(default_factory=dict)

    def build_asteroid(self) -> Asteroid:
        return make_asteroid(self.asteroid)

    # -- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "asteroid": self.asteroid, "config": self.config.to_dict()}
        if self.design is not None:
            d["design"] = self.design.to_dict()
        d["algorithm"] = {"name": self.algorithm, "params": dict(self.algorithm_params)}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        if "design" not in d and ("spacecraft" in d or "orbital_spacecraft" in d or "sa" in d):
            raise ValueError("this is a Milestone 3 scenario (7 variables per spacecraft); load it with "
                             "spacecov.legacy.m3_scenario.Scenario")
        algo = d.get("algorithm", {})
        return cls(name=d.get("name", "untitled"), asteroid=d.get("asteroid", dict(PRESETS["eros"])),
                   config=MissionConfig.from_dict(d.get("config")),
                   design=Design.from_dict(d["design"]) if d.get("design") else None,
                   algorithm=algo.get("name", "sa"), algorithm_params=dict(algo.get("params", {})))

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def default_design(n: int, T_days: float = 3.0, a: float = 45.0) -> Design:
    """A reasonable hand-made design: N spacecraft spread in inclination, node and phase."""
    # radii 10 % apart and nodes spread so no two orbits share a plane (ι and 180° - ι with nodes
    # 180° apart would be the same plane flown backwards, and the spacecraft could collide)
    return Design(T_days, [CircularOrbit(a=a * (1.0 + 0.1 * i), inc=30.0 + 120.0 * i / max(n - 1, 1),
                                         raan=(360.0 * i / n + 90.0 * i) % 360.0, M0=(137.5 * i) % 360.0)
                           for i in range(n)])

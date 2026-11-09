"""Scenario = asteroid spec + mission settings + fleet (+ optimiser settings), stored as JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .asteroid import PRESETS, Asteroid, make_asteroid
from .config import OrbitalConfig
from .algorithms.sa import SAParams
from .design import OrbitalSolution, OrbitalSpacecraft


@dataclass
class Scenario:
    name: str = "untitled"
    asteroid: dict[str, Any] = field(default_factory=lambda: dict(PRESETS["eros"]))
    config: OrbitalConfig = field(default_factory=OrbitalConfig)
    fleet: OrbitalSolution | None = None
    sa: SAParams = field(default_factory=SAParams)

    def build_asteroid(self) -> Asteroid:
        return make_asteroid(self.asteroid)

    # -- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "asteroid": self.asteroid, "config": self.config.to_dict(),
                             "sa": self.sa.to_dict()}
        if self.fleet is not None:
            d["spacecraft"] = [s.to_dict() for s in self.fleet.spacecraft]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        cfg = d.get("config", d.get("orbital_config"))              # older files used "orbital_*"
        fleet = d.get("spacecraft", d.get("orbital_spacecraft"))
        sc = cls(name=d.get("name", "untitled"), asteroid=d.get("asteroid", dict(PRESETS["eros"])),
                 config=OrbitalConfig.from_dict(cfg), sa=SAParams(**d.get("sa", {})))
        if fleet:
            sc.fleet = OrbitalSolution([OrbitalSpacecraft(**s) for s in fleet])
        return sc

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def default_fleet(n: int) -> OrbitalSolution:
    """A reasonable hand-made starting fleet (used by the viewer's N buttons): 6-degree cameras on
    45-50 km orbits with spread tilts and phases."""
    out = []
    for i in range(n):
        out.append(OrbitalSpacecraft(a=45.0 + 5.0 * (i % 2), e=0.05, inc=30.0 + 120.0 * i / max(n - 1, 1),
                                     raan=360.0 * i / n, argp=0.0, M0=360.0 * i / n, type=1))
    return OrbitalSolution(out)

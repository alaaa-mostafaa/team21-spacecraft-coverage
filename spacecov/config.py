"""Mission and problem constants: a fleet of N spacecraft imaging an asteroid (433 Eros, 216 Kleopatra).

Units used everywhere inside the core: km, s, radians.  Angles are given in
degrees only at the interface (decision vectors, JSON scenarios, the viewer).

Real-world values used (433 Eros, NEAR Shoemaker results):
  * GM  = 4.463e-4 km^3/s^2           -> circular speed ~3.6 m/s at 35 km
  * rotation period 5.27 h            -> the tip of the 34 km body moves at ~5.8 m/s
  * shape: Gaskell shape model (data/eros/raw), R_max ~17.6 km, mean radius ~8.4 km

The config has three groups:
  physics      read by spacecov.physics (orbits, spin, Sun, cameras, power)
  constraints  read by spacecov.feasibility
  cost         read by spacecov.engine (the fitness J)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scanner:
    fov_deg: float      # psi_t, full cone angle
    pixels: int         # p_t, pixels across the field of view (2048 x 2048 sensor, 12 bit)
    cost: float = 0.0   # camera cost; a spacecraft costs BUS_COST + this
    name: str = ""


# Modelled on real small-body mapping cameras.  With GSD_max = 3 m the usable range is
# l_max = GSD_max * p / (2 tan(psi/2)):  ~59 km and ~29 km.
#   1 narrow: Hayabusa2 ONC-T (6.35 deg), Dawn Framing Camera (5.5 deg) - reaches from most orbits
#   2 wide:   Rosetta OSIRIS wide-angle camera (12 deg)                 - 4x the area, low orbits only
SCANNERS: dict[int, Scanner] = {
    1: Scanner(6.0, 2048, 0.2, "narrow"),
    2: Scanner(12.0, 2048, 0.3, "wide"),
}
BUS_COST = 1.0

# Target bodies: real physical data + mission settings scaled with the body's size.  The orbit band
# and the camera range keep the SAME proportion to the body as for Eros (r_min = 1.5 R_max always,
# r_max ~ 3.8 r_min), so only the body changes.  For Kleopatra (6.5x larger) that means mapping at
# 20 m per pixel instead of 3 m (Dawn mapped the similar-sized Vesta at ~20 m/px).
#   Kleopatra: mass 2.97e18 kg (Broz et al. 2021) -> GM = 0.198 km^3/s^2, rotation period 5.385 h,
#              mean radius 55.3 km of the radar model (eclipses).
BODIES: dict[str, dict] = {
    "eros": dict(mu=4.463e-4, p_rot_h=5.27, gsd_max_m=3.0, r_max=100.0, r0=50.0, d_sep=2.0,
                 eclipse_radius_km=8.4),
    "kleopatra": dict(mu=0.1982, p_rot_h=5.385, gsd_max_m=20.0, r_max=650.0, r0=320.0, d_sep=10.0,
                      eclipse_radius_km=55.3),
}

# Representative spacecraft size for drawing at true scale (a NEAR-class bus is ~1.7 m;
# a small mapping spacecraft with two deployed solar panels spans ~5-6 m).
SPACECRAFT_BUS_M = 1.6
SPACECRAFT_SPAN_M = 5.6


@dataclass
class MissionConfig:
    # -- physics: asteroid, time grid ---------------------------------------------------------
    mu: float = 4.463e-4              # km^3/s^2  (= 4.463e5 m^3/s^2, Eros)
    p_rot_h: float = 5.27             # asteroid rotation period [h]
    # time step = camera frame interval: one frame per step, and each 6.3 MB frame must be
    # downlinked (503 s at 100 kbps) before the next, so dt_min >= 8.4 min is required
    dt_min: float = 10.0
    # -- physics: imaging -----------------------------------------------------------------------
    camera: int = 1                   # every spacecraft carries this camera (1 narrow, 2 wide)
    gsd_max_m: float = 3.0            # maximum ground sample distance [m]                      (d)
    eta_max_deg: float = 65.0         # maximum emission angle                                   (b)
    occlusion: bool = True            # (c) line of sight blocked by terrain (ray casting)
    # (e) sunlight: a facet is lit at step k if the Sun is >= 90 - incidence_max above its
    # horizon AND no terrain blocks the Sun (cast shadows).  The Sun direction is fixed in
    # inertial space over the mission; sun_lat_deg is the sub-solar latitude (0 = equinox).
    lighting: bool = True
    shadows: bool = True
    sun_lat_deg: float = 0.0
    incidence_max_deg: float = 70.0
    # -- physics: power (battery constraint) and Delta-V (information only) --------------------
    p_base_w: float = 50.0            # bus housekeeping power, always on
    p_cam_w: float = 10.0             # camera power while exposing
    e_img_j: float = 100.0            # energy per frame (10 W x 10 s)
    p_tx_w: float = 30.0              # transmitter power
    downlink_bps: float = 100e3       # downlink rate
    p_solar_w: float = 180.0          # solar array output in sunlight (0 in eclipse)
    battery_wh: float = 200.0         # battery capacity
    eclipse_radius_km: float = 8.4    # Eros as a sphere of its mean radius for eclipses
    r0: float = 50.0                  # km, circular equatorial arrival orbit (Delta-V, energy E)
    body: str = "eros"                # which entry of BODIES the values above were taken from

    # -- decision-variable ranges ---------------------------------------------------------------
    n_max: int = 5                    # spacecraft slots; N = number of slots switched on, 1 .. n_max
    fixed_n: bool = False             # every slot flies (N = n_max): used to test a forced fleet size
    t_min_days: float = 0.25          # mission duration T in [t_min, t_max]
    t_max_days: float = 10.0          #   (also the simulation horizon)
    # a in [r_min, r_max], inc in [0, 180], RAAN and M0 in [0, 360), switch t in {0, 1}

    # -- constraints (spacecov.feasibility) -----------------------------------------------------
    kappa: float = 1.5                # r_min = kappa * R_max (closest safe orbit radius)       g1
    r_min: float | None = None        # km; None -> kappa * asteroid.r_max
    r_max: float = 100.0              # km, farthest orbit radius (apoapsis limit)               g2
    d_sep: float = 2.0                # km, minimum distance between spacecraft at every step    g3
    soc_min: float = 0.2              # minimum battery state of charge at every step            g4

    # -- fitness (spacecov.engine) ------------------------------------------------------------
    #   J = w_C (1 - C) + w_Z Z + w_E E + w_T T / T_max      (minimise; 0 = perfect, 1 = worst)
    weights: tuple[float, float, float, float] = (0.5, 0.15, 0.15, 0.2)   # coverage, cost, energy, time
    # Z and E are normalised by a FIXED reference fleet size (not by n_max), so one spacecraft costs
    # the same whatever the number of slots and cases with different n_max are comparable
    n_ref: int = 10
    infeasible_cost: float = 1e3      # cost of a design that fails feasibility (> any feasible J)
    rho: float = 1e3                  # + rho * sum(violation^2), so infeasible designs can be ranked

    scanners: dict[int, Scanner] = field(default_factory=lambda: dict(SCANNERS))

    # -- derived quantities -------------------------------------------------
    @property
    def omega_ast(self) -> float:
        """Asteroid spin rate [rad/s]."""
        return 2.0 * np.pi / (self.p_rot_h * 3600.0)

    @property
    def dt(self) -> float:
        return self.dt_min * 60.0

    @property
    def t_max(self) -> float:
        return self.t_max_days * 86400.0

    @property
    def t_min(self) -> float:
        return self.t_min_days * 86400.0

    @property
    def duration(self) -> float:
        """Simulation horizon [s] (= T_max); a design's own T is a decision variable."""
        return self.t_max

    @property
    def n_steps(self) -> int:
        """Steps on the simulation grid, N_t = T_max / dt + 1."""
        return int(round(self.duration / self.dt)) + 1

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_steps) * self.dt

    def steps_within(self, T: float) -> int:
        """Number of grid steps t_k = k dt with t_k <= T (at least 1)."""
        return int(np.clip(np.floor(T / self.dt + 1e-9) + 1, 1, self.n_steps))

    @property
    def sun_inertial(self) -> np.ndarray:
        lat = np.deg2rad(self.sun_lat_deg)
        return np.array([np.cos(lat), 0.0, np.sin(lat)])

    @property
    def image_bits(self) -> float:
        p = max(s.pixels for s in self.scanners.values())
        return p * p * 12.0

    @property
    def t_downlink(self) -> float:
        """Seconds to downlink one frame (6.3 MB at 100 kbps -> 503 s)."""
        return self.image_bits / self.downlink_bps

    @property
    def e_tx_j(self) -> float:
        """Transmitter energy per frame (~15 kJ)."""
        return self.p_tx_w * self.t_downlink

    @property
    def v0(self) -> float:
        """Circular speed of the arrival orbit [km/s]."""
        return float(np.sqrt(self.mu / self.r0))

    @property
    def dv_ref(self) -> float:
        """Delta-V of one spacecraft that normalises E (3 v0 > the largest transfer + plane change)."""
        return 3.0 * self.v0

    @property
    def e_ref_j(self) -> float:
        """Electrical energy of one spacecraft running everything for the longest mission T_max."""
        return (self.p_base_w + self.p_cam_w + self.p_tx_w) * self.t_max

    def unit_cost(self) -> float:
        """Cost of one spacecraft: bus + the camera every spacecraft carries."""
        return BUS_COST + self.scanners[int(self.camera)].cost

    def fov(self, scanner_type: int) -> float:
        return np.deg2rad(self.scanners[int(scanner_type)].fov_deg)

    def max_range(self, scanner_type: int) -> float:
        """l_max = GSD_max * p / (2 tan(psi/2))  [km]."""
        s = self.scanners[int(scanner_type)]
        return (self.gsd_max_m / 1000.0) * s.pixels / (2.0 * np.tan(np.deg2rad(s.fov_deg) / 2.0))

    def resolve_r_min(self, asteroid_r_max: float) -> float:
        return self.r_min if self.r_min is not None else self.kappa * asteroid_r_max

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("scanners")
        d["weights"] = list(self.weights)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MissionConfig":
        if not d:
            return cls()
        d = dict(d)
        if "duration_days" in d:                     # older files: the fixed mission time = T_max
            d.setdefault("t_max_days", d.pop("duration_days"))
        d.pop("coverage_floor", None)                # older files: used by the former 1/S_a cost
        if "weights" in d and len(d["weights"]) != 4:
            raise ValueError("weights must be 4 numbers (coverage, cost, energy, time)")
        names = {f.name for f in fields(cls)} - {"scanners"}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown MissionConfig keys: {sorted(unknown)}")
        kwargs = {k: (tuple(v) if k == "weights" else v) for k, v in d.items()}
        return replace(cls(), **kwargs)


def for_body(name: str, base: MissionConfig | None = None) -> MissionConfig:
    """A config whose body-dependent values (GM, spin, resolution, orbit band, ...) are those of ``name``."""
    if name not in BODIES:
        raise ValueError(f"unknown body {name!r}; available: {sorted(BODIES)}")
    return replace(base or MissionConfig(), body=name, **BODIES[name])

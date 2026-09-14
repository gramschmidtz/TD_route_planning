"""
Time-dependent MISSION VALUE per node,  v_j(t).

A new edge/arrival cost for the orbital network: instead of (only) travel time
and dV, each NON-origin node j carries a mission value that depends on the
ARRIVAL node j and the arrival time t -- i.e. v_j(t), a function of j and t
(not of the arc).  Reaching point j when v_j is high is worth more.

With 10 nodes (origin = 1, destination = n = 10) there are 9 non-origin nodes:
  * intermediate nodes 2..n-1: a MIXTURE OF 2-4 Gaussian "value windows"
    (times at which visiting that orbit is operationally valuable);
  * the destination n: a SINGLE Gaussian peaking once, at dest_peak_day (1000 d).

Each Gaussian component is  A * exp(-(t - mu)^2 / (2 sigma^2)).  All parameters
are drawn once from a seeded RNG, so the landscape is fully reproducible.

Later this plugs into the planner as a time-dependent arrival reward
v_j(t_arrival); for now the model + a 9-panel plot of v_j(t) are provided.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class MissionValue:
    """Reproducible time-dependent mission value v_j(t) for the non-origin nodes."""

    nodes: list                       # non-origin node ids (e.g. 2..10)
    destination: int                  # destination node id (single peak)
    horizon: float = 1500.0           # days spanned by the value landscape
    dest_peak_day: float = 1000.0     # destination's lone peak time
    dest_sigma: float = 130.0         # destination peak width (days)
    peaks_range: tuple = (2, 4)       # #Gaussian peaks per intermediate node
    amp_range: tuple = (0.5, 1.0)     # peak amplitudes
    sigma_range: tuple = (60.0, 180.0)  # peak widths (days)
    seed: int = 42
    # node -> list of (A, mu, sigma) Gaussian components
    peaks: dict = field(default_factory=dict)

    def __post_init__(self):
        rng = random.Random(self.seed)
        for j in self.nodes:
            if j == self.destination:
                self.peaks[j] = [(1.0, self.dest_peak_day, self.dest_sigma)]
                continue
            k = rng.randint(*self.peaks_range)
            comps = []
            for _ in range(k):
                A = rng.uniform(*self.amp_range)
                mu = rng.uniform(0.06 * self.horizon, 0.94 * self.horizon)
                sig = rng.uniform(*self.sigma_range)
                comps.append((A, mu, sig))
            self.peaks[j] = sorted(comps, key=lambda c: c[1])

    def value(self, j, t: float) -> float:
        """Mission value v_j(t) (sum of the node's Gaussian components)."""
        return sum(A * math.exp(-(t - mu) ** 2 / (2.0 * sig * sig))
                   for A, mu, sig in self.peaks[j])

    def peak_times(self, j) -> list:
        return [mu for _, mu, _ in self.peaks[j]]


def from_config(cfg: dict, onet) -> "MissionValue":
    """Build a MissionValue from the `orbital.mission` config section and an
    OrbitalNetwork (non-origin nodes + destination taken from the graph)."""
    O = onet.G.graph.get("origin", 1)
    D = onet.G.graph.get("destination", 10)
    mc = cfg.get("orbital", {}).get("mission", {})
    return MissionValue(
        nodes=[v for v in onet.G.nodes() if v != O], destination=D,
        horizon=mc.get("horizon_days", 1500.0),
        dest_peak_day=mc.get("dest_peak_day", 1000.0),
        dest_sigma=mc.get("dest_sigma", 130.0),
        peaks_range=tuple(mc.get("peaks_range", (2, 4))),
        seed=mc.get("seed", 42))

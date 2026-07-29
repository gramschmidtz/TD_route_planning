"""
Build / serialise the 10-orbit fully connected orbital network.

The instance is DESIGNED so the wait-for-RAAN-window behaviour is the dominant
trade-off (see td_route/orbital.py for the physics):

* 10 circular orbits sharing inclination i = 51.6 deg, at DISTINCT altitudes
  spanning ~520-1400 km.  Distinct altitudes => every pair precesses at a
  different J2 rate => their relative RAAN sweeps through 0 with a finite
  synodic period, so every arc has recurring alignment windows.
* Each node also has a RAAN at epoch (raan0), staggered around the circle so
  the windows of different arcs fall at different times -- the planner must
  time departures.
* Fully connected directed graph (every ordered pair i != j is an arc).

Node ids are 1..10; origin = 1 (lowest orbit), destination = 10 (highest),
mirroring the road dataset's O/D convention.
"""
from __future__ import annotations

import json

from .orbital import OrbitalNetwork

# (altitude_km, raan0_deg) for nodes 1..10.  Altitudes are all distinct and
# span LEO; raan0 values are staggered so alignment windows are spread in time.
DEFAULT_ORBITS = [
    (520.0,   0.0),    # 1  origin  (lowest, fastest westward drift)
    (610.0, 145.0),    # 2
    (705.0,  40.0),    # 3
    (800.0, 205.0),    # 4
    (890.0,  95.0),    # 5
    (980.0, 300.0),    # 6
    (1075.0, 120.0),   # 7
    (1170.0, 250.0),   # 8
    (1280.0, 175.0),   # 9
    (1400.0,  60.0),   # 10 destination (highest, slowest drift)
]

DEFAULT_ORIGIN = 1
DEFAULT_DESTINATION = 10


def build_orbital_network(orbits=None, inclination_deg=51.6,
                          thrust_N=0.021, isp_s=2000.0,
                          wet_mass_kg=400.0, fuel_mass_kg=100.0,
                          ref_mass_kg=None,
                          t_start=0.0, t_end=730.0, dt_days=0.5,
                          transit_drift=True,
                          origin=DEFAULT_ORIGIN,
                          destination=DEFAULT_DESTINATION) -> OrbitalNetwork:
    """Create a fully connected OrbitalNetwork from a list of (alt, raan0)."""
    orbits = orbits if orbits is not None else DEFAULT_ORBITS
    net = OrbitalNetwork(
        thrust_N=thrust_N, isp_s=isp_s,
        wet_mass_kg=wet_mass_kg, fuel_mass_kg=fuel_mass_kg,
        ref_mass_kg=ref_mass_kg, inclination_deg=inclination_deg,
        t_start=t_start, t_end=t_end, dt_days=dt_days,
        transit_drift=transit_drift,
    )
    for idx, (alt, raan0) in enumerate(orbits, start=1):
        net.G.add_node(idx, alt=float(alt), raan0=float(raan0))
    nodes = list(net.G.nodes())
    for i in nodes:
        for j in nodes:
            if i != j:
                net.G.add_edge(i, j)
    net.G.graph["origin"] = origin
    net.G.graph["destination"] = destination
    return net


# -------------------------------------------------------------------------- #
# Serialisation (self-contained; arc costs are pure functions of the stored
# node attributes + spacecraft params, so nothing per-arc needs saving).
# -------------------------------------------------------------------------- #
def serialize_network(net: OrbitalNetwork, meta: dict | None = None) -> dict:
    return {
        "meta": {
            "inclination_deg": net.inclination_deg,
            "thrust_N": net.thrust_N, "isp_s": net.isp_s,
            "wet_mass_kg": net.wet_mass_kg, "fuel_mass_kg": net.fuel_mass_kg,
            "ref_mass_kg": net.ref_mass_kg,
            "t_start": net.t_start, "t_end": net.t_end, "dt_days": net.dt_days,
            "transit_drift": net.transit_drift,
            "origin": net.G.graph.get("origin"),
            "destination": net.G.graph.get("destination"),
            **(meta or {}),
        },
        "nodes": [{"id": v, "alt": d["alt"], "raan0": d["raan0"]}
                  for v, d in net.G.nodes(data=True)],
    }


def save_network_json(net: OrbitalNetwork, path: str, meta: dict | None = None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_network(net, meta), f, indent=2)


def load_network_json(path: str) -> OrbitalNetwork:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data["meta"]
    orbits = [(nd["alt"], nd["raan0"]) for nd in sorted(data["nodes"],
                                                        key=lambda n: n["id"])]
    return build_orbital_network(
        orbits=orbits, inclination_deg=m["inclination_deg"],
        thrust_N=m["thrust_N"], isp_s=m["isp_s"],
        wet_mass_kg=m["wet_mass_kg"], fuel_mass_kg=m["fuel_mass_kg"],
        ref_mass_kg=m.get("ref_mass_kg"),
        t_start=m["t_start"], t_end=m["t_end"], dt_days=m["dt_days"],
        transit_drift=m.get("transit_drift", True),
        origin=m.get("origin", DEFAULT_ORIGIN),
        destination=m.get("destination", DEFAULT_DESTINATION),
    )

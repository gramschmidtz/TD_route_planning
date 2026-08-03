"""
Build / serialise the fully connected orbital mission network.

The instance is SIZED FOR THE IMPULSIVE CHASER (490 N, Isp 315.5 s, 400 kg wet /
100 kg fuel), whose whole budget is

    dV = Isp*g0*ln(400/300) = 890 m/s

Design arithmetic behind the numbers below (see td_route/orbital.py for the
physics):
  * a Hohmann altitude step costs ~54 m/s per 100 km, so the 500->950 km span
    costs ~240 m/s one way -- leaving ~650 m/s for everything else;
  * an impulsive plane change costs ~103 m/s per DEGREE of residual RAAN, so the
    route must depart inside J2 alignment windows (a 1-day grid leaves at most
    half a day's drift, i.e. ~0.1-0.5 deg -> 10-50 m/s per leg);
  * 8 nodes / 7 legs therefore fit comfortably: ~240 m/s of altitude change plus
    ~150-300 m/s of residual plane change, well inside 890 m/s.
Altitudes are all DISTINCT, so every pair precesses at a different J2 rate and
its relative RAAN sweeps through zero with a finite synodic period (recurring
windows).  raan0 DECREASES with altitude so those windows fall in the future
(dOmega starts negative and drifts up through 0), staggered so different arcs
peak at different times.  theta0 is the argument of latitude at epoch, drawn
per node -- with true-anomaly matching it decides WHICH instant of a day the
rendezvous phase actually lines up.

Fully connected directed graph (every ordered pair i != j is an arc).
Node ids are 1..n; origin = 1 (lowest orbit), destination = n (highest).
"""
from __future__ import annotations

import json
import random

from .orbital import OrbitalNetwork

# (altitude_km, raan0_deg, theta0_deg) for nodes 1..8, produced by
# design_orbits(DESIGN_ALTS, DESIGN_WINDOWS, seed=7) -- so consecutive pairs have
# alignment windows at 120, 260, 400, 540, 680, 820, 960 days.
DEFAULT_ORBITS = [
    (500.0,   0.0, 116.6),   # 1  origin  (lowest, fastest westward drift)
    (565.0, 341.5,  54.3),   # 2
    (630.0, 303.2, 234.3),   # 3
    (700.0, 242.3,  26.1),   # 4
    (775.0, 158.2, 192.9),   # 5
    (845.0,  63.8, 131.6),   # 6
    (900.0, 337.8,  20.9),   # 7
    (950.0, 249.2, 182.7),   # 8  destination (highest, slowest drift)
]

DEFAULT_ORIGIN = 1
DEFAULT_DESTINATION = 8


def design_orbits(alts, window_days, inclination_deg=51.6, seed=7):
    """Place raan0 so each CONSECUTIVE pair's alignment window falls at a chosen
    time, and draw a random phase per node.

    Adjacent LEO altitudes precess only ~0.1 deg/day apart, so a pair's relative
    RAAN needs thousands of days to sweep a full turn: inside a ~1000-day horizon
    each adjacent pair gets essentially ONE window, whose date is set purely by
    its initial offset.  That offset is therefore the design knob --

        dOmega_k(t) = (raan0_{k+1} - raan0_k) + (Omdot_{k+1} - Omdot_k) t = 0
        =>  raan0_{k+1} = raan0_k - (Omdot_{k+1} - Omdot_k) * window_days[k]

    -- and chaining it from raan0_1 = 0 puts the k-th window exactly at
    window_days[k].  Windows for non-consecutive pairs then follow from the same
    rates, giving a mix of reachable and unreachable arcs (a non-trivial
    instance) instead of the degenerate "everything aligns at once".
    """
    from .orbital import J2, MU, RE, DEG, SECONDS_PER_DAY, RAD2DEG
    import math as _m

    def rate(alt):                                   # J2 nodal precession
        a = RE + alt
        return (-1.5 * J2 * _m.sqrt(MU) * RE**2 / a**3.5
                * _m.cos(inclination_deg * DEG)) * SECONDS_PER_DAY * RAD2DEG

    rng = random.Random(seed)
    raan = [0.0]
    for k in range(len(alts) - 1):
        raan.append(raan[k] - (rate(alts[k + 1]) - rate(alts[k]))
                    * window_days[k])
    return [(round(a, 1), round(r % 360.0, 1), round(rng.uniform(0, 360), 1))
            for a, r in zip(alts, raan)]


# The altitudes / window schedule the defaults above were generated from
# (kept so the design is reproducible and easy to re-tune).
DESIGN_ALTS = [500.0, 565.0, 630.0, 700.0, 775.0, 845.0, 900.0, 950.0]
DESIGN_WINDOWS = [120.0, 260.0, 400.0, 540.0, 680.0, 820.0, 960.0]


def build_orbital_network(orbits=None, inclination_deg=51.6,
                          thrust_N=490.0, isp_s=315.5,
                          wet_mass_kg=400.0, fuel_mass_kg=100.0,
                          ref_mass_kg=None,
                          t_start=0.0, t_end=1300.0, dt_days=1.0,
                          transit_drift=True, propulsion="impulsive",
                          phase_tol_deg=0.05,
                          origin=DEFAULT_ORIGIN,
                          destination=None) -> OrbitalNetwork:
    """Create a fully connected OrbitalNetwork from (alt, raan0[, theta0])."""
    orbits = orbits if orbits is not None else DEFAULT_ORBITS
    net = OrbitalNetwork(
        thrust_N=thrust_N, isp_s=isp_s,
        wet_mass_kg=wet_mass_kg, fuel_mass_kg=fuel_mass_kg,
        ref_mass_kg=ref_mass_kg, inclination_deg=inclination_deg,
        t_start=t_start, t_end=t_end, dt_days=dt_days,
        transit_drift=transit_drift, propulsion=propulsion,
        phase_tol_deg=phase_tol_deg,
    )
    destination = len(orbits) if destination is None else destination
    for idx, rec in enumerate(orbits, start=1):
        alt, raan0 = float(rec[0]), float(rec[1])
        theta0 = float(rec[2]) if len(rec) > 2 else 0.0
        net.G.add_node(idx, alt=alt, raan0=raan0, theta0=theta0)
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
            "propulsion": net.propulsion, "phase_tol_deg": net.phase_tol_deg,
            "origin": net.G.graph.get("origin"),
            "destination": net.G.graph.get("destination"),
            **(meta or {}),
        },
        "nodes": [{"id": v, "alt": d["alt"], "raan0": d["raan0"],
                   "theta0": d.get("theta0", 0.0)}
                  for v, d in net.G.nodes(data=True)],
    }


def save_network_json(net: OrbitalNetwork, path: str, meta: dict | None = None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_network(net, meta), f, indent=2)


def load_network_json(path: str) -> OrbitalNetwork:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data["meta"]
    orbits = [(nd["alt"], nd["raan0"], nd.get("theta0", 0.0))
              for nd in sorted(data["nodes"], key=lambda n: n["id"])]
    return build_orbital_network(
        orbits=orbits, inclination_deg=m["inclination_deg"],
        thrust_N=m["thrust_N"], isp_s=m["isp_s"],
        wet_mass_kg=m["wet_mass_kg"], fuel_mass_kg=m["fuel_mass_kg"],
        ref_mass_kg=m.get("ref_mass_kg"),
        t_start=m["t_start"], t_end=m["t_end"], dt_days=m["dt_days"],
        transit_drift=m.get("transit_drift", True),
        propulsion=m.get("propulsion", "impulsive"),
        phase_tol_deg=m.get("phase_tol_deg", 0.05),
        origin=m.get("origin", DEFAULT_ORIGIN),
        destination=m.get("destination"),
    )

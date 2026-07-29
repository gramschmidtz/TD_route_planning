"""
Turn one solved route (with its timed schedule) into a frame-by-frame
trajectory that both the MP4 (matplotlib) and the interactive HTML (Three.js)
renderers consume.

Given the route and its timeline (node, arrival, departure -- in days, from the
EECM waiting solution's meta), the chaser's state is sampled over elapsed time:

  * WAITING at a node v in [arrival, departure]: the chaser sits on orbit v; its
    RAAN drifts with v under J2, dV does not change.
  * FLYING leg m from i to j in [departure_i, arrival_j]: altitude interpolates
    a_i -> a_j and the plane interpolates RAAN_i(t_dep) -> RAAN_j(t_arr) so the
    chaser lands exactly on orbit j at arrival; dV accrues linearly with the
    continuous burn (fraction of the leg's dV).

Every NODE orbit also precesses over the animation, so the viewer literally sees
the chaser wait for a RAAN window before departing.  Per-leg dV/ToF-vs-departure
curves (for the right-hand panels) are precomputed with the transit-drift model.
"""
from __future__ import annotations

import math

from .orbital import OrbitalNetwork, RE, wrap180


def _leg_curve(onet: OrbitalNetwork, i, j, t_dep, n=140):
    """dV / ToF vs departure time for arc i->j, over a window around t_dep."""
    syn = onet.synodic_period(i, j)
    half = 0.6 * (syn if math.isfinite(syn) else 500.0)
    half = max(60.0, min(half, 900.0))
    lo = max(0.0, t_dep - half)
    hi = t_dep + half
    ts, dvs, tofs = [], [], []
    for k in range(n + 1):
        t = lo + (hi - lo) * k / n
        tr = onet.transfer(i, j, t)
        ts.append(round(t, 3))
        dvs.append(round(tr["dv"], 2) if tr["feasible"] else None)
        tofs.append(round(tr["tof"], 3) if tr["feasible"] else None)
    tr0 = onet.transfer(i, j, t_dep)
    floor = abs(onet.circ_speed(i) - onet.circ_speed(j)) * 1000.0
    return {"t": ts, "dv": dvs, "tof": tofs, "floor": round(floor, 2),
            "t_dep": round(t_dep, 3), "dv_dep": round(tr0["dv"], 2),
            "tof_dep": round(tr0["tof"], 3)}


def build_trajectory(onet: OrbitalNetwork, route, timeline, n_frames=240,
                     nu_period_days=6.0, mission=None):
    """Build the full trajectory dict (JSON-serialisable) for the renderers.

    route    : node id sequence O..D
    timeline : [{"node", "arrival", "departure"}] in days (departure None at D)
    mission  : optional MissionValue -> adds per-leg arrival value, per-frame
               cumulative value, and the shared per-node v_j(t) curves.
    """
    # --- legs -------------------------------------------------------------
    legs = []
    cum = 0.0
    cum_before = [0.0]
    for m in range(len(route) - 1):
        i, j = route[m], route[m + 1]
        t_dep = float(timeline[m]["departure"])
        t_arr = float(timeline[m + 1]["arrival"])
        dv = onet.transfer(i, j, t_dep)["dv"]
        legs.append({
            "i": i, "j": j, "t_dep": t_dep, "t_arr": t_arr,
            "tof": t_arr - t_dep, "dv": round(dv, 2),
            "wait": float(timeline[m]["departure"] - timeline[m]["arrival"]),
            "curve": _leg_curve(onet, i, j, t_dep),
            "arr_value": (round(mission.value(j, t_arr), 4)
                          if mission is not None else None),
        })
        cum += dv
        cum_before.append(cum)
    total_dv = cum
    total_days = float(timeline[-1]["arrival"])
    arr_times = [lg["t_arr"] for lg in legs]
    arr_vals = [lg["arr_value"] or 0.0 for lg in legs]
    total_value = sum(arr_vals) if mission is not None else None

    # --- per-frame chaser state ------------------------------------------
    def state(tau):
        # locate the segment containing tau
        for m in range(len(route) - 1):
            arr_i = float(timeline[m]["arrival"])
            dep_i = float(timeline[m]["departure"])
            arr_j = float(timeline[m + 1]["arrival"])
            if arr_i <= tau <= dep_i:                       # waiting at route[m]
                v = route[m]
                return dict(leg=m, flying=False, alt=onet.G.nodes[v]["alt"],
                            raan=onet.raan_at(v, tau), cum_dv=cum_before[m])
            if dep_i < tau <= arr_j:                        # flying leg m
                i, j = route[m], route[m + 1]
                f = (tau - dep_i) / (arr_j - dep_i) if arr_j > dep_i else 1.0
                ai, aj = onet.G.nodes[i]["alt"], onet.G.nodes[j]["alt"]
                r0 = onet.raan_at(i, dep_i)
                r1 = onet.raan_at(j, arr_j)
                while r1 - r0 > 180.0:
                    r1 -= 360.0
                while r1 - r0 < -180.0:
                    r1 += 360.0
                return dict(leg=m, flying=True, alt=ai + f * (aj - ai),
                            raan=r0 + f * (r1 - r0),
                            cum_dv=cum_before[m] + f * legs[m]["dv"])
        # tau at/after final arrival -> sit on destination
        v = route[-1]
        return dict(leg=len(route) - 2, flying=False, alt=onet.G.nodes[v]["alt"],
                    raan=onet.raan_at(v, tau), cum_dv=total_dv)

    frames = []
    for k in range(n_frames + 1):
        tau = total_days * k / n_frames
        s = state(tau)
        nu = 2.0 * math.pi * (tau / nu_period_days % 1.0)
        cum_value = (sum(v for at, v in zip(arr_times, arr_vals) if at <= tau)
                     if mission is not None else 0.0)
        frames.append({
            "t": round(tau, 3), "leg": s["leg"], "flying": s["flying"],
            "alt": round(s["alt"], 3), "a": round(RE + s["alt"], 3),
            "raan": round(wrap180(s["raan"]), 3), "nu": round(nu, 4),
            "cum_dv": round(s["cum_dv"], 2), "cum_value": round(cum_value, 4),
        })

    nodes = [{"id": v, "alt": onet.G.nodes[v]["alt"],
              "a": round(onet.sma(v), 3),
              "raan0": onet.G.nodes[v]["raan0"],
              "rate": round(onet.raan_rate(v), 5)}
             for v in onet.G.nodes()]

    # shared per-node mission-value curves v_j(t) (same for every solution)
    mission_curves = {}
    if mission is not None:
        H = getattr(mission, "horizon", max(total_days * 1.1, 1500.0))
        tm = [round(H * k / 200, 2) for k in range(201)]
        for j in mission.nodes:
            mission_curves[str(j)] = {
                "t": tm, "v": [round(mission.value(j, t), 4) for t in tm]}

    return {
        "meta": {
            "route": list(route), "inclination": onet.inclination_deg,
            "RE": RE, "wet_mass": onet.wet_mass_kg, "fuel": onet.fuel_mass_kg,
            "dv_budget": round(onet.dv_budget, 1),
            "total_days": round(total_days, 2), "total_dv": round(total_dv, 2),
            "total_value": (None if total_value is None else round(total_value, 4)),
            "has_mission": mission is not None,
        },
        "nodes": nodes, "legs": legs, "frames": frames,
        "mission_curves": mission_curves,
    }

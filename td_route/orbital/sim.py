"""
Turn one solved route (with its timed schedule) into a frame-by-frame
trajectory that both the MP4 (matplotlib) and the interactive HTML (Three.js)
renderers consume.

Impulsive model (propulsion="impulsive")
---------------------------------------
Each leg has two very different timescales:

  * WAITING on orbit i, from the arrival at i until the phase-matched departure
    -- days to months.  The chaser sits ON node i (same orbit, same position).
  * the TRANSFER itself -- a 2-impulse Hohmann, only ~50 minutes, during which
    the chaser coasts half of the transfer ellipse (true anomaly 0->180 going
    up, 180->360 coming down) and arrives exactly on top of target j.

Sampling both at one uniform rate is useless: at day-per-frame the 50-minute
transfer is a single frame, and at minute-per-frame the waits take millions of
frames.  So the frame schedule is NON-UNIFORM -- long waits get frames in
proportion to their length, and every transfer gets a fixed budget of frames
(`xfer_frames`) so the rendezvous is actually watchable.  Each frame carries the
timescale it belongs to ("wait" / "xfer") so the renderers can say so.

Every frame describes the chaser as an orbit + a position on it:
    cr = [a, e, argp, raan]  (km, -, deg, deg)   and   cnu = true anomaly (deg)
which is a circle (e=0) while waiting and the transfer ellipse while flying, so
the renderers can draw the real conic and the real position.  Node positions are
reconstructed by the renderers from theta0 + phase rate, so true-anomaly
matching (chaser dot meeting the target dot) is visible.

dV is applied AT the two impulses (departure and arrival), not smeared over the
leg, which is what an impulsive burn actually does.
"""
from __future__ import annotations

import math

from .network import OrbitalNetwork, RE, wrap180


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
        tofs.append(round(tr["tof"], 4) if tr["feasible"] else None)
    tr0 = onet.transfer(i, j, t_dep)
    floor = abs(onet.circ_speed(i) - onet.circ_speed(j)) * 1000.0
    return {"t": ts, "dv": dvs, "tof": tofs, "floor": round(floor, 2),
            "t_dep": round(t_dep, 3), "dv_dep": round(tr0["dv"], 2),
            "tof_dep": round(tr0["tof"], 4)}


def build_trajectory(onet: OrbitalNetwork, route, timeline, n_frames=240,
                     mission=None, xfer_frames=18):
    """Build the full trajectory dict (JSON-serialisable) for the renderers.

    route    : node id sequence O..D
    timeline : [{"node", "arrival", "departure"}] in days (departure None at D)
    mission  : optional MissionValue -> per-leg arrival value, per-frame
               cumulative value, and the shared per-node v_j(t) curves.
    """
    impulsive = (onet.propulsion == "impulsive")

    # ------------------------------------------------------------------ #
    # Legs: geometry of each transfer + the impulses that bracket it
    # ------------------------------------------------------------------ #
    legs = []
    cum_dv = 0.0
    for m in range(len(route) - 1):
        i, j = route[m], route[m + 1]
        t_arr_i = float(timeline[m]["arrival"])
        t_dep = float(timeline[m]["departure"])          # step-grid departure
        t_arr_j = float(timeline[m + 1]["arrival"])
        tr = onet.transfer(i, j, t_dep)
        ri, rj = onet.sma(i), onet.sma(j)
        at = 0.5 * (ri + rj)
        et = abs(rj - ri) / (ri + rj)
        # the burn happens at the PHASE-MATCHED instant inside the step
        tau = t_dep + tr.get("phase_wait", 0.0) if impulsive else t_dep
        tof = tr.get("hohmann_tof", tr["tof"]) if impulsive else tr["tof"]
        u_dep = onet.phase_at(i, tau) if impulsive else 0.0
        ascending = rj >= ri
        # transfer plane: the plane change is folded into the burn at the LARGER
        # radius, so the ellipse lies in the plane of the other end
        raan_x = (onet.raan_at(i, tau) if ascending
                  else onet.raan_at(j, tau + tof))
        legs.append({
            "i": i, "j": j,
            "t_arr_i": t_arr_i, "t_dep_step": t_dep, "tau": tau,
            "t_burn2": tau + tof, "t_arr_j": t_arr_j,
            "wait": tau - t_arr_i, "tof": tof, "phase_wait": tr.get("phase_wait", 0.0),
            "dv": round(tr["dv"], 3),
            "dv1": round(tr.get("dv1", tr["dv"] * 0.5), 3),
            "dv2": round(tr.get("dv2", tr["dv"] * 0.5), 3),
            "draan": round(tr.get("draan_thrust", 0.0), 4),
            "at": round(at, 3), "et": round(et, 6),
            "argp": round((u_dep if ascending else u_dep + 180.0) % 360.0, 4),
            "raan_x": round(raan_x % 360.0, 4),
            "u_dep": round(u_dep % 360.0, 4), "ascending": ascending,
            "cum_dv_before": round(cum_dv, 3),
            "arr_value": (round(mission.value(j, t_arr_j), 4)
                          if mission is not None else None),
            "curve": _leg_curve(onet, i, j, t_dep),
        })
        cum_dv += tr["dv"]
    total_dv = cum_dv
    total_days = float(timeline[-1]["arrival"])
    total_value = (sum(lg["arr_value"] for lg in legs)
                   if mission is not None else None)

    # ------------------------------------------------------------------ #
    # Non-uniform frame schedule: waits get frames pro rata, every transfer
    # gets `xfer_frames` so the ~50-minute rendezvous is visible.
    # ------------------------------------------------------------------ #
    wait_spans = [(lg["t_arr_i"], lg["tau"], m) for m, lg in enumerate(legs)]
    wait_total = sum(max(0.0, b - a) for a, b, _ in wait_spans) or 1.0
    n_wait_budget = max(1, n_frames - xfer_frames * len(legs))

    def node_state(v, t, leg, scale):
        return {"t": round(t, 5), "leg": leg, "flying": False, "scale": scale,
                "cr": [round(onet.sma(v), 3), 0.0, 0.0,
                       round(onet.raan_at(v, t) % 360.0, 4)],
                "cnu": round(onet.phase_at(v, t) % 360.0, 4),
                "alt": round(onet.G.nodes[v]["alt"], 3)}

    frames = []
    for m, lg in enumerate(legs):
        # --- waiting on orbit i ---
        span = max(0.0, lg["tau"] - lg["t_arr_i"])
        nw = max(1, int(round(n_wait_budget * span / wait_total)))
        for k in range(nw):
            t = lg["t_arr_i"] + span * k / nw
            fr = node_state(lg["i"], t, m, "wait")
            fr["cum_dv"] = lg["cum_dv_before"]
            frames.append(fr)
        # --- transfer: coast the ellipse, first impulse already spent ---
        for k in range(xfer_frames + 1):
            f = k / xfer_frames
            t = lg["tau"] + lg["tof"] * f
            nu = (0.0 + 180.0 * f) if lg["ascending"] else (180.0 + 180.0 * f)
            r = (lg["at"] * (1.0 - lg["et"] ** 2)
                 / (1.0 + lg["et"] * math.cos(math.radians(nu))))
            frames.append({
                "t": round(t, 5), "leg": m, "flying": True, "scale": "xfer",
                "cr": [lg["at"], lg["et"], lg["argp"], lg["raan_x"]],
                "cnu": round(nu % 360.0, 4),
                "alt": round(r - RE, 3),
                "cum_dv": round(lg["cum_dv_before"] + lg["dv1"], 3)})
    # --- final hold on the destination ---
    lastj = route[-1]
    t_end = float(timeline[-1]["arrival"])
    for k in range(max(2, xfer_frames // 3)):
        fr = node_state(lastj, t_end + k * 0.02, len(legs) - 1, "wait")
        fr["cum_dv"] = round(total_dv, 3)
        frames.append(fr)

    # cumulative mission value (steps at each arrival)
    if mission is not None:
        arr = [(lg["t_arr_j"], lg["arr_value"]) for lg in legs]
        for fr in frames:
            fr["cum_value"] = round(sum(v for at, v in arr if at <= fr["t"]), 4)
    else:
        for fr in frames:
            fr["cum_value"] = 0.0

    nodes = [{"id": v, "alt": onet.G.nodes[v]["alt"],
              "a": round(onet.sma(v), 3),
              "raan0": onet.G.nodes[v]["raan0"],
              "rate": round(onet.raan_rate(v), 6),
              "theta0": onet.G.nodes[v].get("theta0", 0.0),
              "n": round(onet.phase_rate(v), 4)}
             for v in onet.G.nodes()]

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
            "propulsion": onet.propulsion,
            "total_days": round(total_days, 2), "total_dv": round(total_dv, 2),
            "total_value": (None if total_value is None else round(total_value, 4)),
            "has_mission": mission is not None,
        },
        "nodes": nodes, "legs": legs, "frames": frames,
        "mission_curves": mission_curves,
    }

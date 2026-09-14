"""
Agent-journey trajectory computation (pure, no plotting).

Given a network, a node route O..D and a departure clock time t0, this builds
the continuous space-time trajectory of an agent that traverses the route,
respecting the FIFO time-dependent speed model (speed changes when a horizon
boundary is crossed mid-arc). It exposes per-frame samples used by the
animation in examples/road/simulate_agent.py.

Conventions
-----------
* clock time is minutes from midnight.
* An arc's preference u_ij is evaluated at the arc's DEPARTURE time (matching
  the model f1 = sum u_ijt x_ijt) and, for a smooth "earned so far" curve, is
  accumulated linearly over the arc's travel time.
"""
from __future__ import annotations


def edge_breakpoints(net, i, j, t_dep):
    """Piecewise (clock, cumulative_distance) points for traversing arc i->j
    starting at t_dep, honouring the FIFO speed model. First point is
    (t_dep, 0), last is (arrival, length)."""
    d = net.G[i][j]["length"]
    cat = net.G[i][j]["speed_cat"]
    speeds = net.speed_categories[cat]
    v0 = net.free_flow_speed
    pts = [(float(t_dep), 0.0)]
    t, cum = float(t_dep), 0.0
    guard = 0
    while cum < d - 1e-9:
        guard += 1
        if guard > 10 * (len(net.horizons) + 2):
            break
        k = net.horizon_index(t)
        v = v0 * speeds[k]
        last = k >= len(net.horizons) - 1
        h_end = float("inf") if last else net.horizons[k][1]
        dist_avail = float("inf") if last else v * (h_end - t)
        remaining = d - cum
        if remaining <= dist_avail:
            t += remaining / v
            cum = d
        else:
            cum += dist_avail
            t = h_end
        pts.append((t, cum))
    return pts


def _dist_at(bps, tc):
    """Cumulative distance covered at clock tc (linear interp of breakpoints)."""
    if tc <= bps[0][0]:
        return 0.0
    if tc >= bps[-1][0]:
        return bps[-1][1]
    for (t0, d0), (t1, d1) in zip(bps, bps[1:]):
        if t0 <= tc <= t1:
            if t1 == t0:
                return d1
            return d0 + (d1 - d0) * (tc - t0) / (t1 - t0)
    return bps[-1][1]


def _next_pref_change(net, i, j, t):
    """Next clock time > t at which arc (i,j)'s preference value changes.

    Returns +inf if the arc has no schedule or t is at/after the day's end
    (u stays constant thereafter)."""
    sched = net.G[i][j].get("pref_schedule")
    if not sched:
        return float("inf")
    for a, b, _m in sched:
        if t < a:
            return a
        if a <= t < b:
            return b
    return float("inf")


def pref_time_integral(net, i, j, a, b):
    """Exact integral of the (piecewise-constant) preference u_ij(t) over
    [a, b]. Because u_ij is constant on each schedule state, this is a finite
    sum over the states the interval touches."""
    if b <= a:
        return 0.0
    total, t = 0.0, a
    while t < b:
        nb = min(_next_pref_change(net, i, j, t), b)
        total += net.preference(i, j, t) * (nb - t)
        t = nb
    return total


def build_journey(net, route, t0, dep_times=None):
    """List of travel-segment dicts for the route. Raises on a missing arc.

    Each segment: {i, j, arr_at_i, t_dep, t_arr, u, length, cat, bps, pref_before}.
    `arr_at_i` is when the agent ARRIVES at node i; `t_dep` is when it LEAVES.
    If dep_times is given (departure time from each of route[:-1]), the agent may
    WAIT at a node (t_dep > arr_at_i); otherwise it leaves immediately (no wait).

    The per-arc preference `u` is the TIME-AVERAGE of u_ij(t) over the arc's
    traversal (= integral / travel-time); if a state boundary is crossed mid-arc
    the two levels are blended (time-weighted).
    """
    if len(route) < 2:
        raise SystemExit("route must list at least two nodes (O .. D)")
    segments = []
    t = float(t0)                    # arrival time at the current node route[k]
    pref_acc = 0.0
    for k, (i, j) in enumerate(zip(route, route[1:])):
        if not net.G.has_edge(i, j):
            raise SystemExit(f"route is not a connected path: no arc {i}->{j}")
        arr_at_i = t
        # clamp to the arrival: dep_times from the (integer-ceil) ILP can round
        # a hair below the continuous-FIFO arrival used here; never depart early.
        t_dep = max(float(dep_times[k]), arr_at_i) if dep_times is not None else t
        bps = edge_breakpoints(net, i, j, t_dep)
        t_arr = bps[-1][0]
        T = t_arr - t_dep
        u = (pref_time_integral(net, i, j, t_dep, t_arr) / T if T > 0
             else net.preference(i, j, t_dep))
        segments.append({"i": i, "j": j, "arr_at_i": arr_at_i, "t_dep": t_dep,
                         "t_arr": t_arr, "u": u, "length": net.G[i][j]["length"],
                         "cat": net.G[i][j]["speed_cat"], "bps": bps,
                         "pref_before": pref_acc})
        pref_acc += u
        t = t_arr
    return segments


def speed_at(net, seg, tc):
    """Instantaneous speed on a segment's arc at clock tc."""
    v0 = net.free_flow_speed
    return v0 * net.speed_categories[seg["cat"]][net.horizon_index(tc)]


def sample_journey(net, segments, frame_minutes=0.5):
    """Sample the whole journey (waiting included) on a uniform clock grid.

    Each segment owns the interval [arr_at_i, t_arr]: first a WAIT at node i
    over [arr_at_i, t_dep] (only if the agent waited), then TRAVEL over
    [t_dep, t_arr]. Consecutive segments tile the timeline with no gaps.

    Returns per-frame arrays: clock, seg_idx, frac, x, y, speed, cumpref,
    elapsed, waiting (bool). During a wait the agent sits at node i (speed 0,
    preference flat); seg_idx points at the arc it is about to take.
    """
    pos = {v: (net.G.nodes[v]["x"], net.G.nodes[v]["y"]) for v in net.G.nodes}
    start = segments[0]["arr_at_i"]
    t_end = segments[-1]["t_arr"]
    n = max(2, int(round((t_end - start) / frame_minutes)) + 1)

    out = {k: [] for k in ("clock", "seg_idx", "frac", "x", "y",
                           "speed", "cumpref", "elapsed", "waiting")}
    s = 0
    for f in range(n):
        tc = start + f * frame_minutes
        if f == n - 1 or tc > t_end:
            tc = t_end
        while s < len(segments) - 1 and tc > segments[s]["t_arr"]:
            s += 1
        seg = segments[s]
        xi, yi = pos[seg["i"]]
        xj, yj = pos[seg["j"]]

        if tc < seg["t_dep"]:                     # WAITING at node i
            out["frac"].append(0.0)
            out["x"].append(xi)
            out["y"].append(yi)
            out["speed"].append(0.0)
            out["cumpref"].append(seg["pref_before"])
            out["waiting"].append(True)
        else:                                     # TRAVELLING on arc i->j
            length = seg["length"] or 1.0
            frac = min(1.0, max(0.0, _dist_at(seg["bps"], tc) / length))
            dur = seg["t_arr"] - seg["t_dep"]
            part = (pref_time_integral(net, seg["i"], seg["j"], seg["t_dep"], tc)
                    / dur) if dur > 0 else seg["u"]
            out["frac"].append(frac)
            out["x"].append(xi + (xj - xi) * frac)
            out["y"].append(yi + (yj - yi) * frac)
            out["speed"].append(speed_at(net, seg, tc))
            out["cumpref"].append(seg["pref_before"] + part)
            out["waiting"].append(False)

        out["clock"].append(tc)
        out["seg_idx"].append(s)
        out["elapsed"].append(tc - start)
    return out

"""
Single transformed single-objective ILP over the time-expanded variables.

This is the P_k solver used by both the endpoint computation (Section IV.B) and
the epsilon-constraint Pareto tracer. It builds constraints (3),(4),(7),(8),(9)
explicitly; (5),(6) are already baked into the variable set (see
time_expanded.py).

Objective convention:  f1 = preference (max),  f2 = travel time (min).
"""
from __future__ import annotations

import pulp

from ..decision import Solution
from .time_expanded import TimeExpanded


def solve_single(te: TimeExpanded, sense, *, eps=None, fix_f1=None, fix_f2=None,
                 time_limit=None, msg=False):
    """Build and solve one single-objective ILP.

    sense    : "min_f2" or "max_f1" -- which objective to optimise.
    eps      : if given, add  sum u*x >= eps   (the P2 epsilon-cut, eq. 12).
    fix_f1   : if given, add  sum u*x == fix_f1.
    fix_f2   : if given, add  sum c*x == fix_f2.

    Returns a Solution (route + f1 + f2) or None if infeasible.
    """
    if not te.reachable_destination:
        return None
    if getattr(te, "waiting", False):
        return _solve_waiting(te, sense, eps=eps, fix_f1=fix_f1, fix_f2=fix_f2,
                              time_limit=time_limit, msg=msg)

    prob = pulp.LpProblem("Pk", pulp.LpMinimize)
    x = {k: pulp.LpVariable(f"x_{k[0]}_{k[1]}_{k[2]}", cat="Binary")
         for k in te.arcs}

    f1 = pulp.lpSum(te.arcs[k][1] * x[k] for k in te.arcs)   # preference
    f2 = pulp.lpSum(te.arcs[k][0] * x[k] for k in te.arcs)   # travel time

    prob += (f2 if sense == "min_f2" else -f1)               # objective

    O, D = te.O, te.D

    # (3) exactly one arc leaves O (at t0)
    prob += pulp.lpSum(x[k] for k in te.out_arcs(O)) == 1, "out_origin"
    # (4) exactly one arc enters D
    prob += pulp.lpSum(x[k] for k in te.in_arcs(D)) == 1, "in_dest"
    # (5) no arc enters O and (6) no arc leaves D are enforced structurally in
    # TimeExpanded._build (those variables are never created == fixed to 0).

    for node in te.nodes:
        ins = te.in_arcs(node)
        outs = te.out_arcs(node)
        if ins:                                              # (8) node once
            prob += pulp.lpSum(x[k] for k in ins) <= 1, f"once_{node}"
        if node not in (O, D):                               # (7) flow cons.
            prob += (pulp.lpSum(x[k] for k in ins) ==
                     pulp.lpSum(x[k] for k in outs)), f"flow_{node}"

    # (9) timing: leaving an intermediate node happens at the arrival time
    for (i, j, t), (c, _u) in te.arcs.items():
        if j == D:
            continue
        succ = [(j, k2, t + c) for k2 in te.net.G.successors(j)
                if (j, k2, t + c) in te.arcs]
        prob += x[(i, j, t)] <= pulp.lpSum(x[s] for s in succ), \
            f"time_{i}_{j}_{t}"

    if eps is not None:
        prob += f1 >= eps, "eps_cut"
    if fix_f1 is not None:
        prob += f1 == fix_f1, "fix_f1"
    if fix_f2 is not None:
        prob += f2 == fix_f2, "fix_f2"

    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return extract_solution(te, x)


def extract_solution(te: TimeExpanded, x) -> Solution:
    """Rebuild the route and objective values from a solved model."""
    chosen = {k for k in te.arcs if x[k].value() and x[k].value() > 0.5}
    nxt = {i: (i, j, t) for (i, j, t) in chosen}
    route, f1, f2 = [te.O], 0.0, 0
    cur = te.O
    while cur in nxt and cur != te.D:
        i, j, t = nxt[cur]
        c, u = te.arcs[(i, j, t)]
        f1 += u
        f2 += c
        route.append(j)
        cur = j
        if len(route) > len(te.nodes) + 1:        # safety against cycles
            break
    return Solution(route=route, f1=float(f1), f2=int(f2), meta={"t0": te.t0})


# -------------------------------------------------------------------------- #
# Waiting model: time-expanded graph with wait arcs (equivalent to the
# summation constraint (9') but sparse). f2 = arrival(D) - t0, i.e. total
# elapsed time INCLUDING waiting.
# -------------------------------------------------------------------------- #
def _solve_waiting(te, sense, *, eps, fix_f1, fix_f2, time_limit, msg):
    prob = pulp.LpProblem("Pk_wait", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}_{a[2]}", cat="Binary")
         for a in te.arcs}
    w = {a: pulp.LpVariable(f"w_{a[0]}_{a[1]}", cat="Binary") for a in te.waits}

    O, D, t0 = te.O, te.D, te.t0
    f1 = pulp.lpSum(te.arcs[a][1] * x[a] for a in te.arcs)
    into_D = [a for a in te.arcs if a[1] == D]           # arcs (i, D, t)
    # arrival at D = t + c for the single selected arc into D; f2 = arrival - t0
    f2 = pulp.lpSum((a[2] + te.arcs[a][0]) * x[a] for a in into_D) - t0

    prob += (f2 if sense == "min_f2" else -f1)

    # (3) exactly one arc leaves O at t0
    prob += pulp.lpSum(x[a] for a in te.departures.get((O, t0), [])) == 1, "out_O"
    # (4) exactly one arc enters D
    prob += pulp.lpSum(x[a] for a in into_D) == 1, "in_D"

    # time-node flow balance for every reachable (v, tau), v not in {O, D}
    for (v, tau) in te.time_nodes:
        if v == O or v == D:
            continue
        inflow = [x[a] for a in te.arrivals.get((v, tau), [])]
        if (v, tau - 1) in te.waits:
            inflow.append(w[(v, tau - 1)])
        outflow = [x[a] for a in te.departures.get((v, tau), [])]
        if (v, tau) in te.waits:
            outflow.append(w[(v, tau)])
        if inflow or outflow:
            prob += pulp.lpSum(inflow) == pulp.lpSum(outflow), f"bal_{v}_{tau}"

    # (8) each node entered at most once (via a travel arc; waiting excluded)
    heads = {a[1] for a in te.arcs}
    for v in heads:
        if v == O:
            continue
        prob += pulp.lpSum(x[a] for a in te.arcs if a[1] == v) <= 1, f"once_{v}"

    if eps is not None:
        prob += f1 >= eps, "eps_cut"
    if fix_f1 is not None:
        prob += f1 == fix_f1, "fix_f1"
    if fix_f2 is not None:
        prob += f2 == fix_f2, "fix_f2"

    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return _extract_waiting(te, x, w)


def _extract_waiting(te, x, w) -> Solution:
    """Walk O..D through active travel/wait arcs; f2 = arrival(D) - t0.

    Records the full timed schedule in meta:
      timeline  : [{node, arrival, departure}]  (departure=None at D)
      dep_times : departure time from each node of the route except D
    so the waiting (= departure - arrival at a node) is fully preserved and the
    journey -- including waits -- can be replayed by the simulator.
    """
    O, D, t0 = te.O, te.D, te.t0
    route, f1 = [O], 0.0
    timeline = []
    cur, tau = O, t0
    arr_here = t0                                 # arrival time at `cur`
    steps = 0
    limit = len(te.time_nodes) + 2
    while cur != D and steps < limit:
        steps += 1
        dep = None
        for a in te.departures.get((cur, tau), []):
            if x[a].value() and x[a].value() > 0.5:
                dep = a
                break
        if dep is not None:
            c, u = te.arcs[dep]
            timeline.append({"node": cur, "arrival": arr_here, "departure": tau})
            f1 += u
            route.append(dep[1])
            arr_here = tau + c                     # arrival at the next node
            cur, tau = dep[1], tau + c
            continue
        if (cur, tau) in te.waits and w[(cur, tau)].value() and \
                w[(cur, tau)].value() > 0.5:
            tau += 1
            continue
        break
    timeline.append({"node": D, "arrival": arr_here, "departure": None})
    dep_times = [e["departure"] for e in timeline[:-1]]
    return Solution(route=route, f1=float(f1), f2=int(arr_here - t0),
                    meta={"t0": t0, "arrival": arr_here, "waiting": True,
                          "timeline": timeline, "dep_times": dep_times})
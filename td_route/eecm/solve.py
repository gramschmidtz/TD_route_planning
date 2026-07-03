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

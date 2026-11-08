"""
Single transformed single-objective ILP over the time-expanded variables.

This is the P_k solver used by both the endpoint computation (Section IV.B) and
the epsilon-constraint Pareto tracer. It builds constraints (3),(4),(7),(8),(9)
explicitly; (5),(6) are already baked into the variable set (see
time_expanded.py).

Objective convention:  f1 = preference (max),  f2 = travel time (min).
"""
from __future__ import annotations

import os
import re
import tempfile

import pulp

from ..decision import Solution
from .time_expanded import TimeExpanded


# --------------------------------------------------------------------------- #
# Solver selection (CBC by default; CPLEX via config exact.solver/cplex_path)
# --------------------------------------------------------------------------- #
_SOLVER_CFG = {"solver": "cbc", "cplex_path": None, "mip_gap": None,
               "threads": None, "integrality": None}
_LAST = {"log": None, "text": ""}     # per-solve CPLEX log (final MIP gap, diagnostics)


def configure_solver(exact_cfg=None):
    """Set the MILP solver from a config `exact` dict: solver = 'cbc' | 'cplex',
    cplex_path = path to cplex.exe (or its folder). Call once at start-up.
    Optional exactness controls (None = solver default, the old behaviour):
      mip_gap     : relative AND absolute MIP gap (0 = prove optimality exactly)
      threads     : solver threads
      integrality : CPLEX integrality tolerance (0 = no fractional x accepted)"""
    exact_cfg = exact_cfg or {}
    _SOLVER_CFG["solver"] = str(exact_cfg.get("solver") or "cbc").lower()
    _SOLVER_CFG["cplex_path"] = exact_cfg.get("cplex_path")
    for k in ("mip_gap", "threads", "integrality", "feasibility"):
        _SOLVER_CFG[k] = exact_cfg.get(k)


def _build_solver(time_limit, msg):
    gap = _SOLVER_CFG.get("mip_gap")
    kw = {}
    if gap is not None:
        kw.update(gapRel=float(gap), gapAbs=float(gap))
    if _SOLVER_CFG.get("threads"):
        kw["threads"] = int(_SOLVER_CFG["threads"])
    if _SOLVER_CFG["solver"] == "cplex":
        path = _SOLVER_CFG["cplex_path"]
        if path and os.path.isdir(path):          # folder given -> append exe
            path = os.path.join(path, "cplex.exe")
        opts = []
        if _SOLVER_CFG.get("integrality") is not None:
            opts.append(f"set mip tolerances integrality {_SOLVER_CFG['integrality']}")
        if _SOLVER_CFG.get("feasibility") is not None:     # primal feasibility (eprhs)
            opts.append(f"set simplex tolerances feasibility {_SOLVER_CFG['feasibility']}")
        # own log file per solve: PuLP deletes "cplex.log", and the log is the
        # only place CPLEX_CMD reports the final MIP gap of a time-limited run
        fd, log = tempfile.mkstemp(prefix="cplex_", suffix=".log")
        os.close(fd)
        _LAST["log"] = log
        return pulp.CPLEX_CMD(path=path, msg=msg, timeLimit=time_limit,
                              options=opts, logPath=log, **kw)
    _LAST["log"] = None
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, **kw)


def _cplex_log_gap():
    """(proven_optimal_in_log, relative MIP gap) parsed from the last CPLEX log."""
    log = _LAST.get("log")
    if not log or not os.path.exists(log):
        return None, None
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    finally:
        try:
            os.remove(log)
        except OSError:
            pass
        _LAST["log"] = None
    _LAST["text"] = txt
    m = re.findall(r"gap = [-+0-9.eE]+, ([-+0-9.eE]+)%", txt)
    gap = float(m[-1]) / 100.0 if m else None
    opt = "Integer optimal solution" in txt or "integer optimal, tolerance" in txt
    if opt and gap is None:
        gap = 0.0
    return opt, gap


def last_solve_tolerance_failure() -> bool:
    """True if the last CPLEX_CMD run found a MIP incumbent but PuLP's follow-up
    fixed-integer LP rejected it (constraint violated beyond the LP tolerance) --
    PuLP then reports 'Infeasible' although the MIP is feasible."""
    t = _LAST.get("text") or ""
    return ("MIP - Integer optimal" in t or "integer feasible" in t) and         "No solution exists" in t


def _finish(prob, te, sol):
    """Decide the return value after solving. Accepts a proven-optimal solution
    OR a time-limited INCUMBENT (a feasible route reaching D that the solver
    found but could not prove optimal). sol.meta['optimal'] records which it is.
    Returns None only when no feasible O..D solution was found.

    NB PuLP's CPLEX_CMD maps CPLEX 107 'time limit, integer feasible' (and
    104/105/109/113) to LpStatusOptimal; only prob.sol_status distinguishes a
    proven optimum (LpSolutionOptimal) from an incumbent (IntegerFeasible)."""
    _LAST["text"] = ""
    log_opt, log_gap = _cplex_log_gap()
    if pulp.LpStatus[prob.status] == "Infeasible":
        return None
    if len(sol.route) < 2 or sol.route[-1] != te.D:
        return None                                # no incumbent found
    opt = (pulp.LpStatus[prob.status] == "Optimal"
           and prob.sol_status == pulp.LpSolutionOptimal)
    sol.meta["optimal"] = opt
    sol.meta["mip_gap"] = 0.0 if (opt and log_gap is None) else log_gap
    return sol


def solve_single(te: TimeExpanded, sense, *, eps=None, fix_f1=None, fix_f2=None,
                 arc_cap=None, tof_cap=None, cost_cap=None, time_limit=None,
                 msg=False):
    """Build and solve one single-objective ILP.

    sense    : "min_f2" or "max_f1" -- which objective to optimise.
    eps      : if given, add  sum u*x >= eps   (the P2 epsilon-cut, eq. 12).
    fix_f1   : if given, add  sum u*x == fix_f1.
    fix_f2   : if given, add  sum c*x == fix_f2.
    arc_cap  : if given, add  sum_{i,j,t} x_ijt <= arc_cap   (a cap on the number
        of maneuvers).
    tof_cap  : if given, add  sum_{i,j,t} c_ijt x_ijt <= tof_cap   (a cap on the
        total FLIGHT time = burn time, excluding waiting).
    cost_cap : if given (needs te.costs, i.e. net.arc_cost), add
        sum cost_ijt x_ijt <= cost_cap -- the space-mission propellant budget
        sum dv_ijt x_ijt < 5642.4 m/s when arc_cost = dV.

    f2 is  sum cost*x  when te.costs is populated (net.arc_cost given, e.g.
    dV -- waiting is then free), else travel time (+ waits in the waiting model).

    Returns a Solution (route + f1 + f2) or None if infeasible.
    """
    if not te.reachable_destination:
        return None
    if getattr(te, "waiting", False):
        return _solve_waiting(te, sense, eps=eps, fix_f1=fix_f1, fix_f2=fix_f2,
                              arc_cap=arc_cap, tof_cap=tof_cap,
                              cost_cap=cost_cap, time_limit=time_limit, msg=msg)

    prob = pulp.LpProblem("Pk", pulp.LpMinimize)
    x = {k: pulp.LpVariable(f"x_{k[0]}_{k[1]}_{k[2]}", cat="Binary")
         for k in te.arcs}

    f1 = pulp.lpSum(te.arcs[k][1] * x[k] for k in te.arcs)   # preference
    if te.costs:                                             # f2 = sum cost*x (dv)
        f2 = pulp.lpSum(te.costs[k] * x[k] for k in te.arcs)
    else:                                                    # f2 = travel time
        f2 = pulp.lpSum(te.arcs[k][0] * x[k] for k in te.arcs)

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
    if arc_cap is not None:                        # sum x_ijt <= arc_cap
        prob += pulp.lpSum(x[k] for k in te.arcs) <= arc_cap, "arc_cap"
    if tof_cap is not None:                        # sum c_ijt x_ijt <= tof_cap
        prob += pulp.lpSum(te.arcs[k][0] * x[k] for k in te.arcs) <= tof_cap, \
            "tof_cap"
    if cost_cap is not None and te.costs:          # sum cost_ijt x_ijt <= cost_cap
        prob += pulp.lpSum(te.costs[k] * x[k] for k in te.arcs) <= cost_cap, \
            "cost_cap"

    prob.solve(_build_solver(time_limit, msg))
    return _finish(prob, te, extract_solution(te, x))


def extract_solution(te: TimeExpanded, x) -> Solution:
    """Rebuild the route and objective values from a solved model."""
    chosen = {k for k in te.arcs if x[k].value() and x[k].value() > 0.5}
    nxt = {i: (i, j, t) for (i, j, t) in chosen}
    route, f1, f2 = [te.O], 0.0, 0.0
    cur = te.O
    while cur in nxt and cur != te.D:
        i, j, t = nxt[cur]
        c, u = te.arcs[(i, j, t)]
        f1 += u
        f2 += te.costs[(i, j, t)] if te.costs else c
        route.append(j)
        cur = j
        if len(route) > len(te.nodes) + 1:        # safety against cycles
            break
    f2 = float(round(f2, 6)) if te.costs else int(f2)
    return Solution(route=route, f1=float(f1), f2=f2, meta={"t0": te.t0})


# -------------------------------------------------------------------------- #
# Waiting model with explicit wait variables w_it and TIME-NODE FLOW
# CONSERVATION (the tight, network-flow formulation):
#     max  sum u_ijt x_ijt
#     min  sum c_ijt x_ijt + sum w_it                 (travel + waiting time)
#     flow balance at every (v, tau), v in V\{1,n}:
#         sum_{(i,t): t+c_ivt=tau} x_ivt + w_{v,tau-1}
#             == sum_{k} x_{v,k,tau} + w_{v,tau}
#     plus (3),(4) endpoints, (5),(6) structural, (8) node-once.
# The equality balance IMPLIES the arrive->leave/wait and wait->leave/wait
# coupling inequalities (so they are dropped) and additionally forbids
# "floating" wait variables, giving a totally-unimodular network-flow LP
# relaxation -> tight, fast under branch-and-cut. Under the balance,
# f2 = sum c x + sum w equals the true elapsed time arrival(D) - t0.
# -------------------------------------------------------------------------- #
def _solve_waiting(te, sense, *, eps, fix_f1, fix_f2, arc_cap=None,
                   tof_cap=None, cost_cap=None, time_limit=None, msg=False):
    prob = pulp.LpProblem("Pk_wait", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}_{a[2]}", cat="Binary")
         for a in te.arcs}
    w = {a: pulp.LpVariable(f"w_{a[0]}_{a[1]}", cat="Binary") for a in te.waits}

    O, D, t0 = te.O, te.D, te.t0

    f1 = pulp.lpSum(te.arcs[a][1] * x[a] for a in te.arcs)
    if te.costs:
        # space-mission f2 = sum cost*x (e.g. dV): waiting burns no propellant,
        # so the wait variables carry NO objective coefficient.
        f2 = pulp.lpSum(te.costs[a] * x[a] for a in te.arcs)
    else:
        f2 = (pulp.lpSum(te.arcs[a][0] * x[a] for a in te.arcs)  # travel time
              + pulp.lpSum(w[a] for a in te.waits))              # + waiting time
    prob += (f2 if sense == "min_f2" else -f1)

    # (4) exactly one arc enters D
    into_D = [a for a in te.arcs if a[1] == D]
    prob += pulp.lpSum(x[a] for a in into_D) == 1, "in_D"

    origin_wait = getattr(te, "origin_wait", False)
    if not origin_wait:
        # (3) exactly one arc leaves O at t0 (origin departs immediately)
        prob += pulp.lpSum(x[a] for a in te.departures.get((O, t0), [])) == 1, "out_O"

    # time-node flow conservation:  inflow(v,tau) == outflow(v,tau)
    #   inflow  = travel arcs arriving at (v,tau) + wait-in  w_{v,tau-1}
    #             (+ a unit SOURCE injected at (O,t0) when origin_wait)
    #   outflow = travel arcs leaving  (v,tau)    + wait-out w_{v,tau}
    # With origin_wait the origin joins the balance as a source of 1 at t0, so
    # the unit may ride origin wait arcs (counted in f2) until its lone
    # departure -- realising sum_{i,t} x_{1it} = 1 at a FREE departure time.
    for (v, tau) in te.time_nodes:
        if v == D:
            continue
        if v == O and not origin_wait:
            continue
        inflow = [x[a] for a in te.arrivals.get((v, tau), [])]
        if (v, tau - 1) in te.waits:
            inflow.append(w[(v, tau - 1)])
        outflow = [x[a] for a in te.departures.get((v, tau), [])]
        if (v, tau) in te.waits:
            outflow.append(w[(v, tau)])
        src = 1 if (origin_wait and v == O and tau == t0) else 0
        if inflow or outflow or src:
            prob += pulp.lpSum(inflow) + src == pulp.lpSum(outflow), \
                f"bal_{v}_{tau}"

    # (8) each node entered at most once (via a travel arc; waiting excluded)
    for v in {a[1] for a in te.arcs}:
        if v == O:
            continue
        prob += pulp.lpSum(x[a] for a in te.arcs if a[1] == v) <= 1, f"once_{v}"

    if eps is not None:
        prob += f1 >= eps, "eps_cut"
    if fix_f1 is not None:
        prob += f1 == fix_f1, "fix_f1"
    if fix_f2 is not None:
        prob += f2 == fix_f2, "fix_f2"
    if arc_cap is not None:                        # sum x_ijt <= arc_cap
        prob += pulp.lpSum(x[a] for a in te.arcs) <= arc_cap, "arc_cap"
    if tof_cap is not None:                        # sum c_ijt x_ijt <= tof_cap
        prob += pulp.lpSum(te.arcs[a][0] * x[a] for a in te.arcs) <= tof_cap, \
            "tof_cap"
    if cost_cap is not None and te.costs:          # sum cost_ijt x_ijt <= cost_cap
        prob += pulp.lpSum(te.costs[a] * x[a] for a in te.arcs) <= cost_cap, \
            "cost_cap"

    prob.solve(_build_solver(time_limit, msg))
    return _finish(prob, te, _extract_waiting(te, x, w))


def _extract_waiting(te, x, w) -> Solution:
    """Walk O..D through active travel/wait arcs.

    f2 = arrival(D) - t0 (elapsed time), or sum of arc costs (e.g. total dV)
    when the cost mode is active (te.costs populated).  Records the full timed
    schedule in meta:
      timeline  : [{node, arrival, departure}]  (departure=None at D)
      dep_times : departure time from each node of the route except D
    so the waiting (= departure - arrival at a node) is fully preserved and the
    journey -- including waits -- can be replayed by the simulator.
    """
    O, D, t0 = te.O, te.D, te.t0
    route, f1, cost = [O], 0.0, 0.0
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
            if te.costs:
                cost += te.costs[dep]
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
    f2 = float(round(cost, 6)) if te.costs else int(arr_here - t0)
    return Solution(route=route, f1=float(f1), f2=f2,
                    meta={"t0": t0, "arrival": arr_here, "waiting": True,
                          "elapsed": arr_here - t0,
                          "timeline": timeline, "dep_times": dep_times})
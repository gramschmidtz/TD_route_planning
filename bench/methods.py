"""
Methods M0-M5 of the comparison (core code is imported, never modified).

Each method returns {"points": [pt(dv, vs, ...)], "wall_s", "n_solves" | "n_evals",
"peak_mb", "timeout", ...}. Points are the method's reported non-dominated set.
"""
from __future__ import annotations

import os
import time

import numpy as np

from adr_data.adapter import ADRNetwork, dv_max_mps
from adr_data.pareto_exact import ExactFront
from adr_data.run_pareto import raw_dv_of, setup_solver
from td_route.eecm import TimeExpanded, solve_single

from bench.common import WORK, K, PeakMem, nondominated, pt


def _vs(net, s):
    """sum v' along the extracted route (independent of the objective used)."""
    d = net.describe_solution(s)
    return sum(int(round(K * v["score"])) for v in d["visits"])


# ------------------------------------------------------------------ M0
def _exact_run(cls, sd, sc, inst, tag):
    """Resumable: the engine checkpoint (outside OneDrive) is continued after a restart;
    wall time = the engine's cumulative step time t_total (includes model builds), so an
    interrupted + resumed run reports the same kind of time as an uninterrupted one."""
    setup_solver(sc)
    out = os.path.join(WORK, inst, tag)
    with PeakMem() as pm:
        ef = cls(sd, sc, method="score_exact", out_dir=out, log=lambda *a: None)
        st = ef.run()
    pts = [pt(p["total_dv_mps"], p["total_score_scaled"], optimal=p["optimal"],
              mip_gap=p.get("mip_gap")) for p in st["points"]]
    return {"points": nondominated(pts), "wall_s": st["t_total"], "n_solves": len(st["calls"]),
            "peak_mb": pm.peak, "all_optimal": all(p["optimal"] for p in st["points"]),
            "timeout": any(c["optimal"] is False for c in st["calls"]),
            "not_optimal_points": [{"v": p["total_score_scaled"], "dv": p["total_dv_mps"],
                                    "stage_optimal": p.get("stage_optimal"),
                                    "stage_mip_gap": p.get("stage_mip_gap"),
                                    "unproven_interval": p.get("unproven_interval")}
                                   for p in st["points"] if not p["optimal"]]}


def m0(sd, sc, inst):
    return _exact_run(ExactFront, sd, sc, inst, "M0")


# ------------------------------------------------------------------ M1
def m1(sd, sc, n_lambda):
    """Weighted sum  max  lam * sum v'/V* - (1-lam) * sum dV / dV_max  (no lexicographic
    tie-break), lam on a uniform grid of n_lambda values. V* = max sum v' (lam = 1 solve,
    counted). The arc preference of one TEG is overwritten per lam."""
    setup_solver(sc)
    dvmax = dv_max_mps(sc)
    calls, pts, timeout = 0, [], False
    with PeakMem() as pm:
        t0 = time.time()
        net = ADRNetwork(sd, sc)
        te = TimeExpanded(net, net.S, net.T, t0=0, waiting=True, origin_wait=True, budget=net.H)
        base = {k: net.preference(*k) for k in te.arcs}
        cost = {k: (te.costs.get(k, 0.0)) for k in te.arcs}
        vstar = None
        for lam in [1.0] + [x for x in np.linspace(0, 1, n_lambda)[::-1] if x < 1.0]:
            if vstar is None:
                w = {k: base[k] for k in te.arcs}
            else:
                w = {k: lam * base[k] / vstar - (1 - lam) * cost[k] / dvmax for k in te.arcs}
            for k in te.arcs:
                te.arcs[k] = (te.arcs[k][0], w[k])
            s = solve_single(te, "max_f1", cost_cap=dvmax, time_limit=sc["pareto"]["time_limit"])
            calls += 1
            if s is None:
                continue
            timeout |= not bool(s.meta.get("optimal"))
            vs = _vs(net, s)
            if vstar is None:
                vstar = max(vs, 1)
            pts.append(pt(raw_dv_of(net, s), vs, lam=float(lam)))
        wall = time.time() - t0
    return {"points": nondominated(pts), "raw_points": pts, "wall_s": wall, "n_solves": calls,
            "peak_mb": pm.peak, "timeout": timeout, "n_lambda": n_lambda}


# ------------------------------------------------------------------ M2
def m2(sd, sc, delta):
    """dV-axis epsilon-constraint on a FIXED grid eps = dV_max, dV_max - delta, ..., 0:
    stage 1 max sum v' s.t. sum dV <= eps, stage 2 min sum dV s.t. sum v' >= f* (engine)."""
    setup_solver(sc)
    dvmax = dv_max_mps(sc)
    ef = ExactFront(sd, sc, method="dv_exact", log=lambda *a: None)
    pts, timeout = [], False
    with PeakMem() as pm:
        t0 = time.time()
        grid = list(np.arange(dvmax, -1e-9, -float(delta)))
        if grid[-1] > 1e-9:
            grid.append(0.0)                      # grid end point eps = 0
        for eps in grid:
            r = ef._step_dv(float(eps))
            if r is None:
                continue
            timeout |= not r["optimal"]
            pts.append(pt(raw_dv_of(r["net"], r["sol"]), int(round(r["sol"].f1)), eps=float(eps)))
        wall = time.time() - t0
    uniq = nondominated(pts)
    return {"points": uniq, "wall_s": wall, "n_solves": len(ef.calls), "peak_mb": pm.peak,
            "timeout": timeout, "delta": delta, "n_grid": len(pts),
            "n_duplicate": len(pts) - len({(round(p["dv"], 6), p["vs"]) for p in pts})}


# ------------------------------------------------------------------ M5
class NoWaitNet(ADRNetwork):
    """No waiting at debris: depart on arrival. Launch day is still chosen (origin may
    wait), and the mission may end right after any arrival (sink arc at any step)."""

    def wait_feasible(self, v, t):
        return v == self.S

    def arc_feasible(self, i, j, t):
        if j == self.T and i != self.S:
            return not self.reentered(i, self.day(t))
        return super().arc_feasible(i, j, t)


class NoWaitFront(ExactFront):
    def model(self, cap):
        key = round(float(cap), 9)
        if key not in self._te:
            if len(self._te) > 2:
                self._te.clear()
            net = NoWaitNet(self.data, self.cfg, dv_cap=cap)
            te = TimeExpanded(net, net.S, net.T, t0=0, waiting=True, origin_wait=True,
                              budget=net.H)
            self._te[key] = (net, te)
        return self._te[key]


def m5(sd, sc, inst):
    return _exact_run(NoWaitFront, sd, sc, inst, "M5")

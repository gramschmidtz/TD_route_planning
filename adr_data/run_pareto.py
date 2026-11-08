"""
epsilon-constraint Pareto sweep (total score max, total dV <= eps) for the
FY-1C ADR dataset, using the existing exact MILP (td_route.eecm, unchanged):

    for eps in [0, step, 2*step, ..., dV_max]:
        P(eps):  max sum score   s.t. sum dV <= eps        (solve_single "max_f1")
        [lex]    min sum dV      s.t. sum score >= score*   (solve_single "min_f2")

    python -m adr_data.run_pareto [--config config/adr_fy1c_20260605.yaml]
    python -m adr_data.run_pareto --plot-only        # re-plot from the saved JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from td_route.eecm import TimeExpanded, solve_single
from td_route.eecm.solve import configure_solver

from .adapter import ADRNetwork, dv_max_mps
from .load import load_config, load_dataset


def eps_grid(cfg) -> list:
    dvmax = dv_max_mps(cfg)
    step = float(cfg["pareto"]["eps_step_mps"])
    g = list(np.arange(0.0, dvmax, step)) + [dvmax]
    g += [float(e) for e in cfg["pareto"].get("eps_extra_mps", []) or []]
    return sorted({round(float(e), 3) for e in g if 0 <= e <= dvmax + 1e-9})


def setup_solver(cfg):
    p = cfg["pareto"]
    solver = p.get("solver", "cbc")
    if solver == "cplex" and not (p.get("cplex_path") and os.path.exists(p["cplex_path"])):
        print("CPLEX not found -> falling back to CBC")
        solver = "cbc"
    configure_solver({"solver": solver, "cplex_path": p.get("cplex_path"),
                      "mip_gap": p.get("mip_gap"), "threads": p.get("threads"),
                      "integrality": p.get("integrality"),
                      "feasibility": p.get("feasibility")})
    return solver


def raw_dv_of(net, sol) -> float:
    """P5: total dV recomputed from the raw table along the extracted route."""
    tl = sol.meta["timeline"]
    tot = 0.0
    for m in range(1, len(sol.route) - 1):
        prev, j = sol.route[m - 1], sol.route[m]
        if prev != net.S:
            tot += net.transfer_dv(prev, j, tl[m - 1]["departure"])
    return float(tot)


def solve_capped(te, net, sense, cap, *, eps=None, time_limit=None, tries=4):
    """solve_single with sum dV <= cap; P5: if the extracted plan's raw dV
    exceeds cap (solver tolerances), re-solve with a tighter cap."""
    margin = 0.0
    for _ in range(tries):
        s = solve_single(te, sense, eps=eps, cost_cap=cap - margin, time_limit=time_limit)
        if s is None or raw_dv_of(net, s) <= cap + 1e-9:
            return s
        margin = max(2 * margin, 1e-6 * max(1.0, cap), raw_dv_of(net, s) - cap)
    return s


def solve_eps(data, cfg, eps: float) -> dict:
    p = cfg["pareto"]
    dvmax = dv_max_mps(cfg)
    cap = min(eps, dvmax) if p.get("prune_dv_above_eps", True) else dvmax
    net = ADRNetwork(data, cfg, dv_cap=cap)
    te = TimeExpanded(net, net.S, net.T, t0=0, waiting=True, origin_wait=True,
                      budget=net.H)
    tl = p.get("time_limit")
    t0 = time.time()
    s = solve_capped(te, net, "max_f1", eps, time_limit=tl)
    if s is None:
        raise RuntimeError(f"eps={eps}: infeasible (should never happen: eps=0 "
                           f"admits a single-visit plan)")
    opt = bool(s.meta.get("optimal"))
    if p.get("lexicographic", True) and s.f2 > 1e-9:
        tol = 0.5 if net.score_scale else 1e-6
        s2 = solve_capped(te, net, "min_f2", eps, eps=s.f1 - tol, time_limit=tl)
        if s2 is not None and s2.f2 <= s.f2 + 1e-9:
            s, opt = s2, opt and bool(s2.meta.get("optimal"))
    d = net.describe_solution(s)
    d.update(eps=eps, optimal=opt, solve_s=time.time() - t0,
             n_arcs=len(te.arcs), n_waits=len(te.waits),
             f1=s.f1, f2=s.f2, route=[net.name(v) for v in s.route])
    return d


def mark_pareto(rows):
    for r in rows:
        r["pareto"] = not any(
            (o["total_score"] >= r["total_score"] - 1e-6 and o["total_dv"] <= r["total_dv"] + 1e-6)
            and (o["total_score"] > r["total_score"] + 1e-6 or o["total_dv"] < r["total_dv"] - 1e-6)
            for o in rows)
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["eps_mps", "total_dv_mps", "total_score", "n_visits", "launch_day",
            "sequence", "arrival_days", "departure_days", "arc_dv_mps",
            "arc_scores", "wait_days", "optimal", "pareto", "solve_s"]
    fmt = lambda xs, f: ";".join("" if x is None else f % x for x in xs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            v = r["visits"]
            w.writerow([f"{r['eps']:.1f}", f"{r['total_dv']:.2f}", f"{r['total_score']:.3f}",
                        r["n_visits"], f"{r['launch_day']:g}",
                        "-".join(x["id"] for x in v),
                        fmt([x["arrival_day"] for x in v], "%g"),
                        fmt([x["departure_day"] for x in v], "%g"),
                        fmt([x["dv"] for x in v], "%.2f"),
                        fmt([x["score"] for x in v], "%.3f"),
                        fmt([x["wait_days"] for x in v], "%g"),
                        r["optimal"], r["pareto"], f"{r['solve_s']:.1f}"])
    print(f"saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["output"]
    if not args.plot_only:
        data = load_dataset(cfg)
        solver = setup_solver(cfg)
        grid = eps_grid(cfg)
        print(f"solver={solver}  dV_max={dv_max_mps(cfg):.1f} m/s  "
              f"{len(grid)} epsilons  dt={cfg['mission']['dt_days']} d  "
              f"dV table={data.dv_table}")
        rows = []
        for eps in grid:
            r = solve_eps(data, cfg, eps)
            rows.append(r)
            print(f"eps={eps:7.1f}  dV={r['total_dv']:7.1f}  score={r['total_score']:7.2f}"
                  f"  n={r['n_visits']}  tL={r['launch_day']:g}  {'-'.join(r['route'][1:-1])}"
                  f"  opt={r['optimal']}  {r['solve_s']:.0f}s  arcs={r['n_arcs']}", flush=True)
        mark_pareto(rows)
        write_csv(rows, out["csv"])
        with open(out["json"], "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "rows": rows}, f, indent=1)
        print(f"saved -> {out['json']}")
    from .plots import plot_all
    plot_all(cfg)


if __name__ == "__main__":
    main()

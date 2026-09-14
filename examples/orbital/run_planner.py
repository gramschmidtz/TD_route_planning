"""
Run the exact epsilon-constraint planner on the ORBITAL network and plot the
(dV, ToF) Pareto front.

Settings come from config.yaml `orbital` (+ `orbital.eecm`, and `exact.solver /
cplex_path` for the MILP backend).  The planner:
  * discretises time at eecm.dt_days (1 day) -- ToF is rounded up to steps,
  * DROPS infeasible arcs (dV > budget at that departure time), and
  * caps the maneuver count per the space-mission formulation:  sum x_ijt < 1243.

    python -m examples.orbital.run_planner
    python -m examples.orbital.run_planner --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse

from td_route.config import load_config
from td_route.orbital.planner import solve_orbital_eecm, save_front
from examples.orbital.plot_transfer import build_from_cfg


def plot_front(front, td, out, objective="max_visits", f2_mode="time"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if f2_mode == "dv":                                   # y = the f2 objective
        ys = [s.f2 for s in front]                        # total dV (m/s)
        ylabel = "total $\\Delta V$  (m/s)   [propellant]"
    else:
        ys = [td.route_elapsed_days(s) for s in front]    # elapsed time (days)
        ylabel = "elapsed time  (days)   [ToF + waiting]"
    if objective == "mission_value":
        xs = [td.route_value(s) for s in front]          # total mission value
        xlabel = "total mission value  $\\sum_j v_j(t_{arr})$"
    elif objective == "max_visits":
        xs = [len(s.route) for s in front]               # visited-node count
        xlabel = "visited nodes"
    else:
        xs = [td.route_delta_v(s) for s in front]        # total dV (m/s)
        xlabel = "total $\\Delta V$  (m/s)"

    order = sorted(range(len(front)), key=lambda k: xs[k])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([xs[k] for k in order], [ys[k] for k in order], "-o",
            color="tab:purple", lw=1.4, zorder=3)
    for s, x, y in zip(front, xs, ys):
        ax.annotate("→".join(map(str, s.route)), (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=7,
                    color="0.25")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Exact Pareto front — orbital network  "
                 f"O={td.G.graph.get('origin')} → D={td.G.graph.get('destination')}"
                 f"   ({objective} vs {f2_mode})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    onet = build_from_cfg(cfg)
    e = cfg.get("orbital", {}).get("eecm", {})
    objective = e.get("objective", "max_visits")
    f2_mode = e.get("f2", "time")

    mission = None
    if objective == "mission_value":
        from td_route.orbital.mission import from_config as mission_from_config
        mission = mission_from_config(cfg, onet)

    front, td = solve_orbital_eecm(
        onet,
        dt_days=e.get("dt_days", 1.0),
        t_end_days=e.get("t_end_days"),
        waiting=e.get("waiting", True),
        origin_wait=e.get("origin_wait", True),
        objective=objective,
        f2=f2_mode,
        mission=mission,
        enforce_budget=e.get("enforce_budget", True),
        tof_budget_days=e.get("tof_budget_days"),
        dv_budget=e.get("dv_budget"),
        delta=e.get("delta"),
        time_limit=e.get("time_limit"),
        exact_cfg=cfg.get("exact", {}),
        verbose=True,
    )
    if not front:
        raise SystemExit("no feasible O->D route within the horizon")
    print(f"\nPareto front ({len(front)} solutions):")
    for s in sorted(front, key=lambda z: z.f2):
        print("  ", td.describe(s), f" optimal={s.meta.get('optimal')}")
    plot_front(front, td, e.get("out", "outputs/orbital/orbital_pareto.png"), objective, f2_mode)
    # persist the front so examples.orbital.animate_solution can replay it
    # WITHOUT re-running this (slow) solve
    save_front(front, td, e.get("solutions", "outputs/orbital/orbital_solutions.json"),
               objective, f2_mode)


if __name__ == "__main__":
    main()

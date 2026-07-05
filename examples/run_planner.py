"""
Command-line driver for the bi-objective time-dependent route planner.

All settings come from config.yaml -- change values THERE, not on the command
line. The only accepted argument is an optional alternative config path.

    python -m examples.run_planner                  # uses ./config.yaml
    python -m examples.run_planner --config my.yaml

Relevant config sections: network, instance, query, nsga2, exact, run.
Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse
import time

from td_route.config import (load_config, build_network, build_od, build_t0,
                             build_nsga2_config)
from td_route.eecm import epsilon_constraint
from td_route.nsga2 import NSGA2
from td_route import metrics, decision


def _print_front(name, front, dt):
    print(f"\n=== {name}: |F|={len(front)}  ({dt:.2f}s) ===")
    print(f"  {'f1 (pref)':>10}{'f2 (time)':>11}   route")
    for s in front:
        route = "-".join(map(str, s.route))
        if len(route) > 60:
            route = route[:57] + "..."
        print(f"  {s.f1:>10.1f}{s.f2:>11d}   {route}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_network(cfg)
    O, D = build_od(cfg, net)
    t0 = build_t0(cfg, net)
    run_c = cfg.get("run", {})
    ex_c = cfg.get("exact", {})
    solver = run_c.get("solver", "nsga2")

    print(f"network |V|={net.n_nodes} |A|={net.n_arcs} "
          f"pref-density={net.pref_density():.0%}   O={O} D={D}  t0={t0}")

    fronts = {}
    if solver in ("exact", "both"):
        t = time.time()
        fronts["exact"] = epsilon_constraint(
            net, O, D, t0=t0, delta=ex_c.get("delta", 1),
            time_limit=ex_c.get("time_limit"),
            max_iter=ex_c.get("max_iter", 1000),
            waiting=ex_c.get("waiting", False), budget=ex_c.get("budget"),
            verbose=True)
        _print_front("EXACT epsilon-constraint", fronts["exact"],
                     time.time() - t)
    if solver in ("nsga2", "both"):
        t = time.time()
        alg = NSGA2(net, O, D, t0=t0, config=build_nsga2_config(cfg))
        fronts["nsga2"] = alg.run(verbose=True)
        _print_front("NSGA-II", fronts["nsga2"], time.time() - t)

    if solver == "both" and fronts.get("exact") and fronts.get("nsga2"):
        print("\n=== comparison (A = exact reference, R = NSGA-II) ===")
        print(f"  HR  = {metrics.hypervolume_ratio(fronts['nsga2'], fronts['exact']):.3f}")
        print(f"  D   = {metrics.e_dominance(fronts['exact'], fronts['nsga2']):.3f}")

    front = fronts.get("nsga2") or fronts.get("exact")
    if front:
        w1 = run_c.get("w1", 0.5)
        best, _ = decision.fuzzy_select(front, w1=w1)
        print(f"\nfuzzy preferred solution (w1={w1}): "
              f"f1={best.f1:.1f}  f2={best.f2}")
        print(f"  route: {'-'.join(map(str, best.route))}")

    if run_c.get("plot", False) and front:
        _plot(net, O, D, front, run_c.get("plot_out", "planner_result.png"))


def _plot(net, O, D, front, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axn, axf) = plt.subplots(1, 2, figsize=(14, 6))
    pos = {v: (d["x"], d["y"]) for v, d in net.G.nodes(data=True)}
    for i, j, d in net.G.edges(data=True):
        x = [pos[i][0], pos[j][0]]; y = [pos[i][1], pos[j][1]]
        col = "0.8" if d["bp"] == 0 else plt.cm.autumn(1 - d["bp"] / 10)
        axn.plot(x, y, "-", lw=0.4 + d["bp"] / 6, color=col, alpha=0.7)
    if front:
        best = front[len(front) // 2]
        axn.plot([pos[v][0] for v in best.route],
                 [pos[v][1] for v in best.route],
                 "-", color="navy", lw=2.5, label="a route", zorder=5)
    axn.scatter(*pos[O], s=140, marker="*", color="green", zorder=6, label="O")
    axn.scatter(*pos[D], s=140, marker="*", color="red", zorder=6, label="D")
    axn.set_title("Network & one route"); axn.legend(); axn.set_aspect("equal")

    axf.plot([s.f2 for s in front], [s.f1 for s in front], "o-", color="navy")
    axf.set_xlabel("f2  total travel time  (min)")
    axf.set_ylabel("f1  total preference score  (max)")
    axf.set_title(f"Pareto front  (|F|={len(front)})")
    axf.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()

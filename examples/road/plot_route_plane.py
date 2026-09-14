"""
Plot where a given route lands in the (f1, f2) objective plane.

Given an arbitrary node order and departure time (config `simulation.route`,
`simulation.t0`), this evaluates the route's two objectives
    f1 = total preference score   (maximised, y-axis)
    f2 = total travel time        (minimised, x-axis)
and marks it as a point. For context it can overlay (config `plane`):
  * the NSGA-II Pareto front for the same O, D and t0, and
  * a cloud of random feasible routes.

f1/f2 use TDNetwork.evaluate_route (departure-time preference summed, the same
definition the solvers optimise), so the point is directly comparable to the
Pareto front.

Settings come from config.yaml. No numeric CLI arguments.

    python -m examples.road.plot_route_plane
    python -m examples.road.plot_route_plane --config my.yaml
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import networkx as nx

from td_route.config import (load_config, load_dataset_network,
                             build_nsga2_config)
from td_route.decision import Solution, dominates


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _eval(net, route, t0):
    res = net.evaluate_route(route, t0)
    if not res["feasible"]:
        raise SystemExit(f"route {route} is not a connected path in the network")
    return res["preference"], res["travel_time"]


def build_plane_figure(net, route, t0, cfg):
    pl = cfg.get("plane", {})
    O, D = route[0], route[-1]
    f1, f2 = _eval(net, route, t0)

    fig, ax = plt.subplots(figsize=(9, 7))

    # ---- optional cloud of random feasible routes ----
    if pl.get("show_cloud", True):
        from td_route.road.nsga2.imoop import generate_candidate_routes
        import random
        sols = generate_candidate_routes(net, O, D, t0,
                                         n_random=pl.get("cloud_size", 80),
                                         rng=random.Random(0))
        if sols:
            ax.scatter([s.f1 for s in sols], [s.f2 for s in sols], s=18,
                       color="0.7", alpha=0.6, zorder=2,
                       label=f"random feasible routes ({len(sols)})")

    # ---- optional NSGA-II Pareto front for the same O, D, t0 ----
    if pl.get("show_pareto", True):
        try:
            from td_route.road.nsga2 import NSGA2
            front = NSGA2(net, O, D, t0=t0,
                          config=build_nsga2_config(cfg)).run()
            if front:
                front = sorted(front, key=lambda s: s.f1)
                ax.plot([s.f1 for s in front], [s.f2 for s in front], "o-",
                        color="royalblue", ms=6, lw=1.6, zorder=3,
                        label=f"NSGA-II Pareto front ({len(front)})")
        except Exception as e:                       # keep the point even if it fails
            print(f"(Pareto overlay skipped: {e})")
            front = []
    else:
        front = []

    # ---- the queried route ----
    ax.scatter([f1], [f2], s=260, marker="*", color="red", edgecolor="k",
               linewidths=0.8, zorder=5, label="this route")
    dominated = any(dominates(s, Solution(f1=f1, f2=f2)) for s in front)
    tag = "dominated" if dominated else "non-dominated"
    ax.annotate(f"  {'-'.join(map(str, route))}\n  f1={f1:.1f}, f2={f2}  ({tag})",
                (f1, f2), fontsize=9, va="center",
                color="firebrick", zorder=6)

    ax.set_xlabel("$f_1$  total preference score  →  maximise")
    ax.set_ylabel("$f_2$  total travel time  (min)  →  minimise")
    ax.set_title(f"Route in the objective plane   "
                 f"net {cfg.get('network_number','')}   "
                 f"O={O} D={D}  depart {_hms(t0)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig, f1, f2, dominated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    net, num = load_dataset_network(cfg)
    sim = cfg.get("simulation", {})
    t0 = sim.get("t0")
    t0 = net.t_start if t0 is None else int(t0)

    O = net.G.graph.get("origin")
    D = net.G.graph.get("destination")
    route = sim.get("route")
    if not route:
        route = nx.shortest_path(net.G, O, D,
                                 weight=lambda i, j, d: net.travel_time(i, j, t0))
        print(f"route not set -> auto shortest O->D: {route}")

    fig, f1, f2, dominated = build_plane_figure(net, route, t0, cfg)
    print(f"network {num}  route {'-'.join(map(str, route))}  depart {_hms(t0)}")
    print(f"  f1 (preference) = {f1:.2f}   f2 (travel time) = {f2}   "
          f"({'dominated' if dominated else 'non-dominated'} vs NSGA-II front)")

    # Show the figure in a window.
    plt.show()

    # --- To SAVE instead of showing, comment out plt.show() above and uncomment:
    # fig.savefig(cfg.get("plane", {}).get("out", "outputs/road/route_plane.png"), dpi=130)


if __name__ == "__main__":
    main()

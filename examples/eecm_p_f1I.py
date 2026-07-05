"""
Solve ONLY P(f1^I) and show the result.

    P(f1^I):  f1^I = max  sum u_ijt x_ijt   s.t. (3)-(10)

i.e. the maximum total preference score achievable on the selected dataset
network for its origin O, destination D and departure time t0 -- no full Pareto
front, no endpoint sweep, just this one single-objective solve.

Settings come from config.yaml (`network_number`, `query.t0`, `exact.waiting`,
`exact.time_limit`). No numeric CLI arguments.

    python -m examples.eecm_p_f1I
    python -m examples.eecm_p_f1I --config my.yaml
"""
from __future__ import annotations

import argparse
import json
import time

from td_route.config import load_config, load_dataset_network, build_t0
from td_route.eecm import TimeExpanded
from td_route.eecm.solve import solve_single


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _hhmm(t):
    return f"{int(t) // 60:02d}{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    net, num = load_dataset_network(cfg)
    O = net.G.graph.get("origin")
    D = net.G.graph.get("destination")
    t0 = build_t0(cfg, net)
    ex = cfg.get("exact", {})
    waiting = ex.get("waiting", False)

    print(f"network {num}  O={O}  D={D}  depart {_hms(t0)}"
          f"{'   [waiting allowed; f2 = arrival - t0]' if waiting else ''}")
    te = TimeExpanded(net, O, D, t0=t0, waiting=waiting,
                      budget=ex.get("budget"))
    nvars = len(te.arcs) + (len(te.waits) if waiting else 0)
    print(f"time-expanded vars = {nvars}   horizon = [{te.t0}, {te.t_horizon}] "
          f"({te.t_horizon - te.t0} min)\n")

    print("solving P(f1^I): maximise total preference ...")
    t = time.time()
    s = solve_single(te, "max_f1", time_limit=ex.get("time_limit"), msg=True)
    dt = time.time() - t

    if s is None:
        raise SystemExit("infeasible / no solution (D unreachable?)")

    print(f"\n=== P(f1^I) result  ({dt:.1f}s) ===")
    print(f"  f1^I (max total preference) = {s.f1:.0f}")
    print(f"  f2 of this solution         = {s.f2}"
          f"{'  (= arrival - t0, incl. waiting)' if waiting else '  (total travel time)'}")
    if waiting and s.meta.get("arrival") is not None:
        print(f"  arrival at D                = {_hms(s.meta['arrival'])}")
    print(f"  route                       = {'-'.join(map(str, s.route))}")

    # save the solution so simulate_agent.py can replay it (route + dep_times).
    # Written as a one-row list, sol id "f1I", matching the eecm JSON format.
    row = {"sol": "f1I", "eps": None, "f1": s.f1, "f2": s.f2,
           "route": list(s.route), "arrival": s.meta.get("arrival"),
           "solve_seconds": round(dt, 3)}
    if s.meta.get("dep_times") is not None:
        row["dep_times"] = s.meta["dep_times"]
        row["timeline"] = s.meta.get("timeline")
    out = f"eecm_net{num}_depart{_hhmm(t0)}_pf1I.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump([row], f, indent=2)
    print(f"\nsaved solution -> {out}")
    print("  animate it with:  simulation.solution_json: " + out + "\n"
          "                    simulation.solution_id: f1I\n"
          "  then  python -m examples.simulate_agent")


if __name__ == "__main__":
    main()

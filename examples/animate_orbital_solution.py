"""
Build the interactive HTML simulator (and optionally an MP4) from the Pareto
front that examples.run_orbital_planner already solved.

This script does NOT re-run the (slow) EECM solve.  It reads the front saved by
the planner (`orbital.eecm.solutions`), rebuilds the network and the mission
model from config.yaml (both are deterministic), replays each solution's stored
timed schedule, and writes ONE HTML containing every Pareto solution with a
dropdown to switch between them.

    python -m examples.run_orbital_planner        # solve once  -> solutions JSON
    python -m examples.animate_orbital_solution   # replay      -> HTML (+ MP4)

MP4 (`orbital.simulation.mp4`, null to skip) is rendered for the single solution
picked by `orbital.simulation.select`:
  "max_visits" | "max_value" | "min_time" | "min_dv" | <int k> (front index).

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse
import os

from td_route.config import load_config
from td_route.orbital_eecm import load_front
from td_route.orbital_sim import build_trajectory
from td_route.orbital_anim import save_html, save_mp4
from examples.plot_transfer import build_from_cfg


def _pick(sols, select):
    """Pick one solution dict from the loaded front."""
    if isinstance(select, int):
        return sols[max(0, min(select, len(sols) - 1))]
    if select == "min_time":
        return min(sols, key=lambda s: s["f2"])
    if select == "min_dv":
        return min(sols, key=lambda s: s["delta_v"])
    if select == "max_value":
        return max(sols, key=lambda s: (s["value"] or 0.0))
    return max(sols, key=lambda s: s["visits"])          # "max_visits"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    e = cfg.get("orbital", {}).get("eecm", {})
    sim = cfg.get("orbital", {}).get("simulation", {})
    sol_path = e.get("solutions", "orbital_solutions.json")
    if not os.path.isfile(sol_path):
        raise SystemExit(
            f"solutions file not found: {sol_path}\n"
            f"Solve the front first:  python -m examples.run_orbital_planner")

    data = load_front(sol_path)
    meta, sols = data["meta"], data["solutions"]
    dt = meta["dt_days"]
    objective = meta["objective"]
    print(f"loaded {len(sols)} solutions from {sol_path}  "
          f"(objective={objective}, dt={dt} d)")

    # network + mission are deterministic -> rebuilt, never re-solved
    onet = build_from_cfg(cfg)
    mission = None
    if objective == "mission_value":
        from td_route.orbital_mission import from_config as mission_from_config
        mission = mission_from_config(cfg, onet)

    frames = sim.get("frames", 240)

    def to_days(tl):
        return [{"node": x["node"], "arrival": x["arrival"] * dt,
                 "departure": None if x["departure"] is None
                 else x["departure"] * dt} for x in tl]

    def label_of(s):
        val = "" if s["value"] is None else f"val {s['value']:.2f}  "
        return (f"{'→'.join(map(str, s['route']))}   {val}"
                f"t {s['elapsed_days']:.0f}d  dV {s['delta_v']:.0f}")

    trajs, labels = [], []
    for s in sols:                                   # already sorted by time
        if not s["timeline"]:
            raise SystemExit("stored solution has no timeline "
                             "-- re-solve with waiting: true")
        print("  ", label_of(s))
        trajs.append(build_trajectory(onet, s["route"], to_days(s["timeline"]),
                                      n_frames=frames, mission=mission))
        labels.append(label_of(s))

    save_html(trajs, sim.get("html", "orbital_solution.html"),
              title="Orbital Pareto-front solutions", labels=labels)

    if sim.get("mp4"):
        sel = _pick(sols, sim.get("select", "max_visits"))
        print("MP4 of: " + label_of(sel))
        traj = build_trajectory(onet, sel["route"], to_days(sel["timeline"]),
                                n_frames=frames, mission=mission)
        save_mp4(traj, sim.get("mp4", "orbital_solution.mp4"),
                 fps=sim.get("fps", 20))


if __name__ == "__main__":
    main()

"""
Plot the exact Pareto front on a dataset network via the epsilon-constraint
method (Section IV.B) -- endpoints AND the P2(eps_k) iteration, drawn live.

Waiting is controlled by config `exact.waiting`:
  * false -> no waiting (leave each node on arrival); f2 = total travel time.
  * true  -> waiting allowed; f2 = arrival(D) - t0 (includes waiting time).

On the selected dataset network (config `network_number`), for its origin O,
destination D and departure time t0, this first solves the four corner problems

    P(f1^I) -> f1^I   P(f2^I) -> f2^I   P(f1^N) -> f1^N   P(f2^N) -> f2^N

to get the two extreme Pareto points (f1^I, f2^N) / (f1^N, f2^I) and the first
epsilon eps_1 = f1^N + delta (the preference FLOOR of P2; it starts at f1^N+delta
and sweeps UP to f1^I -- it is NOT f1^I - f1^N). Then it iterates

    P2(eps_k): min f2 s.t. sum u*x >= eps_k,   eps_{k+1} = f1(sol_k) + delta

adding each solution to the SAME plot as sol1, sol2, ... Each solution is
appended to  eecm_net<NN>_depart<HHMM>_solutions.json  the moment it is computed
(JSON keeps the route as a list, ready to re-plot). At the end the DOMINATED
solutions are removed and the non-dominated Pareto front is saved.

Responsiveness: the ILP solves run in a background thread while a canvas timer
on the main thread drains results and updates the plot, so the window never
freezes during a long solve.

    python -m examples.road.plot_pareto_front
    python -m examples.road.plot_pareto_front --config my.yaml
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time

import matplotlib.pyplot as plt

from td_route.config import load_config, load_dataset_network, build_t0
from td_route.eecm import TimeExpanded, compute_endpoints
from td_route.eecm.solve import solve_single, configure_solver


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _hhmm(t):
    return f"{int(t) // 60:02d}{int(t) % 60:02d}"


# -------------------------------------------------------------------------- #
# JSON solution log (route kept as a list; no CSV)
# -------------------------------------------------------------------------- #
class SolutionLog:
    def __init__(self, json_path):
        self.json_path = json_path
        self.rows = []

    def _write(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)

    def add(self, sol_id, eps, s, seconds):
        row = {
            "sol": sol_id,
            "eps": (None if eps is None else round(float(eps), 3)),
            "f1": s.f1, "f2": s.f2,
            "route": list(s.route),
            "arrival": s.meta.get("arrival"),
            "solve_seconds": round(seconds, 3),
            # False => f2 is an UPPER BOUND from a time-limited incumbent, not
            # a certified optimum (the Pareto point may lie below it).
            "optimal": s.meta.get("optimal", True),
        }
        # waiting solutions also carry the timed schedule, so the animation can
        # replay the waits (dep_times = departure time from each node except D)
        if s.meta.get("dep_times") is not None:
            row["dep_times"] = s.meta["dep_times"]
            row["timeline"] = s.meta.get("timeline")
        self.rows.append(row)
        self._write()

    def keep_non_dominated(self):
        """Drop dominated solutions (f1 max, f2 min), rewrite the JSON, and
        return (kept_ids, removed_ids)."""
        def dom(a, b):        # a dominates b
            return (a["f1"] >= b["f1"] and a["f2"] <= b["f2"]
                    and (a["f1"] > b["f1"] or a["f2"] < b["f2"]))
        kept, removed = [], []
        for r in self.rows:
            if any(dom(o, r) for o in self.rows if o is not r):
                removed.append(r["sol"])
            else:
                kept.append(r)
        self.rows = kept
        self._write()
        return [r["sol"] for r in kept], removed


# -------------------------------------------------------------------------- #
# Figure
# -------------------------------------------------------------------------- #
def build_base_figure(ep, num, O, D, t0, waiting=False):
    """Endpoint figure: the two extreme Pareto points only."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter([ep.f1_N], [ep.f2_I], s=170, marker="o", facecolor="none",
               edgecolor="tab:blue", linewidths=1.8, zorder=4,
               label="(f1$^N$, f2$^I$)  min-travel-time end")
    ax.scatter([ep.f1_I], [ep.f2_N], s=170, marker="s", facecolor="none",
               edgecolor="tab:red", linewidths=1.8, zorder=4,
               label="(f1$^I$, f2$^N$)  max-preference end")
    mode = "waiting allowed" if waiting else "no waiting"
    ax.set_xlabel("$f_1$  total preference score  →  maximise")
    ax.set_ylabel("$f_2$  total travel time  (min)  →  minimise")
    ax.set_title(f"Exact Pareto front ({mode})   net {num}   "
                 f"O={O} D={D}  depart {_hms(t0)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return fig, ax


def _plot_sol(ax, sol_id, s):
    sc = ax.scatter([s.f1], [s.f2], s=70, color="tab:green", edgecolor="k",
                    linewidths=0.5, zorder=6)
    tx = ax.annotate(sol_id, (s.f1, s.f2), fontsize=8, fontweight="bold",
                     textcoords="offset points", xytext=(5, 4),
                     color="darkgreen", zorder=7)
    return sc, tx


# -------------------------------------------------------------------------- #
# Background solve worker (producer) -> queue -> main-thread timer (consumer)
# -------------------------------------------------------------------------- #
def _worker(q, te, ep, delta, tl, max_iter):
    q.put(("sol1", None, ep.sol_f1N, 0.0))          # min-time endpoint
    eps = ep.first_eps
    k = 1
    while k < max_iter:
        t = time.time()
        s = solve_single(te, "min_f2", eps=eps, time_limit=tl)
        dt = time.time() - t
        if s is None:
            q.put(("__stop__", eps, None, dt))
            break
        k += 1
        q.put((f"sol{k}", eps, s, dt))
        eps = s.f1 + delta
    q.put(None)                                     # sentinel: finished


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
    configure_solver(ex)
    delta = ex.get("delta", 1)
    tl = ex.get("time_limit")
    max_iter = ex.get("max_iter", 1000)

    waiting = ex.get("waiting", False)
    print(f"network {num}  O={O}  D={D}  depart {_hms(t0)}"
          f"{'   [waiting allowed; f2 = arrival - t0]' if waiting else ''}")
    te = TimeExpanded(net, O, D, t0=t0, waiting=waiting,
                      budget=ex.get("budget"))
    print(f"time-expanded vars = {len(te.arcs)}   budget = {te.budget} min\n")

    ep = compute_endpoints(te, delta=delta, time_limit=tl, verbose=True)
    if ep is None:
        raise SystemExit("could not compute endpoints (D unreachable / solve failed)")
    print(f"\nepsilon interval [f1^N, f1^I] = [{ep.f1_N:.0f}, {ep.f1_I:.0f}]  "
          f"(width {ep.f1_I - ep.f1_N:.0f})")
    print(f"FIRST EPSILON eps_1 = f1^N + delta = {ep.f1_N:.0f} + {delta} "
          f"= {ep.first_eps:.0f}\n")

    fig, ax = build_base_figure(ep, num, O, D, t0, waiting=waiting)
    tag = "wait" if waiting else "nowait"
    log = SolutionLog(f"outputs/road/pareto_net{num}_depart{_hhmm(t0)}_{tag}.json")
    artists = {}                                    # sol_id -> (scatter, text)

    print(f"{'sol':>6} {'eps':>6} {'f1':>6} {'f2':>6} {'time(s)':>9}")

    q = queue.Queue()
    worker = threading.Thread(target=_worker,
                              args=(q, te, ep, delta, tl, max_iter), daemon=True)
    state = {"done": False}

    def finalize():
        kept, removed = log.keep_non_dominated()
        for sid in removed:                         # grey out dominated points
            sc, tx = artists.get(sid, (None, None))
            if sc is not None:
                sc.set_color("0.75"); tx.set_color("0.6")
        fig.canvas.draw_idle()
        print(f"\nremoved {len(removed)} dominated solution(s): "
              f"{removed if removed else '(none)'}")
        print(f"{len(kept)} non-dominated solutions saved -> {log.json_path}")

    def drain(*_):
        if state["done"]:
            return
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:                        # finished
                state["done"] = True
                finalize()
                return
            sid, eps, s, dt = item
            if sid == "__stop__":
                print(f"{'--':>6} {eps:>6.0f} {'--':>6} {'--':>6} {dt:>9.2f}   "
                      f"infeasible (eps > f1^I) -> stop")
                continue
            tag = "(endpt)" if eps is None else f"{dt:.2f}"
            # mark time-limited incumbents (f2 not proven optimal) with a '*'
            mark = "" if s.meta.get("optimal", True) else "  * time-limited"
            print(f"{sid:>6} {'--' if eps is None else f'{eps:.0f}':>6} "
                  f"{s.f1:>6.0f} {s.f2:>6d} {tag:>9}{mark}")
            artists[sid] = _plot_sol(ax, sid, s)
            log.add(sid, eps, s, dt)
        fig.canvas.draw_idle()

    worker.start()
    timer = fig.canvas.new_timer(interval=150)
    timer.add_callback(drain)
    timer.start()
    plt.show()

    # --- To SAVE the figure, uncomment:
    # fig.savefig(f"outputs/road/eecm_net{num}_depart{_hhmm(t0)}_front.png", dpi=130)


if __name__ == "__main__":
    main()

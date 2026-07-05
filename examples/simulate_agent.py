"""
Animate an agent travelling a given node route on the selected dataset network.

Layout
------
LEFT : the network. Every arc's WIDTH and COLOUR reflect its current
       time-dependent preference u_ij(t) (with the piecewise schedule applied),
       so the whole network pulses as the clock advances. The agent (blue dot)
       moves O..D, its travelled path in navy, the current arc highlighted.
RIGHT (2x2 + a text strip):
  top-left  : velocity profile v(t) of the CURRENT arc over 07:00-21:00, with a
              red dashed vertical bar at the current clock time.
  top-right : preference profile u(t) of the CURRENT arc over 07:00-21:00, same
              red bar. Both switch when the agent moves onto the next arc.
  bot-left  : instantaneous speed drawn in real time, x-axis = departure .. arrival.
  bot-right : preference score obtained so far, drawn in real time, same x-axis.
              Node-arrival instants are black dashed bars, each segment labelled
              "i->j".
  strip     : elapsed travel time and total preference score, updating live.

Settings come from config.yaml (`network_number`, `simulation`). Change only
`network_number` to animate a different network (route auto-defaults to the
shortest O->D path); set `simulation.route` to force a specific node order and
`simulation.free_flow_speed` to change the agent's base speed.

    python -m examples.simulate_agent
    python -m examples.simulate_agent --config my.yaml
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

import networkx as nx

from td_route.config import load_config, load_dataset_network
from td_route.simulate import build_journey, sample_journey


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _vel_profile(net, cat):
    v0 = net.free_flow_speed
    sp = net.speed_categories[cat]
    xs, ys = [], []
    for k, (a, b) in enumerate(net.horizons):
        xs += [a, b]
        ys += [v0 * sp[k], v0 * sp[k]]
    return xs, ys


def _pref_profile(net, i, j):
    xs = list(range(net.t_start, net.t_end + 1))
    ys = [net.preference(i, j, t) for t in xs]
    return xs, ys


def _edge_style(u, umax):
    """Width and colour for an arc from its CURRENT preference u_ij(t)."""
    if u <= 0:
        return "0.82", 0.5
    frac = min(1.0, u / umax)
    return cm.autumn(1 - frac), 0.6 + 3.2 * frac


def build_animation(net, segments, samples, cfg_sim):
    pos = {v: (net.G.nodes[v]["x"], net.G.nodes[v]["y"]) for v in net.G.nodes}
    route = [segments[0]["i"]] + [s["j"] for s in segments]
    O, D = route[0], route[-1]
    t0 = segments[0]["t_dep"]
    t_end = segments[-1]["t_arr"]
    v0 = net.free_flow_speed
    max_speed = v0 * max(max(v) for v in net.speed_categories.values())
    total_pref = segments[-1]["pref_before"] + segments[-1]["u"]
    umax = max([math.ceil(d["bp"] * (1 + d.get("beta", 0.0)))
                for _i, _j, d in net.G.edges(data=True) if d["bp"] > 0] or [1])

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.09,
                          left=0.02, right=0.955, top=0.95, bottom=0.055)
    ax_net = fig.add_subplot(gs[0, 0])
    # four graphs stacked in a single column (4x1) + a thin text strip
    gr = gs[0, 1].subgridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.34],
                              hspace=0.42)
    ax_vel = fig.add_subplot(gr[0, 0])
    ax_prof = fig.add_subplot(gr[1, 0])
    ax_spd = fig.add_subplot(gr[2, 0])
    ax_cum = fig.add_subplot(gr[3, 0])
    ax_txt = fig.add_subplot(gr[4, 0]); ax_txt.axis("off")

    # ---- network on the left: keep a Line2D per arc so we can restyle it ----
    edge_lines = {}
    for i, j, d in net.G.edges(data=True):
        (ln,) = ax_net.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                            "-", color="0.82", lw=0.5, alpha=0.85, zorder=2)
        edge_lines[(i, j)] = ln
    ax_net.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
                   s=14, color="0.3", zorder=3)
    fs = max(5, min(9, int(360 / max(1, net.n_nodes) ** 0.5)))
    for v, (px, py) in pos.items():
        ax_net.annotate(str(v), (px, py), textcoords="offset points",
                        xytext=(3, 3), fontsize=fs, color="0.15", zorder=4)
    ax_net.scatter(*pos[O], s=180, marker="*", color="green", edgecolor="k",
                   linewidths=0.5, zorder=6, label=f"O={O}")
    ax_net.scatter(*pos[D], s=180, marker="*", color="red", edgecolor="k",
                   linewidths=0.5, zorder=6, label=f"D={D}")
    (cur_edge_ln,) = ax_net.plot([], [], "-", color="royalblue", lw=4.0,
                                 alpha=0.6, zorder=5)
    (travel_ln,) = ax_net.plot([], [], "-", color="navy", lw=2.0, zorder=7)
    (agent_pt,) = ax_net.plot([], [], "o", color="royalblue", ms=13,
                              markeredgecolor="k", zorder=8)
    ax_net.legend(loc="upper right", fontsize=9)
    ax_net.set_aspect("equal", adjustable="datalim")
    ax_net.set_xticks([]); ax_net.set_yticks([])
    ax_net.margins(0.03)
    ax_net.set_title(f"Agent on network {cfg_sim.get('_num', '')}   "
                     f"route {'-'.join(map(str, route))}", fontsize=11)
    wait_banner = ax_net.text(0.5, 0.98, "", transform=ax_net.transAxes,
                              ha="center", va="top", fontsize=13,
                              color="red", fontweight="bold", zorder=9)

    day_ticks = [net.t_start] + [b for _a, b in net.horizons]

    def _style_day(ax, ylabel, title, show_x=True):
        for a, _b in net.horizons:
            ax.axvline(a, color="0.9", lw=0.7, zorder=0)
        ax.axvline(net.horizons[-1][1], color="0.9", lw=0.7, zorder=0)
        ax.set_xlim(net.t_start, net.t_end)
        ax.set_xticks(day_ticks)
        if show_x:
            ax.set_xticklabels([_hms(t) for t in day_ticks], fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    jt = [t0] + [s["t_arr"] for s in segments]
    for ax, ylab, ttl, ymax, show_x in [
            (ax_spd, "speed", "speed vs. journey time", max_speed * 1.15, False),
            (ax_cum, "preference so far", "total preference vs. journey time",
             total_pref * 1.15 + 1, True)]:
        ax.set_xlim(t0, t_end); ax.set_ylim(0, ymax)
        for seg in segments:
            ax.axvline(seg["t_arr"], color="k", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(t0, color="k", ls="--", lw=0.8, alpha=0.7)
        for seg in segments:
            xm = (seg["t_dep"] + seg["t_arr"]) / 2
            ax.text(xm, ymax * 0.045, f"{seg['i']}→{seg['j']}", ha="center",
                    va="bottom", fontsize=6.5, color="0.35", rotation=90)
        ax.set_xticks(jt)
        if show_x:
            ax.set_xticklabels([_hms(t) for t in jt], fontsize=6.5, rotation=45)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(ttl, fontsize=9.5, pad=3)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    (spd_ln,) = ax_spd.plot([], [], "-", color="tab:blue", lw=1.8)
    (cum_ln,) = ax_cum.plot([], [], "-", color="tab:red", lw=1.8)

    txt = ax_txt.text(0.5, 0.6, "", ha="center", va="center", fontsize=13,
                      family="monospace")

    state = {"s": -1}

    def _draw_top(s):
        seg = segments[s]
        i, j = seg["i"], seg["j"]
        ax_vel.clear(); ax_prof.clear()
        vx, vy = _vel_profile(net, seg["cat"])
        ax_vel.plot(vx, vy, color="tab:blue", lw=2)
        ax_vel.fill_between(vx, 0, vy, color="tab:blue", alpha=0.12)
        _style_day(ax_vel, "velocity  $v_{ijt}$",
                   f"current arc {i}→{j}  ({seg['cat']})  velocity", show_x=False)
        px, py = _pref_profile(net, i, j)
        ax_prof.step(px, py, where="post", color="tab:red", lw=1.6)
        ax_prof.fill_between(px, 0, py, step="post", color="tab:red", alpha=0.10)
        _style_day(ax_prof, "preference  $u_{ijt}$",
                   f"current arc {i}→{j}  preference", show_x=True)

    def update(f):
        tc = samples["clock"][f]
        s = samples["seg_idx"][f]
        seg = segments[s]

        # restyle every arc by its CURRENT preference u_ij(tc)
        for (i, j), ln in edge_lines.items():
            col, lw = _edge_style(net.preference(i, j, tc), umax)
            ln.set_color(col); ln.set_linewidth(lw)

        if s != state["s"]:
            _draw_top(s)
            state["s"] = s
            state["velbar"] = ax_vel.axvline(tc, color="red", ls="--", lw=1.6)
            state["profbar"] = ax_prof.axvline(tc, color="red", ls="--", lw=1.6)
            cur_edge_ln.set_data([pos[seg["i"]][0], pos[seg["j"]][0]],
                                 [pos[seg["i"]][1], pos[seg["j"]][1]])
        state["velbar"].set_xdata([tc, tc])
        state["profbar"].set_xdata([tc, tc])

        waiting = samples["waiting"][f]
        ax, ay = samples["x"][f], samples["y"][f]
        # while WAITING the agent sits at node i (bigger red dot); else moves
        agent_pt.set_data([ax], [ay])
        agent_pt.set_color("red" if waiting else "royalblue")
        agent_pt.set_markersize(16 if waiting else 13)
        tx = [pos[route[k]][0] for k in range(s + 1)] + [ax]
        ty = [pos[route[k]][1] for k in range(s + 1)] + [ay]
        travel_ln.set_data(tx, ty)

        spd_ln.set_data(samples["clock"][:f + 1], samples["speed"][:f + 1])
        cum_ln.set_data(samples["clock"][:f + 1], samples["cumpref"][:f + 1])

        wait_banner.set_text(f"WAITING at node {seg['i']}" if waiting else "")
        txt.set_text(f"elapsed: {samples['elapsed'][f]:6.1f} min      "
                     f"total preference: {samples['cumpref'][f]:6.2f}"
                     f"   ({_hms(tc)})")
        return ()

    n = len(samples["clock"])
    anim = FuncAnimation(fig, update, frames=n,
                         interval=1000 / cfg_sim.get("fps", 20), blit=False)
    return fig, anim, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    net, num = load_dataset_network(cfg)
    sim = dict(cfg.get("simulation", {}))
    sim["_num"] = num

    # optional agent base-speed override (dataset stores its own value)
    ffs = sim.get("free_flow_speed")
    if ffs is not None:
        net.free_flow_speed = float(ffs)

    t0 = sim.get("t0")
    t0 = net.t_start if t0 is None else int(t0)

    O, D = net.G.graph.get("origin"), net.G.graph.get("destination")

    # -- solution source: a saved EECM solution (with waiting) > explicit route --
    route = sim.get("route")
    dep_times = sim.get("dep_times")
    sol_json = sim.get("solution_json")
    if sol_json:                        # load route + dep_times from a saved sol
        import json
        rows = json.load(open(sol_json, encoding="utf-8"))
        sid = sim.get("solution_id")
        row = next((r for r in rows if r["sol"] == sid), rows[0] if rows else None)
        if row is None:
            raise SystemExit(f"no solution '{sid}' in {sol_json}")
        route = row["route"]
        dep_times = row.get("dep_times")
        t0 = int(dep_times[0]) if dep_times else t0
        print(f"loaded {row['sol']} from {sol_json}: f1={row['f1']} f2={row['f2']}"
              f"{'  (with waiting)' if dep_times else ''}")

    if not route:                       # auto: shortest travel-time path O->D
        route = nx.shortest_path(net.G, O, D,
                                 weight=lambda i, j, d: net.travel_time(i, j, t0))
        print(f"route not set -> auto shortest O->D: {route}")
    elif route[0] != O or route[-1] != D:
        print(f"note: route starts at {route[0]} ends at {route[-1]} "
              f"(network O={O}, D={D})")

    segments = build_journey(net, route, t0, dep_times=dep_times)
    samples = sample_journey(net, segments, sim.get("frame_minutes", 0.5))
    fig, anim, n = build_animation(net, segments, samples, sim)

    print(f"network {num}  route {'-'.join(map(str, route))}  "
          f"depart {_hms(t0)}  arrival {_hms(segments[-1]['t_arr'])}  "
          f"travel {segments[-1]['t_arr'] - t0:.1f} min  "
          f"total preference {samples['cumpref'][-1]:.2f}  base_speed={net.free_flow_speed}")

    # Show the animation in an interactive window first.
    plt.show()

    # --- To SAVE instead of (or after) showing, comment out plt.show() above
    #     and uncomment ONE of the following (needs the matching writer):
    from matplotlib.animation import PillowWriter
    anim.save(sim.get("out", "simulation.gif"), writer=PillowWriter(fps=sim.get("fps", 20)))
    # anim.save("simulation.mp4", writer="ffmpeg", fps=sim.get("fps", 20), dpi=120)


if __name__ == "__main__":
    main()

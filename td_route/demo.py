"""
Demo / sanity check for the time-dependent network layer.

Run:  python -m td_route.demo      (from the parent of td_route/)
  or: python demo.py               (from inside td_route/, after fixing imports)

It:
  1. generates a random network matching the paper's setup,
  2. verifies travel time & preference actually change with departure time,
  3. checks region numbering / adjacency against the paper (A_10 = {6,9,11}),
  4. saves a plot of the network with the region grid overlaid.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .network import HORIZON_NAMES
from .config import load_config, build_network, build_od


def sanity_report(net, O, D):
    print(f"nodes={net.n_nodes}  arcs={net.n_arcs}  "
          f"pref_arcs={net.n_pref_arcs}  density={net.pref_density():.2f}")
    print(f"O={O}  D={D}")

    # time-dependence check on one preference arc
    i, j = next((a, b) for a, b, d in net.G.edges(data=True) if d["bp"] > 0)
    print(f"\nArc ({i}->{j})  length={net.G[i][j]['length']:.2f}  "
          f"bp={net.G[i][j]['bp']}  speed_cat={net.G[i][j]['speed_cat']}  "
          f"beta={net.G[i][j].get('beta', 0):.2f}")
    print(f"  {'horizon':<10}{'depart':>8}{'c_ijt':>8}{'u_ijt':>8}")
    for name, t in zip(HORIZON_NAMES, [480, 720, 1080, 1200]):
        c = net.travel_time(i, j, t)
        u = net.preference(i, j, t)
        print(f"  {name:<10}{t:>8}{c:>8}{u:>8d}")

    # region adjacency check (paper: 4x3 grid -> A_10 = {6,9,11})
    adj = net.region_adjacency()
    print(f"\nregion adjacency A_10 = {sorted(adj.get(10, set()))}  "
          f"(paper expects [6, 9, 11])")

    # full-route evaluation example (shortest hop path if one exists)
    import networkx as nx
    try:
        path = nx.shortest_path(net.G, O, D)
        res = net.evaluate_route(path, t0=net.t_start)
        print(f"\nexample path {path}")
        print(f"  -> travel_time(f2)={res['travel_time']}  "
              f"preference(f1)={res['preference']:.2f}  "
              f"feasible={res['feasible']}")
    except nx.NetworkXNoPath:
        print("\n(no O->D path in this random instance; try another seed)")


def plot_network(net, O, D, path="network.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = {v: (d["x"], d["y"]) for v, d in net.G.nodes(data=True)}

    # arcs: grey for plain, warm colour scaled by bp for preference arcs
    for i, j, d in net.G.edges(data=True):
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        if d["bp"] > 0:
            ax.plot(x, y, "-", lw=0.5 + d["bp"] / 4,
                    color=plt.cm.autumn(1 - d["bp"] / 10), alpha=0.8, zorder=2)
        else:
            ax.plot(x, y, "-", lw=0.4, color="0.8", alpha=0.6, zorder=1)

    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=12, color="0.3", zorder=3)
    ax.scatter(*pos[O], s=120, marker="*", color="green", zorder=5, label="O")
    ax.scatter(*pos[D], s=120, marker="*", color="red", zorder=5, label="D")

    # region grid + numbers
    g = net.region_grid
    for c in range(g["n_cols"] + 1):
        ax.axvline(g["xmin"] + c * g["cw"], color="0.6", lw=0.6, ls="--")
    for r in range(g["n_rows"] + 1):
        ax.axhline(g["ymin"] + r * g["ch"], color="0.6", lw=0.6, ls="--")
    for rid in range(1, g["n_cols"] * g["n_rows"] + 1):
        rr, cc = (rid - 1) // g["n_cols"], (rid - 1) % g["n_cols"]
        cx = g["xmin"] + (cc + 0.5) * g["cw"]
        # row 0 is at the TOP -> map back from ymax
        cy = g["ymax"] - (rr + 0.5) * g["ch"]
        ax.text(cx, cy, str(rid), color="steelblue", fontsize=14,
                ha="center", va="center", alpha=0.5, zorder=4)

    ax.set_title(f"Time-dependent network  |V|={net.n_nodes}  "
                 f"|A|={net.n_arcs}  pref-density={net.pref_density():.0%}")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nsaved plot -> {path}")


if __name__ == "__main__":
    cfg = load_config()
    net = build_network(cfg)
    O, D = build_od(cfg, net)
    sanity_report(net, O, D)
    plot_network(net, O, D, path="network.png")
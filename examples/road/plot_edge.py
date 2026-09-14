"""
Plot the time-dependent travel time c_ijt and preference score u_ijt of one
arc over the whole day.

All settings come from config.yaml -- change values THERE, not on the command
line. Pick the arc with `query.edge: [i, j]` (or leave it null to auto-pick a
preference arc); the output path is `plot_edge.out`.

    python -m examples.road.plot_edge                    # uses ./config.yaml
    python -m examples.road.plot_edge --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config, build_network
from td_route.road.network import HORIZON_NAMES


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _pick_pref_edge(net):
    for a, b, d in net.G.edges(data=True):
        if d.get("bp", 0) > 0:
            return a, b
    return next(iter(net.G.edges()))


def plot_edge(net, i, j, out="outputs/road/edge_profile.png"):
    if not net.G.has_edge(i, j):
        raise SystemExit(f"arc {i}->{j} does not exist in this network")
    d = net.G[i][j]
    bp = d.get("bp", 0)
    beta = d.get("beta", 0.0)

    ts = list(range(net.t_start, net.t_end + 1))          # 1-minute resolution
    c = [net.travel_time(i, j, t) for t in ts]
    u = [net.preference(i, j, t) for t in ts]

    fig, (ax_c, ax_u) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_c.plot(ts, c, color="tab:blue", lw=1.6)
    ax_c.set_ylabel("travel time  $c_{ijt}$  (min)")
    ax_c.set_title(f"Arc {i}→{j}   length={d['length']:.1f}   "
                   f"speed_cat={d['speed_cat']}   bp={bp}   β={beta:.2f}")
    ax_c.grid(alpha=0.3)

    ax_u.step(ts, u, where="post", color="tab:red", lw=1.6)
    ax_u.set_ylabel("preference  $u_{ijt}$")
    ax_u.set_xlabel("departure time of day")
    ax_u.grid(alpha=0.3)
    if bp > 0:
        for lvl, lab in [(bp, "bp"),
                         (bp * (1 - beta), "bp(1-β)"),
                         (bp * (1 + beta), "bp(1+β)")]:
            ax_u.axhline(lvl, color="0.6", ls=":", lw=0.8)
            ax_u.text(net.t_end + 5, lvl, lab, va="center", fontsize=8,
                      color="0.4")

    for ax in (ax_c, ax_u):
        for a, b in net.horizons:
            ax.axvline(a, color="0.85", lw=0.8, zorder=0)
        ax.axvline(net.horizons[-1][1], color="0.85", lw=0.8, zorder=0)
    for name, (a, b) in zip(HORIZON_NAMES, net.horizons):
        ax_c.text((a + b) / 2, ax_c.get_ylim()[1], name, ha="center",
                  va="top", fontsize=8, color="0.5")

    ticks = [net.t_start] + [b for _a, b in net.horizons]
    ax_u.set_xticks(ticks)
    ax_u.set_xticklabels([_hms(t) for t in ticks])

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"arc {i}->{j}: bp={bp}  beta={beta:.2f}  speed_cat={d['speed_cat']}")
    print(f"  c_ijt range [{min(c)}, {max(c)}]   u_ijt range [{min(u)}, {max(u)}]")
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_network(cfg)
    edge = cfg.get("query", {}).get("edge")
    i, j = tuple(edge) if edge else _pick_pref_edge(net)
    out = cfg.get("plot_edge", {}).get("out", "outputs/road/edge_profile.png")
    plot_edge(net, i, j, out=out)


if __name__ == "__main__":
    main()

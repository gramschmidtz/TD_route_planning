"""
Plot the 9 time-dependent mission values v_j(t) (one panel per non-origin node).

Intermediate nodes show a mixture of 2-4 Gaussian value windows; the destination
(node n) shows its single peak at ~1000 days.  All settings come from
config.yaml (`orbital.mission`).

    python -m examples.plot_mission_value
    python -m examples.plot_mission_value --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config
from td_route.orbital_mission import MissionValue
from examples.plot_transfer import build_from_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_from_cfg(cfg)                    # orbital net -> node ids/alts
    O = net.G.graph.get("origin", 1)
    D = net.G.graph.get("destination", 10)
    nodes = [v for v in net.G.nodes() if v != O]     # 9 non-origin nodes

    m = cfg.get("orbital", {}).get("mission", {})
    mv = MissionValue(
        nodes=nodes, destination=D,
        horizon=m.get("horizon_days", 1500.0),
        dest_peak_day=m.get("dest_peak_day", 1000.0),
        dest_sigma=m.get("dest_sigma", 130.0),
        peaks_range=tuple(m.get("peaks_range", (2, 4))),
        seed=m.get("seed", 42),
    )

    ts = np.linspace(0.0, mv.horizon, 900)
    ncol = 3
    nrow = int(np.ceil(len(nodes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow), sharex=True)
    ymax = max(mv.value(j, t) for j in nodes for t in ts) * 1.12

    for ax, j in zip(axes.flat, nodes):
        vs = [mv.value(j, t) for t in ts]
        is_dest = (j == D)
        color = "tab:red" if is_dest else "tab:blue"
        ax.plot(ts, vs, color=color, lw=1.6)
        ax.fill_between(ts, 0, vs, color=color, alpha=0.15)
        for mu in mv.peak_times(j):
            ax.axvline(mu, color="0.6", ls=":", lw=0.8)
            ax.text(mu, ymax * 0.98, f"{mu:.0f}", fontsize=7, color="0.4",
                    ha="center", va="top")
        alt = net.G.nodes[j]["alt"]
        npk = len(mv.peaks[j])
        tag = "  [DESTINATION]" if is_dest else ""
        ax.set_title(f"node {j}   alt {alt:.0f} km   ({npk} peak"
                     f"{'s' if npk > 1 else ''}){tag}", fontsize=9,
                     color=("tab:red" if is_dest else "0.15"))
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.3)

    for ax in axes.flat[len(nodes):]:            # hide any unused panel
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time  (days)")
    for r in range(nrow):
        axes[r, 0].set_ylabel("mission value  $v_j(t)$")

    fig.suptitle("Time-dependent mission value  $v_j(t)$  per node "
                 f"(origin {O} excluded; destination {D} peaks at "
                 f"{mv.dest_peak_day:.0f} d)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = m.get("out", "mission_value.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"nodes: {nodes}   destination={D}")
    for j in nodes:
        pk = ", ".join(f"(A={A:.2f}, mu={mu:.0f}, sig={sig:.0f})"
                       for A, mu, sig in mv.peaks[j])
        print(f"  v_{j}(t): {pk}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

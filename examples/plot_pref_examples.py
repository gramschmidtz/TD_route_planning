"""
Plot example time-dependent preference scores u_ijt as a 3x1 subplot figure.

Three preference arcs from the network are shown, one per subplot, each as a
step function over the day. u_ijt = ceil(bp * m_ij(t)) with the per-arc
multiplier m(t) in {1, 1-beta, 1+beta} and states lasting U[60,240] min. The
dotted lines mark the bp, bp(1-beta), bp(1+beta) levels.

Everything is read from config.yaml (network + instance sections build the
network; optional `query.pref_edges: [[i,j], ...]` picks specific arcs). No
numeric CLI arguments.

    python -m examples.plot_pref_examples
    python -m examples.plot_pref_examples --config my.yaml
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config, build_network
from td_route.network import HORIZON_NAMES


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _pick_pref_edges(net, k=3):
    """Pick up to k preference arcs, preferring variety in (bp, beta)."""
    prefs = [(a, b, d) for a, b, d in net.G.edges(data=True) if d.get("bp", 0) > 0]
    seen, chosen = set(), []
    for a, b, d in prefs:
        key = (d["bp"], round(d.get("beta", 0), 1))
        if key not in seen:
            seen.add(key)
            chosen.append((a, b))
        if len(chosen) == k:
            return chosen
    for a, b, d in prefs:                       # top up if not enough variety
        if (a, b) not in chosen:
            chosen.append((a, b))
        if len(chosen) == k:
            break
    return chosen


def plot_pref_examples(net, edges, out="pref_examples.png"):
    ts = list(range(net.t_start, net.t_end + 1))
    n = len(edges)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, edges):
        d = net.G[i][j]
        bp, beta = d["bp"], d.get("beta", 0.0)
        u = [net.preference(i, j, t) for t in ts]

        ax.step(ts, u, where="post", color="tab:red", lw=1.6)
        ax.fill_between(ts, 0, u, step="post", color="tab:red", alpha=0.10)
        for lvl, lab in [(bp, "bp"),
                         (bp * (1 - beta), "bp(1-β)"),
                         (bp * (1 + beta), "bp(1+β)")]:
            ax.axhline(lvl, color="0.6", ls=":", lw=0.8)
            ax.text(net.t_end + 5, lvl, lab, va="center", fontsize=8, color="0.4")
        for a, _b in net.horizons:
            ax.axvline(a, color="0.9", lw=0.7, zorder=0)
        ax.axvline(net.horizons[-1][1], color="0.9", lw=0.7, zorder=0)

        ax.set_ylim(0, bp * (1 + beta) * 1.2 + 1)
        ax.set_ylabel("$u_{ijt}$")
        ax.set_title(f"arc {i}→{j}   bp={bp}   β={beta:.2f}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    ticks = [net.t_start] + [b for _a, b in net.horizons]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_hms(t) for t in ticks])
    axes[-1].set_xlabel("departure time of day")

    fig.suptitle("Example time-dependent preference scores  "
                 "$u_{ijt}=\\lceil bp\\cdot m_{ij}(t)\\rceil$", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    for i, j in edges:
        d = net.G[i][j]
        print(f"arc {i}->{j}: bp={d['bp']} beta={d.get('beta',0):.2f}")
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_network(cfg)
    spec = cfg.get("query", {}).get("pref_edges")
    edges = [tuple(e) for e in spec] if spec else _pick_pref_edges(net, 3)
    plot_pref_examples(net, edges)


if __name__ == "__main__":
    main()

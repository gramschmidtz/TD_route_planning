"""
Plot the five speed-category profiles v_{c,k} as a 5x1 subplot figure.

Each subplot shows one arc category's speed multiplier over the day as a step
function across the four time periods (Table 1 of Verbeeck et al. 2014, ref
[11]). Values are read from config.yaml (network.speed_categories / horizons),
so editing the YAML changes the figure. No numeric CLI arguments.

    python -m examples.road.plot_speed_profiles
    python -m examples.road.plot_speed_profiles --config my.yaml
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config
from td_route.road.network import SPEED_CATEGORIES, DEFAULT_HORIZONS


def _hms(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _step(horizons, mults):
    xs, ys = [], []
    for (a, b), m in zip(horizons, mults):
        xs += [a, b]
        ys += [m, m]
    return xs, ys


def plot_speed_profiles(speed_categories, horizons, out="outputs/road/speed_profiles.png"):
    cats = list(speed_categories)
    n = len(cats)
    ymax = max(max(v) for v in speed_categories.values()) * 1.15
    ticks = [horizons[0][0]] + [b for _a, b in horizons]

    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        mults = speed_categories[cat]
        xs, ys = _step(horizons, mults)
        ax.plot(xs, ys, color="tab:blue", lw=2)
        ax.fill_between(xs, 0, ys, color="tab:blue", alpha=0.12)
        ax.axhline(1.0, color="0.6", ls=":", lw=0.8)          # baseline v=1
        for a, _b in horizons:
            ax.axvline(a, color="0.85", lw=0.7, zorder=0)
        ax.axvline(horizons[-1][1], color="0.85", lw=0.7, zorder=0)
        # annotate each period's value
        for (a, b), m in zip(horizons, mults):
            ax.text((a + b) / 2, m + 0.05, f"{m:g}", ha="center", va="bottom",
                    fontsize=8, color="tab:blue")
        ax.set_ylim(0, ymax)
        ax.set_ylabel(cat, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title("Speed-category profiles  $v_{c,k}$  "
                      "(multiplier per time period)")
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_hms(t) for t in ticks])
    axes[-1].set_xlabel("time of day")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net_c = cfg.get("network", {})
    speed_categories = net_c.get("speed_categories", SPEED_CATEGORIES)
    horizons = [tuple(h) for h in net_c.get("horizons", DEFAULT_HORIZONS)]
    plot_speed_profiles(speed_categories, horizons)


if __name__ == "__main__":
    main()

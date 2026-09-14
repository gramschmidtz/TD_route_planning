"""
Surveillance mission value over the whole mission, all targets.

For every target, finds each instant it is abeam of the satellite within the
roll limit, evaluates

    V_I = P_I * NIIRS(GSD_GM) * g(eps_sun) * h(t)

keeps at most ONE target per orbital revolution (the highest V), and plots the
per-target pass values and the accumulated mission value.  Every pass is also
written to a CSV.

All settings come from config.yaml (`surveillance` section).

    python -m examples.surveillance.plot_surveillance_value
    python -m examples.surveillance.plot_surveillance_value --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config
from td_route.surveillance.value import from_config, select_per_revolution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    m = from_config(cfg)
    vc = cfg.get("surveillance", {}).get("value", {})
    out = vc.get("out", "outputs/surveillance/surveillance_value.png")
    csv_path = vc.get("passes_csv", "outputs/surveillance/surveillance_passes.csv")

    orb = m.orbit()
    passes = m.find_passes()
    chosen = select_per_revolution(passes)
    epoch = m.epoch_utc

    # --- CSV ------------------------------------------------------------- #
    if passes:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(passes[0].keys()))
            wr.writeheader()
            wr.writerows(passes)

    # --- figure ---------------------------------------------------------- #
    names = [t.name for t in m.targets]
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(13, 2.1 * len(names) + 3),
                             sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    vmax = max([p["P"] * p["niirs"] for p in passes] + [1e-9])

    for k, (tgt, ax) in enumerate(zip(m.targets, axes)):
        ps = [p for p in passes if p["target"] == tgt.name]
        c = colors[k % len(colors)]
        if tgt.event_start is not None or tgt.event_end is not None:
            a = 0.0 if tgt.event_start is None else \
                (tgt.event_start - epoch) / np.timedelta64(1, "D")
            b = m.duration_days if tgt.event_end is None else \
                (tgt.event_end - epoch) / np.timedelta64(1, "D")
            ax.axvspan(a, b, color=c, alpha=0.10, lw=0)
        d = np.array([p["day"] for p in ps])
        pg = np.array([p["P"] * p["niirs"] * p["g"] for p in ps])  # V without h
        V = np.array([p["V"] for p in ps])
        sel = np.array([p["selected"] for p in ps], dtype=bool)
        if ps:
            ax.scatter(d, pg, s=14, facecolors="none", edgecolors="0.6", lw=0.7,
                       label="P·NIIRS·g (ignoring event)")
            pos = V > 0
            ax.scatter(d[pos & ~sel], V[pos & ~sel], s=22, color=c, alpha=0.35,
                       label="V > 0, not chosen")
            ax.scatter(d[sel], V[sel], s=40, color=c, edgecolors="k", lw=0.7,
                       marker="D", label="chosen")
        tot = V[sel].sum() if ps else 0.0
        ax.set_ylabel(f"{tgt.name}\nV", rotation=0, ha="right", va="center")
        ax.set_ylim(-0.1, vmax * 1.12)
        ax.grid(alpha=0.3)
        ax.set_title(f"{tgt.name}  ({tgt.lat_deg:.1f}°N, {tgt.lon_deg:.1f}°E)  "
                     f"P={tgt.priority:g}   visible passes {len(ps)}   "
                     f"chosen {int(sel.sum())}   ΣV={tot:.2f}",
                     fontsize=9, loc="left")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.9)

    ax = axes[-1]
    if chosen:
        dd = [0.0] + [p["day"] for p in chosen] + [m.duration_days]
        cum = np.concatenate([[0.0], np.cumsum([p["V"] for p in chosen])])
        cum = np.append(cum, cum[-1])
        ax.step(dd, cum, where="post", color="k", lw=1.5)
        for k, name in enumerate(names):
            pts = [(p["day"], c) for p, c in
                   zip(chosen, np.cumsum([p["V"] for p in chosen]))
                   if p["target"] == name]
            if pts:
                ax.scatter(*zip(*pts), s=14, color=colors[k % len(colors)],
                           zorder=3, label=name)
        ax.legend(fontsize=8, loc="upper left", ncol=len(names))
    total = sum(p["V"] for p in chosen)
    ax.set_ylabel("Σ V")
    ax.set_xlabel(f"days since epoch  ({str(epoch.astype('datetime64[s]'))}Z)")
    ax.set_title(f"accumulated mission value (one target per revolution)   "
                 f"total = {total:.2f}   from {len(chosen)} images", fontsize=9,
                 loc="left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, m.duration_days)

    fig.suptitle(f"Surveillance mission value   h={m.altitude_km:g} km  "
                 f"i={m.inclination_deg:g}°  LTAN {m.ltan_hours:g} h  "
                 f"RAAN0={orb.raan0_deg:.2f}°   roll ≤ {m.sensor.max_off_nadir_deg:g}°",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # --- console summary ------------------------------------------------- #
    n_rev = orb.revolution(m.duration_days * 86400.0) + 1
    print(f"orbit: RAAN0={orb.raan0_deg:.3f} deg  period={orb.period_s/60:.2f} min"
          f"  RAAN drift={np.degrees(orb.raan_rate)*86400:.4f} deg/day"
          f"  revolutions={n_rev}")
    print(f"{'target':>6} {'P':>5} {'vis':>4} {'day':>4} {'event':>5} {'V>0':>4}"
          f" {'chosen':>6} {'sum V':>7} {'NIIRS rng':>11}")
    for t in m.targets:
        ps = [p for p in passes if p["target"] == t.name]
        q = [p["niirs"] for p in ps]
        print(f"{t.name:>6} {t.priority:5.2f} {len(ps):4d}"
              f" {sum(p['g'] > 0 for p in ps):4d}"
              f" {sum(p['h'] > 0 for p in ps):5d}"
              f" {sum(p['V'] > 0 for p in ps):4d}"
              f" {sum(p['selected'] for p in ps):6d}"
              f" {sum(p['V'] for p in ps if p['selected']):7.2f}"
              f"   {min(q, default=np.nan):4.2f}-{max(q, default=np.nan):4.2f}")
    per_rev = Counter(p["rev"] for p in passes if p["V"] > 0)
    print(f"revolutions with >1 valuable target (conflicts): "
          f"{sum(1 for v in per_rev.values() if v > 1)}")
    print(f"total mission value = {total:.3f} from {len(chosen)} images")
    print(f"saved -> {out}, {csv_path}")


if __name__ == "__main__":
    main()

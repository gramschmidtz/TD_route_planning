"""
Plot the time-dependent Time of Flight (ToF) and Delta V of one orbital
transfer arc i -> j as a function of the DEPARTURE time.

Space-domain twin of examples/plot_edge.py.  Where the road script sweeps the
travel time c_ijt and preference u_ijt over the day, this one sweeps

    ToF(i, j, t)   [days]      (the network's travel time)
    dV(i, j, t)    [m/s]       (preference = -dV)

over elapsed mission days.  The structure comes entirely from J2 nodal
precession: at the RAAN-alignment windows (dOmega = 0) the plane change
vanishes, so dV drops to the pure-altitude floor and ToF is shortest; half a
synodic period away the residual plane change is maximal and both blow up.
Waiting at the origin orbit for a window is what saves propellant.

All settings come from config.yaml (`orbital` section); pick the arc with
`orbital.plot.arc: [i, j]` (null => origin -> destination).

    python -m examples.plot_transfer                 # uses ./config.yaml
    python -m examples.plot_transfer --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config
from td_route.orbital import OrbitalNetwork
from td_route.orbital_instance import build_orbital_network, DEFAULT_ORBITS


# -------------------------------------------------------------------------- #
# Build the orbital network from the config `orbital` section
# -------------------------------------------------------------------------- #
def build_from_cfg(cfg: dict) -> OrbitalNetwork:
    o = cfg.get("orbital", {})
    sc = o.get("spacecraft", {})
    orbits = o.get("orbits")
    if orbits:
        orbits = [(float(a), float(r)) for a, r in orbits]
    return build_orbital_network(
        orbits=orbits or DEFAULT_ORBITS,
        inclination_deg=o.get("inclination_deg", 51.6),
        thrust_N=sc.get("thrust_N", 0.021),
        isp_s=sc.get("isp_s", 2000.0),
        wet_mass_kg=sc.get("wet_mass_kg", 400.0),
        fuel_mass_kg=sc.get("fuel_mass_kg", 100.0),
        ref_mass_kg=sc.get("ref_mass_kg"),
        t_start=o.get("t_start", 0.0),
        t_end=o.get("t_end", 730.0),
        dt_days=o.get("dt_days", 0.5),
        transit_drift=o.get("transit_drift", True),
        origin=o.get("origin", 1),
        destination=o.get("destination", 10),
    )


def _windows(ts, dv, floor, tol=1.12):
    """Departure times where dV reaches within `tol`x the pure-altitude floor
    (the RAAN-alignment windows).  One time per contiguous below-threshold run
    (its minimum), so noise does not spawn duplicates."""
    thr = tol * floor
    wins, run = [], []
    for k in range(len(ts)):
        if dv[k] <= thr:
            run.append(k)
        elif run:
            kmin = min(run, key=lambda m: dv[m])
            wins.append(ts[kmin])
            run = []
    if run:
        wins.append(ts[min(run, key=lambda m: dv[m])])
    return wins


def plot_transfer(net: OrbitalNetwork, i, j, out="transfer_profile.png",
                  horizon=None, dt=None):
    if not net.G.has_edge(i, j):
        raise SystemExit(f"arc {i}->{j} does not exist in this network")

    dt = dt or min(net.dt_days, 0.5)
    syn = net.synodic_period(i, j)
    # show ~2 synodic periods (so >=2 windows are visible), within sensible bounds
    if horizon is None:
        horizon = net.t_end
        if math.isfinite(syn):
            horizon = max(net.t_end, min(2.2 * syn, 4000.0))

    n = int(horizon / dt)
    ts = [net.t_start + k * dt for k in range(n + 1)]
    # transit-drift model (the network's actual cost); infeasible departure
    # times (required plane change exceeds the whole budget) are masked to NaN
    # so the feasible bands around each shifted window stand out.
    sol = [net.transfer(i, j, t, transit_drift=True) for t in ts]
    dv = [s["dv"] if s["feasible"] else float("nan") for s in sol]
    tof = [s["tof"] if s["feasible"] else float("nan") for s in sol]
    dv_feasible = [s["dv"] for s in sol if s["feasible"]]
    tof_feasible = [s["tof"] for s in sol if s["feasible"]]
    # departure-only model (no drift during transfer) for comparison
    dv_dep = [net.delta_v(i, j, t, transit_drift=False) for t in ts]
    draan = [net.rel_raan(i, j, t) for t in ts]           # departure relative RAAN

    floor = abs(net.circ_speed(i) - net.circ_speed(j)) * 1000.0   # m/s
    budget = net.dv_budget
    # ToF that would exhaust the whole budget, for a comparable y-cap
    tof_budget = (net.ref_mass * net.exhaust_speed / net.thrust_N
                  * (1.0 - math.exp(-budget / net.exhaust_speed))) / 86400.0
    wins = _windows(ts, dv, floor)               # transit-drift windows (actual)
    wins_dep = _windows(ts, dv_dep, floor)       # departure-only windows

    fig, (ax_dv, ax_tof, ax_om) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.5]})

    # --- Delta V ---
    ax_dv.plot(ts, dv_dep, color="tab:orange", lw=1.0, ls="--", alpha=0.7,
               label="departure-only (no transit drift)")
    ax_dv.plot(ts, dv, color="tab:red", lw=1.6,
               label="with transit drift (arrival match)")
    ax_dv.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_dv.axhline(floor, color="0.5", ls="--", lw=0.9)
    ax_dv.text(ts[-1], floor, "  altitude-only floor", va="center",
               fontsize=8, color="0.4")
    ax_dv.axhline(budget, color="tab:green", ls=":", lw=1.1)
    ax_dv.text(ts[-1], budget, "  full-fuel budget", va="center",
               fontsize=8, color="tab:green")
    dv_top = max(dv_feasible + dv_dep) if dv_feasible else max(dv_dep)
    ax_dv.set_ylim(0, min(dv_top * 1.05, budget * 1.15))
    ax_dv.set_ylabel("$\\Delta V$  (m/s)")
    ai, aj = net.G.nodes[i]["alt"], net.G.nodes[j]["alt"]
    syn_txt = f"{syn:.0f} d" if math.isfinite(syn) else "inf"
    ax_dv.set_title(
        f"Transfer  {i} -> {j}   "
        f"alt {ai:.0f}->{aj:.0f} km   i={net.inclination_deg:.1f}$^\\circ$   "
        f"synodic={syn_txt}   floor={floor:.0f} m/s   "
        f"(preference $u_{{ijt}} = -\\Delta V$)")
    ax_dv.grid(alpha=0.3)

    # --- ToF ---
    ax_tof.plot(ts, tof, color="tab:blue", lw=1.6)
    tof_top = max(tof_feasible) if tof_feasible else tof_budget
    ax_tof.set_ylim(0, min(tof_top * 1.15, tof_budget * 1.15))
    ax_tof.set_ylabel("Time of Flight  (days)")
    ax_tof.grid(alpha=0.3)

    # --- relative RAAN (the driver) ---
    ax_om.plot(ts, draan, color="tab:purple", lw=1.3)
    ax_om.axhline(0.0, color="0.5", lw=0.8)
    ax_om.set_ylabel("$\\Delta\\Omega$  (deg)")
    ax_om.set_xlabel("departure time  (days from epoch)")
    ax_om.set_ylim(-185, 185)
    ax_om.set_yticks([-180, -90, 0, 90, 180])
    ax_om.grid(alpha=0.3)

    # mark alignment windows across all panels: departure-only (grey, unshifted)
    # vs transit-drift (green, shifted earlier by the transit lead angle)
    for w in wins_dep:
        for ax in (ax_dv, ax_tof, ax_om):
            ax.axvline(w, color="0.6", ls=":", lw=0.9, alpha=0.6, zorder=0)
    for w in wins:
        for ax in (ax_dv, ax_tof, ax_om):
            ax.axvline(w, color="tab:green", ls="-", lw=0.9, alpha=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    n_feas = len(dv_feasible)
    print(f"arc {i}->{j}: alt {ai:.0f}->{aj:.0f} km  synodic={syn_txt}")
    print(f"  feasible at {n_feas}/{len(ts)} departure times")
    if n_feas:
        print(f"  dV  range [{min(dv_feasible):.0f}, {max(dv_feasible):.0f}] m/s "
              f"  (floor {floor:.0f})")
        print(f"  ToF range [{min(tof_feasible):.1f}, {max(tof_feasible):.1f}] days")
    print(f"  windows (transit-drift)  t = {[round(w) for w in wins]} days")
    print(f"  windows (departure-only) t = {[round(w) for w in wins_dep]} days")
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_from_cfg(cfg)
    o = cfg.get("orbital", {})
    plot_c = o.get("plot", {})
    arc = plot_c.get("arc")
    if arc:
        i, j = int(arc[0]), int(arc[1])
    else:
        i = net.G.graph.get("origin", o.get("origin", 1))
        j = net.G.graph.get("destination", o.get("destination", 10))
    out = plot_c.get("out", "transfer_profile.png")
    horizon = plot_c.get("horizon_days")
    plot_transfer(net, i, j, out=out, horizon=horizon)


if __name__ == "__main__":
    main()

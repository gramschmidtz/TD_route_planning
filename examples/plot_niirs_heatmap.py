"""
Heatmap over cross-track ground distance from the target (x, km; 0 = directly
overhead) vs satellite altitude, for ONE surveillance point.

Two modes, chosen in config.yaml (`surveillance.plot.niirs_heatmap`):

  * time_utc / target = null  ->  NIIRS only (time-independent geometry)
        GSD_GM = sqrt(GSD_AT * GSD_CT)  ->  NIIRS = GIQE(GSD_GM)
  * time_utc and target set   ->  mission value at that instant
        V_I = P_I * NIIRS * g(eps_sun(t)) * h(t)
    At a fixed instant and point, g and h do not depend on (x, h), so V is the
    NIIRS map scaled by P*g*h (all zero at night or outside the event window).

Roll-only pointing (along-track look angle 0), so the off-nadir angle is set
entirely by x and h.  Left: the heatmap with contours and off-nadir iso-lines.
Right: the slice at the mission altitude, flat vs spherical Earth.

    python -m examples.plot_niirs_heatmap
    python -m examples.plot_niirs_heatmap --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td_route.config import load_config
from td_route.surveillance import (from_config, gsd, niirs, view_geometry,
                                   solar_elevation_deg, to_datetime64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    m = from_config(cfg)
    p = cfg.get("surveillance", {}).get("plot", {}).get("niirs_heatmap", {})
    x_lo, x_hi = p.get("x_km", [-1500, 1500])
    h_lo, h_hi = p.get("alt_km", [300, 1000])
    nx, ny = int(p.get("nx", 601)), int(p.get("ny", 351))
    h_slice = float(p.get("slice_alt_km", m.altitude_km))
    eta_lines = p.get("off_nadir_isolines_deg", [15, 30, 45, 60])
    clim = p.get("niirs_clim")
    step = float(p.get("contour_step", 0.5))
    out = p.get("out", "niirs_heatmap.png")

    # --- mode: NIIRS only, or V_I at a given instant for a given target -- #
    t_utc, tname = to_datetime64(p.get("time_utc")), p.get("target")
    if (t_utc is None) != (tname is None):
        raise SystemExit("surveillance.plot.niirs_heatmap: set BOTH time_utc "
                         "and target, or leave both null")
    tgt = None
    scale = 1.0
    if tname is not None:
        by_name = {t.name: t for t in m.targets}
        if str(tname) not in by_name:
            raise SystemExit(f"unknown target {tname!r}; "
                             f"choose from {list(by_name)}")
        tgt = by_name[str(tname)]
        eps = float(solar_elevation_deg(tgt.lat_deg, tgt.lon_deg, t_utc))
        g_val = float(m.illumination.g(eps))
        h_val = float(tgt.h(t_utc))
        scale = tgt.priority * g_val * h_val
        stamp = str(t_utc.astype("datetime64[m]")).replace("-", "").replace(":", "")
        root, ext = os.path.splitext(out)
        out = f"{root}_{tgt.name}_{stamp}{ext or '.png'}"
    zlabel = "NIIRS" if tgt is None else "$V_I = P_I\\cdot$NIIRS$\\cdot g\\cdot h$"

    xs = np.linspace(x_lo, x_hi, nx)
    hs = np.linspace(h_lo, h_hi, ny)
    X, Hh = np.meshgrid(xs, hs)
    Q = niirs(X, Hh, m.sensor, m.giqe)
    ETA = np.degrees(view_geometry(X, Hh, m.sensor.earth_model)[0])

    q_lo, q_hi = (clim if clim else (max(0.0, np.nanmin(Q)), np.nanmax(Q)))
    levels = np.arange(np.ceil(q_lo / step) * step, q_hi, step)   # NIIRS levels
    zero = scale == 0.0
    vmin, vmax = (0.0, 1.0) if zero else (q_lo * scale, q_hi * scale)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.4 if tgt is not None else 6),
        gridspec_kw={"width_ratios": [1.6, 1]})

    # --- heatmap --------------------------------------------------------- #
    pc = ax.pcolormesh(X, Hh, Q * scale, cmap="viridis", vmin=vmin, vmax=vmax,
                       shading="auto")
    cb = fig.colorbar(pc, ax=ax, pad=0.01)
    cb.set_label(zlabel)
    if not zero:
        # V contours sit exactly on the NIIRS contours (V = scale * NIIRS)
        cs = ax.contour(X, Hh, Q * scale, levels=levels * scale, colors="k",
                        linewidths=0.7)
        ax.clabel(cs, fmt="%.1f" if tgt is None else "%.2f", fontsize=8)
    else:
        reason = ("night (eps_sun < 0)" if eps < 0 else
                  "outside event window" if h_val == 0 else "g = 0")
        ax.text(0.5, 0.5, f"V = 0\n{reason}", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="w",
                bbox=dict(facecolor="k", alpha=0.5, lw=0))
    ce = ax.contour(X, Hh, ETA, levels=eta_lines, colors="w",
                    linewidths=0.9, linestyles="--")
    ax.clabel(ce, fmt=lambda v: f"η={v:.0f}°", fontsize=8)
    ax.axhline(m.altitude_km, color="tab:red", lw=1.0, ls=":")
    ax.text(x_hi, m.altitude_km, f" h={m.altitude_km:.0f} km", color="tab:red",
            va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("cross-track ground distance from target  x  (km)")
    ax.set_ylabel("satellite altitude  h  (km)")
    lim = ("" if m.sensor.max_off_nadir_deg is None
           else f"   roll ≤ {m.sensor.max_off_nadir_deg:g}°")
    title = (f"NIIRS (GIQE-4)   β={m.sensor.beta_rad:.3g} rad   "
             f"RER={m.giqe.rer:g}  SNR={m.giqe.snr:g}  H={m.giqe.H:g}   "
             f"{m.sensor.earth_model} Earth{lim}")
    if tgt is not None:
        title = (f"{tgt.name} ({tgt.lat_deg:g}°N, {tgt.lon_deg:g}°E)  @ "
                 f"{str(t_utc.astype('datetime64[s]'))}Z   "
                 f"ε_sun={eps:.1f}°  g={g_val:.3f}  h={h_val:.0f}  "
                 f"P={tgt.priority:g}  →  V = {scale:.3f}·NIIRS\n" + title)
    ax.set_title(title, fontsize=10)

    # --- slice at the mission altitude: flat vs spherical --------------- #
    xs_pos = np.linspace(0.0, max(abs(x_lo), abs(x_hi)), 1200)
    for model, style in (("flat", "-"), ("spherical", "--")):
        sen = dataclasses.replace(m.sensor, earth_model=model)
        ax2.plot(xs_pos, niirs(xs_pos, h_slice, sen, m.giqe) * scale, style,
                 label=f"{model} Earth")
    ax2.set_xlabel("|x|  (km)")
    ax2.set_ylabel(zlabel)
    ax2.set_title(f"slice at h = {h_slice:.0f} km", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    pad = 0.05 * (vmax - vmin)
    ax2.set_ylim(vmin, vmax + 6 * pad)
    for e in eta_lines:                          # off-nadir marks (flat x)
        xe = h_slice * np.tan(np.radians(e))
        if xe <= xs_pos[-1]:
            ax2.axvline(xe, color="0.6", lw=0.8, ls=":")
            ax2.text(xe, vmax + 5 * pad, f"{e}°", ha="center", va="top",
                     fontsize=7, color="0.4")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # --- console table at the slice altitude ---------------------------- #
    a, b = m.giqe.coeffs
    print(f"GIQE-4: a={a}, b={b} (RER={m.giqe.rer}), earth={m.sensor.earth_model}")
    if tgt is not None:
        print(f"{tgt.name} ({tgt.lat_deg}N, {tgt.lon_deg}E) at "
              f"{str(t_utc.astype('datetime64[s]'))}Z:  eps_sun={eps:.2f} deg  "
              f"g={g_val:.4f}  h={h_val:.0f}  P={tgt.priority}  "
              f"-> V = {scale:.4f} * NIIRS")
    print(f"slice h = {h_slice:.0f} km")
    print("  eta[deg]   x[km]   GSD_AT[m]  GSD_CT[m]  GSD_GM[m]  NIIRS"
          + ("      V" if tgt is not None else ""))
    for e in [0] + list(eta_lines):
        xe = h_slice * np.tan(np.radians(e))     # flat-Earth x for this eta
        g = gsd(xe, h_slice, m.sensor)
        q = float(niirs(xe, h_slice, m.sensor, m.giqe))
        print(f"  {e:7.1f}  {xe:7.1f}   {float(g['at']):8.3f}  {float(g['ct']):8.3f}"
              f"   {float(g['gm']):8.3f}   {q:5.2f}"
              + (f"  {q * scale:6.3f}" if tgt is not None else ""))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

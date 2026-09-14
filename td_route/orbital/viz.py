"""
Visualisations of the orbital transfer network (space twin of dataset.network_png).

Three views:
  * orbits_3d_png   -- static 3D: Earth + the 10 circular orbits (matplotlib).
  * raan_drift_png  -- 2D: every node's RAAN drifting under J2 vs time; two
                       curves crossing (mod 360) is an alignment window for that
                       arc -- the network's whole time-dependence in one chart.
  * orbits_3d_html  -- interactive 3D with a departure-time slider that drifts
                       the RAANs (Plotly; falls back with a clear message if
                       Plotly is not installed).

Geometry: a circular orbit of radius a, shared inclination inc and RAAN Omega
is  r_ECI(nu) = Rz(Omega) Rx(inc) [a cos nu, a sin nu, 0]^T  (AOP undefined for
a circle, so the argument of latitude is just nu).
"""
from __future__ import annotations

import math

import numpy as np

from .network import OrbitalNetwork, RE


# -------------------------------------------------------------------------- #
# Geometry helpers
# -------------------------------------------------------------------------- #
def _orbit_points(a, inc_deg, raan_deg, n=240):
    """ECI xyz (km) of a circular orbit, sampled over one revolution."""
    inc, raan = math.radians(inc_deg), math.radians(raan_deg)
    nu = np.linspace(0.0, 2 * math.pi, n)
    x, y, z = a * np.cos(nu), a * np.sin(nu), np.zeros_like(nu)
    # Rx(inc)
    y1 = y * math.cos(inc) - z * math.sin(inc)
    z1 = y * math.sin(inc) + z * math.cos(inc)
    x1 = x
    # Rz(raan)
    xr = x1 * math.cos(raan) - y1 * math.sin(raan)
    yr = x1 * math.sin(raan) + y1 * math.cos(raan)
    return xr, yr, z1


def _colors(net: OrbitalNetwork):
    import matplotlib.pyplot as plt
    alts = [net.G.nodes[v]["alt"] for v in net.G.nodes()]
    lo, hi = min(alts), max(alts)
    return {v: plt.cm.viridis((net.G.nodes[v]["alt"] - lo) / (hi - lo or 1.0))
            for v in net.G.nodes()}


# -------------------------------------------------------------------------- #
# Static 3D (matplotlib)
# -------------------------------------------------------------------------- #
def orbits_3d_png(net: OrbitalNetwork, path: str, t: float = 0.0,
                  title: str = "", dpi: int = 130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth
    u = np.linspace(0, 2 * math.pi, 40)
    v = np.linspace(0, math.pi, 20)
    ex = RE * np.outer(np.cos(u), np.sin(v))
    ey = RE * np.outer(np.sin(u), np.sin(v))
    ez = RE * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(ex, ey, ez, color="#3a6ea5", alpha=0.25, linewidth=0,
                    zorder=0)

    cols = _colors(net)
    amax = 0.0
    for vtx in net.G.nodes():
        a = net.sma(vtx)
        amax = max(amax, a)
        raan = net.raan_at(vtx, t)
        xs, ys, zs = _orbit_points(a, net.inclination_deg, raan)
        ax.plot(xs, ys, zs, color=cols[vtx], lw=1.8, alpha=0.9)
        # the mission point's actual POSITION at time t (theta0 + n t): with
        # true-anomaly matching the rendezvous must meet this point, not just
        # its orbit.  Falls back to the ascending node if no phase is defined.
        u = math.radians(net.phase_at(vtx, t)) if "theta0" in net.G.nodes[vtx] \
            else 0.0
        ic, is_ = math.cos(math.radians(net.inclination_deg)), \
            math.sin(math.radians(net.inclination_deg))
        rc, rs = math.cos(math.radians(raan)), math.sin(math.radians(raan))
        px0, py0 = a * math.cos(u), a * math.sin(u)
        py1, pz = py0 * ic, py0 * is_
        px, py = px0 * rc - py1 * rs, px0 * rs + py1 * rc
        ax.scatter([px], [py], [pz], color=cols[vtx], s=55,
                   edgecolor="k", linewidths=0.5, zorder=5, depthshade=False)
        ax.text(px, py, pz, f"  {vtx}", fontsize=10, color="0.1")

    O = net.G.graph.get("origin")
    D = net.G.graph.get("destination")
    lg = []
    if O is not None:
        lg.append(f"origin = {O}")
    if D is not None:
        lg.append(f"destination = {D}")

    R = amax * 1.05
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_zlabel("z (km)")
    ax.set_title(title or (f"Orbital network  |V|={net.n_nodes}  "
                 f"i={net.inclination_deg:.1f} deg   "
                 f"alt {min(net.G.nodes[v]['alt'] for v in net.G.nodes()):.0f}-"
                 f"{max(net.G.nodes[v]['alt'] for v in net.G.nodes()):.0f} km"
                 + (f"   ({', '.join(lg)})" if lg else "")))
    ax.view_init(elev=24, azim=35)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


# -------------------------------------------------------------------------- #
# RAAN drift over time (2D) -- the network's time-dependence at a glance
# -------------------------------------------------------------------------- #
def raan_drift_png(net: OrbitalNetwork, path: str, t_end: float | None = None,
                   dpi: int = 130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_end = t_end if t_end is not None else net.t_end
    ts = np.linspace(net.t_start, t_end, 600)
    cols = _colors(net)

    fig, ax = plt.subplots(figsize=(11, 6))
    for vtx in net.G.nodes():
        raan = (net.G.nodes[vtx]["raan0"] + net.raan_rate(vtx) * ts) % 360.0
        ax.scatter(ts, raan, s=2, color=cols[vtx], label=None)
        ax.plot([], [], color=cols[vtx], lw=3,
                label=f"{vtx}  ({net.G.nodes[vtx]['alt']:.0f} km, "
                      f"{net.raan_rate(vtx):+.2f} deg/d)")
    ax.set_xlabel("time  (days from epoch)")
    ax.set_ylabel("RAAN  $\\Omega(t)$  (deg, mod 360)")
    ax.set_title("J2 RAAN drift of every node — two nodes at equal RAAN is an "
                 "alignment window for that arc")
    ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])
    ax.legend(fontsize=7, ncol=2, loc="upper right", title="node (alt, rate)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


# -------------------------------------------------------------------------- #
# Interactive 3D (Plotly) with a departure-time slider
# -------------------------------------------------------------------------- #
def orbits_3d_html(net: OrbitalNetwork, path: str, frames: int = 24,
                   t_end: float | None = None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed -- interactive HTML skipped.\n"
              "  Install it with:  pip install plotly\n"
              "  (the static orbits_3d_png / raan_drift_png work without it).")
        return False

    t_end = t_end if t_end is not None else net.t_end
    cols = _colors(net)

    def hexcol(vtx):
        r, g, b, _ = cols[vtx]
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    amax = max(net.sma(v) for v in net.G.nodes())

    # Earth sphere
    u = np.linspace(0, 2 * math.pi, 36)
    v = np.linspace(0, math.pi, 18)
    ex = RE * np.outer(np.cos(u), np.sin(v))
    ey = RE * np.outer(np.sin(u), np.sin(v))
    ez = RE * np.outer(np.ones_like(u), np.cos(v))
    earth = go.Surface(x=ex, y=ey, z=ez, showscale=False, opacity=0.35,
                       colorscale=[[0, "#2a5a8a"], [1, "#3a6ea5"]],
                       hoverinfo="skip", name="Earth")

    def traces_at(t):
        tr = []
        for vtx in net.G.nodes():
            xs, ys, zs = _orbit_points(net.sma(vtx), net.inclination_deg,
                                       net.raan_at(vtx, t))
            tr.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", line=dict(color=hexcol(vtx),
                width=4), name=f"node {vtx}",
                hovertext=f"node {vtx}: {net.G.nodes[vtx]['alt']:.0f} km, "
                          f"RAAN {net.raan_at(vtx, t) % 360:.0f} deg",
                hoverinfo="text"))
            tr.append(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="markers+text",
                marker=dict(color=hexcol(vtx), size=4, line=dict(color="black",
                width=1)), text=[str(vtx)], textposition="top center",
                showlegend=False, hoverinfo="skip"))
        return tr

    times = np.linspace(net.t_start, t_end, frames)
    base = traces_at(times[0])
    fig = go.Figure(data=[earth] + base)

    plotly_frames = []
    for t in times:
        plotly_frames.append(go.Frame(
            data=[earth] + traces_at(t), name=f"{t:.0f}"))
    fig.frames = plotly_frames

    steps = [dict(method="animate", label=f"{t:.0f}",
                  args=[[f"{t:.0f}"], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))]) for t in times]
    R = amax * 1.05
    fig.update_layout(
        title=f"Orbital network — {net.n_nodes} circular orbits, "
              f"i={net.inclination_deg:.1f}°, RAAN drifting under J2 "
              f"(slider = departure day)",
        scene=dict(xaxis=dict(range=[-R, R], title="x (km)"),
                   yaxis=dict(range=[-R, R], title="y (km)"),
                   zaxis=dict(range=[-R, R], title="z (km)"),
                   aspectmode="cube"),
        sliders=[dict(active=0, currentvalue=dict(prefix="t = ", suffix=" days"),
                      pad=dict(t=50), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, y=1, x=0.0,
                     xanchor="right", yanchor="top", pad=dict(t=0, r=10),
                     buttons=[dict(label="Play", method="animate",
                              args=[None, dict(frame=dict(duration=250,
                                    redraw=True), fromcurrent=True)])])])
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"saved -> {path}  (interactive; open in a browser)")
    return True

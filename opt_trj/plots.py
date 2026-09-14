"""Figures for the 1 -> 2 primer-vector study."""

import numpy as np
import matplotlib.pyplot as plt

from .constants import DAY, R_E
from .mee import rv_from_mee
from .dynamics import propagate

__all__ = ["plot_raan_drift", "plot_porkchop", "plot_primer",
           "plot_primer_stages", "plot_trajectory", "plot_all"]


def _finish(fig, show=True):
    """Lay out the figure and (unless batching) put it on screen.

    Nothing is written to disk.  ``plot_all`` builds every figure with
    ``show=False`` and calls ``plt.show()`` once at the end, so all windows
    appear together instead of blocking one at a time.
    """
    fig.tight_layout()
    if show:
        plt.show()


# ---------------------------------------------------------------------------
def plot_raan_drift(res, show=True, t_max_days=250.0, n=1200):
    """Relative RAAN vs time -- why the departure epoch is what it is."""
    no, nf = res["node_o"], res["node_f"]
    t = np.linspace(0.0, t_max_days * DAY, n)
    d = np.array([(nf.mean_raan_deg(tt) - no.mean_raan_deg(tt) + 180) % 360 - 180
                  for tt in t])
    dv_per_deg = 2.0 * np.sqrt(res["fm"].mu / no.a) * np.sin(0.5 * np.pi / 180)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(t / DAY, d, lw=1.6, color="C0", label="secular-J2 mean elements")
    ax.axhline(0.0, color="0.5", lw=0.8)
    for w in res["windows"]:
        ax.axvline(w / DAY, color="C3", ls="--", lw=1.2,
                   label=f"secular crossing {w/DAY:.2f} d")
    ax.axvline(res["t_win"] / DAY, color="C1", ls="-.", lw=1.4,
               label=f"propagated crossing {res['t_win']/DAY:.2f} d")
    tr = res["trajectory"]
    ax.axvline(tr.t_imp[0] / DAY, color="C2", lw=1.8,
               label=f"departure {tr.t_imp[0]/DAY:.3f} d")
    ax.set_xlabel("time from epoch [day]")
    ax.set_ylabel(r"$\Omega_2-\Omega_1$  [deg]")
    ax.set_title(f"J2 nodal alignment, points {no.node_id} and {nf.node_id}"
                 f"   ({dv_per_deg:.0f} m/s per degree of residual RAAN)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, show)


def plot_porkchop(res, show=True):
    """dV over the (departure epoch, time of flight) grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    tr = res["trajectory"]
    for ax, key, title in ((axes[0], "scan_coarse", "coarse grid"),
                           (axes[1], "scan_fine", "fine grid")):
        s = res[key]
        Z = np.where(np.isnan(s.DV), np.nan, s.DV).T
        # clamp well above the floor: the feasible corridor is a thin trench
        # inside a field of multi-km/s arcs, so a percentile stretch hides it
        vmin = np.nanmin(Z)
        vmax = min(vmin + 300.0, np.nanpercentile(Z, 60))
        im = ax.pcolormesh(s.t1 / DAY, s.tof / 60.0, Z, shading="nearest",
                           cmap="viridis", vmin=vmin, vmax=max(vmax, vmin + 50))
        fig.colorbar(im, ax=ax, label=r"$\Delta V$ [m/s]")
        ax.plot(tr.t_imp[0] / DAY, (tr.t_imp[-1] - tr.t_imp[0]) / 60.0, "r*",
                ms=13, label="optimum")
        ax.axvline(res["t_win"] / DAY, color="w", ls="--", lw=1.0,
                   label="J2 RAAN window")
        ax.set_xlabel("departure epoch $t_1$ [day]")
        ax.set_ylabel("time of flight [min]")
        ax.set_title(f"two-impulse $\\Delta V$, {title}")
        ax.legend(loc="upper right", fontsize=8)
    _finish(fig, show)


def plot_primer(res, show=True, traj=None, ana=None, subtitle=None):
    """|p(t)|, its rate, and the thrust profile (cf. Figs 2.1-2.8).

    Defaults to the converged trajectory in ``res``.  Pass ``traj``/``ana``
    explicitly to draw any other stage of the necessary-condition loop -- in
    particular the Lambert-seeded two-impulse solution, ``*res["two_impulse"]``,
    which is the direct analogue of the chapter's Figs. 2.5 and 2.7.
    """
    sc = res["sc"]
    if traj is None or ana is None:
        traj, ana = res["trajectory"], res["analysis"]
    tr = traj
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})

    t0 = tr.t_imp[0]
    ax = axes[0]
    for i, seg in enumerate(ana.segments):
        ax.plot((seg.t - t0) / 60.0, seg.pmag, lw=1.8, color=f"C{i}",
                label=f"leg {i+1}")
    ax.axhline(1.0, color="C3", ls="--", lw=1.2, label=r"NC:  $|p|\leq 1$")
    for t in tr.t_imp:
        ax.axvline((t - t0) / 60.0, color="0.6", lw=0.8)
    if ana.p_max > 1.0 + 1e-3:
        # the criterion of Eq. (2.63): an impulse added where |p| > 1 pays
        j = int(np.argmax(ana.segments[ana.seg_p_max].pmag))
        ax.plot((ana.t_p_max - t0) / 60.0, ana.p_max, "kv", ms=9,
                label=f"max$|p|$ = {ana.p_max:.4f} > 1  "
                      r"$\rightarrow$ add an impulse here")
        ax.fill_between(
            (ana.segments[ana.seg_p_max].t - t0) / 60.0, 1.0,
            np.maximum(ana.segments[ana.seg_p_max].pmag, 1.0),
            color="C3", alpha=0.15, lw=0)
        del j
    ax.set_ylabel(r"primer magnitude $|p|$")
    title = (f"{tr.n_imp}-impulse transfer, "
             rf"$\Delta V$ = {tr.total_dv:.3f} m/s"
             f"   (max$|p|$ = {ana.p_max:.6f})")
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, seg in enumerate(ana.segments):
        ax.plot((seg.t - t0) / 60.0, seg.pdot_dot_p * 1e3, lw=1.5, color=f"C{i}")
    ax.axhline(0.0, color="C3", ls="--", lw=1.0)
    ax.set_ylabel(r"$\dot{p}^{\mathsf{T}} p$  [$10^{-3}$/s]")
    ax.grid(alpha=0.3)

    ax = axes[2]
    m = tr.masses
    for i, (t, dv) in enumerate(zip(tr.t_imp, tr.dv_mag)):
        dt = sc.burn_time(dv, m[i])
        g = sc.gamma_max(m[i])
        ax.add_patch(plt.Rectangle(((t - t0 - dt / 2) / 60.0, 0.0),
                                   dt / 60.0, g, color="C3", alpha=0.85))
        ax.annotate(f"{dv:.2f} m/s\n{dt:.1f} s", ((t - t0) / 60.0, g),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)
    ax.set_ylim(0, 2.1 * sc.gamma_max(m[0]))
    ax.set_ylabel(r"$\Gamma$ [m/s$^2$]")
    ax.set_xlabel(f"time from first impulse [min]     "
                  f"($t_1$ = {t0/DAY:.5f} d)")
    ax.grid(alpha=0.3)
    _finish(fig, show)


def plot_trajectory(res, show=True):
    """3D view plus an altitude profile.

    The 3D panel alone is misleading here: the two mission-point orbits differ
    by 65 km on a 7000 km scale, so they overlap.  The altitude panel is what
    actually shows the transfer.
    """
    tr, fm = res["trajectory"], res["fm"]
    no, nf = res["node_o"], res["node_f"]
    t1, t2 = tr.t_imp[0], tr.t_imp[-1]

    def sample(node, t_start, n=400):
        tt = np.linspace(t_start, t_start + node.period, n)
        R = np.array([node.rv(t)[0] for t in tt]).T
        return tt, R

    to_, Ro = sample(no, t1 - no.period)
    tf_, Rf = sample(nf, t2)

    arcs, arc_t = [], []
    for i, leg in enumerate(tr.legs):
        te = np.linspace(leg.t_a, leg.t_b, 400)
        m = tr.masses[i + 1] if tr.masses is not None else 5000.0
        _, X = propagate(leg.x_a, (leg.t_a, leg.t_b), fm, m, t_eval=te)
        arcs.append(np.array([rv_from_mee(X[:, j], fm.mu)[0]
                              for j in range(X.shape[1])]).T)
        arc_t.append(te)

    fig = plt.figure(figsize=(13.5, 6.4))

    # --- 3D ----------------------------------------------------------------
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(R_E*np.cos(u)*np.sin(v)/1e3, R_E*np.sin(u)*np.sin(v)/1e3,
                    R_E*np.cos(v)/1e3, color="0.75", alpha=0.22, linewidth=0)
    ax.plot(*(Ro / 1e3), color="C0", lw=2.2, ls="--",
            label=f"point {no.node_id} ({no.alt/1e3:.0f} km)")
    ax.plot(*(Rf / 1e3), color="C1", lw=2.2, ls=":",
            label=f"point {nf.node_id} ({nf.alt/1e3:.0f} km)")
    for i, A in enumerate(arcs):
        ax.plot(*(A / 1e3), color="C3", lw=1.5,
                label="transfer arc" if i == 0 else None)
    for i in range(tr.n_imp):
        r = arcs[0][:, 0] if i == 0 else (
            arcs[-1][:, -1] if i == tr.n_imp - 1 else arcs[i][:, 0])
        ax.scatter(*(r / 1e3), color="k", s=42, marker="^", zorder=5)
    lim = (R_E + nf.alt) / 1e3 * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.legend(loc="upper left", fontsize=8)

    # --- altitude profile --------------------------------------------------
    ax = fig.add_subplot(1, 2, 2)
    alt = lambda R: (np.linalg.norm(R, axis=0) - R_E) / 1e3
    ax.plot((to_ - t1) / 60.0, alt(Ro), color="C0", lw=1.6, ls="--",
            label=f"point {no.node_id} orbit")
    ax.plot((tf_ - t1) / 60.0, alt(Rf), color="C1", lw=1.6, ls=":",
            label=f"point {nf.node_id} orbit")
    for i, (te, A) in enumerate(zip(arc_t, arcs)):
        ax.plot((te - t1) / 60.0, alt(A), color="C3", lw=2.0,
                label="transfer arc" if i == 0 else None)
    for i, t in enumerate(tr.t_imp):
        r = arcs[0][:, 0] if i == 0 else (
            arcs[-1][:, -1] if i == tr.n_imp - 1 else arcs[i][:, 0])
        h = (np.linalg.norm(r) - R_E) / 1e3
        ax.plot((t - t1) / 60.0, h, "k^", ms=8, zorder=5)
        ax.annotate(rf"$\Delta v_{i+1}$ = {tr.dv_mag[i]:.2f} m/s",
                    ((t - t1) / 60.0, h), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("time from the first impulse [min]")
    ax.set_ylabel("altitude [km]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Mission point {no.node_id} -> {nf.node_id}, "
                 rf"$\Delta V$ = {tr.total_dv:.3f} m/s, "
                 f"departure at t = {t1/DAY:.4f} d", fontsize=12)
    _finish(fig, show)


def plot_primer_stages(res, show=True):
    """One primer figure per stage of the necessary-condition loop.

    The ``n = 2`` figure is the Lambert-seeded two-impulse solution -- the one
    that shows *why* an impulse gets added -- and the last is the converged arc.
    """
    refined = res.get("refined_after_insert", True)
    for traj, ana in res.get("history", []):
        if traj.n_imp == 2:
            sub = ("Lambert-seeded two-impulse solution, terminal times "
                   "converged on Eqs. (2.59)-(2.60)")
        elif refined:
            sub = ("after inserting impulses at max$|p|$ and re-converging "
                   "Eqs. (2.59), (2.75)-(2.76), (2.60)")
        else:
            sub = (r"impulse just inserted at argmax$|p|$ along $p$ "
                   "(Eqs. 2.64-2.69), NOT yet refined")
        plot_primer(res, show=False, traj=traj, ana=ana, subtitle=sub)
    if show:
        plt.show()


def plot_all(res):
    """Build every figure, then put them all on screen at once."""
    plot_raan_drift(res, show=False)
    if res.get("scan_fine") is not None:
        plot_porkchop(res, show=False)
    plot_primer_stages(res, show=False)
    plot_trajectory(res, show=False)
    plt.show()

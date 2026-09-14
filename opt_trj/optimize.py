"""Driver: J2 window location, departure scan, then Lion-Handelsman refinement.

Why a global search is needed before any primer-vector work.  Between mission
points 1 and 2 the *plane* dominates the cost: the two RAANs differ by 18.5 deg
at epoch and only J2 nodal regression (which differs by 0.154 deg/day between a
500 km and a 565 km orbit at i = 51.6 deg) closes the gap, at t = 120.2 d.  An
impulsive plane change costs ~132 m/s per degree, so departing at t = 0 needs
~2.4 km/s -- twice the 1104 m/s budget -- while departing inside the window
costs tens of m/s.  Primer-vector gradients are local; they refine a departure
epoch, they cannot find a window four months away.

Pipeline
--------
1. locate the J2 RAAN crossing analytically (``mission.raan_window``)
2. propagate both mission-point orbits numerically to the window and keep dense
   output  (the secular mean-element model is fine for RAAN but useless for
   *phase*: over ~1800 revolutions the osculating/mean mean-motion difference
   accumulates into a phase error of order a full revolution)
3. two-body Lambert pork-chop over (t1, ToF) on those exact states -> basin
4. re-converge the arc against the full perturbed force model
5. iterate the terminal impulse times on the gradients (2.59)-(2.60)
6. if max|p| > 1, insert and iterate a midcourse impulse (2.63)-(2.76)
7. verify Table 2.1
"""

import numpy as np
from scipy.optimize import minimize

from .constants import DAY
from .mee import rv_from_mee
from .lambert import lambert
from .mission import raan_window
from .twoimpulse import (two_impulse, three_impulse, n_impulse,
                         interior_positions)
from .primer import analyse, terminal_gradients, midcourse_seed

__all__ = ["ScanResult", "scan_departure", "calibrate_window",
           "refine_terminal_times", "refine_n_impulse", "polish_nc",
           "add_impulse", "optimize_transfer"]


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _window_cache_path(node_o, node_f, t_rough):
    import hashlib
    from pathlib import Path
    key = (f"{node_o.node_id}_{node_f.node_id}_{node_o.fm.zonal_degree}"
           f"_{int(node_o.fm.drag is not None)}_{t_rough:.6f}")
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return Path(__file__).with_name("_cache") / f"window_{h}.npy"


def calibrate_window(node_o, node_f, t_rough, t_hi, verbose=True,
                     tol_deg=0.01, maxiter=6):
    """Correct the secular RAAN-crossing epoch using the propagated orbits.

    The closed-form secular rate uses the *osculating* semi-major axis, so each
    node's nodal rate carries an O(J2) relative error of ~1e-3.  On the
    individual rates that is harmless, but the window is set by the tiny
    *difference* of two nearly equal rates (0.154 of 4.7 deg/day), where the
    same absolute error is ~8 %.  Over 120 days that displaces the crossing by
    several days -- more than the whole scan window -- so the crossing has to be
    found from the numerically propagated node lines.

    Both nodes must already be ``prepare``d; the dense window is extended as
    needed.  Returns the corrected crossing epoch.
    """
    ddot = node_f.raan_dot - node_o.raan_dot          # rad/s, secular estimate
    t = float(t_rough)
    for it in range(maxiter):
        need = t + 0.05 * DAY
        for nd in (node_o, node_f):
            if need > nd._t_hi:
                nd.reprepare(min(need + 2.0 * DAY, t_hi + 30 * DAY))
        d = _wrap_deg(node_f.raan_deg(t) - node_o.raan_deg(t))
        if verbose:
            print(f"      window calibration {it}: t = {t/DAY:9.4f} d, "
                  f"exact dRAAN = {d:+8.4f} deg")
        if abs(d) < tol_deg:
            break
        t_new = t - np.radians(d) / ddot
        if t_new < node_o._t_lo:
            raise RuntimeError("corrected window falls before the anchor epoch; "
                               "increase anchor_back_days")
        t = t_new
    return t


class ScanResult:
    """Total dV of the two-body two-impulse arc over a (t1, ToF) grid."""

    def __init__(self, t1, tof, DV, exact):
        self.t1 = t1            # (n1,) [s]
        self.tof = tof          # (n2,) [s]
        self.DV = DV            # (n1, n2) [m/s], NaN where Lambert failed
        self.exact = exact      # endpoint states: osculating (True) or mean

    def best(self, k=1, min_sep=0.0):
        """The ``k`` best ``(t1, ToF, dV)`` triples, >= ``min_sep`` apart in t1."""
        flat = np.where(np.isnan(self.DV), np.inf, self.DV).ravel()
        order = np.argsort(flat)
        out = []
        for idx in order:
            if not np.isfinite(flat[idx]):
                break
            i, j = np.unravel_index(idx, self.DV.shape)
            t1 = float(self.t1[i])
            if any(abs(t1 - c[0]) < min_sep for c in out):
                continue
            out.append((t1, float(self.tof[j]), float(flat[idx])))
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------------------
def scan_departure(node_o, node_f, t1_grid, tof_grid, mu, exact=True,
                   verbose=True):
    """Two-body Lambert pork-chop over departure time and time of flight.

    ``exact=True`` takes the endpoint states from the numerically propagated
    orbits (window must be prepared); ``exact=False`` uses the closed-form
    secular-J2 model, which is only meaningful for the plane geometry.
    """
    state = (lambda nd, t: nd.rv(t)) if exact else (lambda nd, t: nd.mean_rv(t))
    n1, n2 = t1_grid.size, tof_grid.size
    DV = np.full((n1, n2), np.nan)

    for i, t1 in enumerate(t1_grid):
        r1, v1m = state(node_o, t1)
        for j, tof in enumerate(tof_grid):
            r2, v2p = state(node_f, t1 + tof)
            va, vb = lambert(r1, r2, tof, mu)
            if va is None:
                continue
            DV[i, j] = np.linalg.norm(va - v1m) + np.linalg.norm(v2p - vb)
        if verbose and (i % max(1, n1 // 40) == 0):
            best = np.nanmin(DV[:i + 1]) if np.isfinite(DV[:i + 1]).any() else np.nan
            print(f"      scan {100*(i+1)/n1:5.1f}%   best = {best:9.2f} m/s",
                  end="\r", flush=True)
    if verbose:
        print(" " * 72, end="\r")
    return ScanResult(t1_grid, tof_grid, DV, exact)


# ---------------------------------------------------------------------------
def refine_terminal_times(node_o, node_f, fm, sc, t1, t2,
                          bounds_days=None, verbose=True, maxiter=80):
    """Iterate the first/last impulse times on the PVT gradients.

    Minimises ``J(t1, t2) = |dv1| + |dv2|`` using Eqs. (2.59)-(2.60) as the
    analytic gradient -- which costs nothing extra, since the 6x6 primer BVP
    only needs the STM the shooting step already produced.
    """
    cache = {}

    def build(z):
        key = (round(z[0], 12), round(z[1], 12))
        if key not in cache:
            try:
                cache[key] = two_impulse(node_o, node_f, z[0] * DAY,
                                         z[1] * DAY, fm, sc)
            except Exception:                                    # noqa: BLE001
                cache[key] = None
        return cache[key]

    first = [None]

    def fun(z):
        tr = build(z)
        if tr is None:
            return 1e9, np.zeros(2)
        g1, g2, _, _ = terminal_gradients(tr, fm)
        if first[0] is None:
            first[0] = tr.total_dv
        return tr.total_dv, np.array([g1, g2]) * DAY     # per day

    res = minimize(fun, np.array([t1 / DAY, t2 / DAY]), jac=True,
                   method="L-BFGS-B", bounds=bounds_days,
                   options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-14})
    if verbose:
        print(f"      terminal-time refinement: {res.nit} iters, "
              f"dV {first[0]:.4f} -> {res.fun:.4f} m/s")
    return build(res.x), res


# ---------------------------------------------------------------------------
def _pack(t1, mids, tn, scale_r, scale_t, t_ref, r_refs):
    """Decision vector ``[t1, (r_mi, t_mi)..., tn]``, centred and scaled.

    Two separate conditioning fixes live here, both found the hard way:

    * ``scale_t`` is the *transfer duration*, not a day.  This arc lasts
      ~0.03 d, so normalising time by a day left those directions under-scaled
      by ~30 and L-BFGS-B exhausted its iterations with ``dJ/dt_m`` still at
      31 m/s per unit move.
    * everything is measured **relative to the seed** (``t_ref``, ``r_refs``).
      Absolute epochs are ~2100 transfer durations from the mission epoch while
      the meaningful variation is ~0.5, so four digits were pure offset:
      a finite-difference Jacobian and the root finder's internal tests all
      work relative to ``|z|``, and ``root`` gave up immediately ("not making
      good progress") for every step size tried.  Centring drops ``|z|`` to
      O(1) and the same call converges.
    """
    z = [(t1 - t_ref) / scale_t]
    for (r_m, t_m), r0 in zip(mids, r_refs):
        z.extend(list((np.asarray(r_m) - r0) / scale_r) + [(t_m - t_ref) / scale_t])
    z.append((tn - t_ref) / scale_t)
    return np.array(z)


def _unpack(z, scale_r, scale_t, t_ref, r_refs):
    k = (z.size - 2) // 4
    t1, tn = z[0] * scale_t + t_ref, z[-1] * scale_t + t_ref
    mids = [(z[1 + 4 * i:4 + 4 * i] * scale_r + r_refs[i],
             z[4 + 4 * i] * scale_t + t_ref) for i in range(k)]
    return t1, mids, tn


def refine_n_impulse(node_o, node_f, fm, sc, traj, scale_r=None,
                     bounds_days=None, verbose=True, maxiter=400, tag="",
                     gtol_dv=1e-4):
    """Drive every Lion-Handelsman gradient of an n-impulse arc to zero.

    Free variables: the first and last impulse times (i.e. the terminal
    coasts) and the position and time of every interior impulse.  Gradients:
    Eq. (2.59) for ``t_1``, (2.75)-(2.76) for each ``(r_m, t_m)``, (2.60) for
    ``t_n`` -- first-order superposition makes them independent (Eq. 2.74).

    Note that ``dJ/dr_m = pdot+ - pdot-`` *is* the discontinuity in the primer
    rate, so continuity of ``pdot`` is never imposed as a constraint: it is
    what zero gradient means.  Likewise ``dJ/dt_m = -(H+ - H-)`` vanishing makes
    the Hamiltonian continuous, which by Eq. (2.77) forces ``pdot^T p = 0`` at
    the interior impulse, i.e. ``|p|`` peaks there at exactly 1.

    ``gtol_dv`` is the convergence target in m/s of cost per natural unit of
    each variable -- the same measure :meth:`PrimerAnalysis.nc` uses, but a
    decade tighter (1e-4 against the test's 1e-3).  They must not be equal:
    L-BFGS-B stops *at* its threshold, so matching the two leaves the solution
    sitting exactly on the boundary and the subsequent NC test decides by
    rounding.  With gtol 1e-3 the 1->2 three-impulse arc stopped with
    ``dJ/dt_m`` at 2.1e-3 m/s per transfer duration and was reported as
    violating, though every other condition passed by three decades.
    """
    t1, tn = float(traj.t_imp[0]), float(traj.t_imp[-1])
    mids = [(r, float(t)) for r, t in
            zip(interior_positions(traj, fm.mu), traj.t_imp[1:-1])]
    if scale_r is None:
        scale_r = 1e4                     # 10 km: the natural size of a nudge
    scale_t = tn - t1                     # the transfer duration, not a day
    t_ref = t1                            # centre everything on the seed
    r_refs = [np.asarray(r, float) for r, _ in mids]

    def build(z):
        a, m, b = _unpack(z, scale_r, scale_t, t_ref, r_refs)
        return n_impulse(node_o, node_f,
                         [a] + [t for _, t in m] + [b],
                         [r for r, _ in m], fm, sc, t0=traj.t0, tf=traj.tf)

    def fun(z):
        cur = build(z)
        if cur is None:
            return 1e9, np.zeros_like(z)
        a = analyse(cur, fm, n_eval=2)
        g = [a.grad_t_first * scale_t]
        for gr, gt in zip(a.mid_grad_r, a.mid_grad_t):
            g.extend(list(gr * scale_r) + [gt * scale_t])
        g.append(a.grad_t_last * scale_t)
        return cur.total_dv, np.array(g)

    z0 = _pack(t1, mids, tn, scale_r, scale_t, t_ref, r_refs)
    bnds = None
    if bounds_days is not None:
        def _b(d):                          # days -> centred, scaled units
            return (d * DAY - t_ref) / scale_t
        (lo1, hi1), (lo2, hi2) = bounds_days
        bnds = [(_b(lo1), _b(hi1))]
        for _ in mids:
            bnds += [(None, None)] * 3 + [(_b(lo1), _b(hi2))]
        bnds.append((_b(lo2), _b(hi2)))
    res = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": maxiter, "maxfun": 20 * maxiter,
                            "ftol": 1e-16, "gtol": gtol_dv})
    best = build(res.x)
    if best is None:
        return traj
    if verbose:
        gmax = float(np.max(np.abs(fun(res.x)[1])))
        print(f"      {tag}joint refinement over "
              f"{len(mids)*4 + 2} variables: {res.nit} iters, "
              f"dV {traj.total_dv:.4f} -> {best.total_dv:.4f} m/s, "
              f"max|grad| = {gmax:.2e} m/s"
              + ("" if gmax <= gtol_dv else "  <-- NOT converged"))
    return best if best.total_dv <= traj.total_dv else traj


def polish_nc(node_o, node_f, fm, sc, traj, scale_r=1e4, verbose=True,
              tag="", maxfev=200):
    """Solve ``grad J = 0`` directly, instead of minimising a flat cost.

    Near the optimum ``J`` is second-order flat in the interior-impulse
    variables: displacing ``(r_m, t_m)`` far enough to move ``max|p|`` by 1e-3
    changes the cost by less than 1e-9 m/s.  A cost minimiser therefore stops
    -- correctly, as far as the cost is concerned -- with the Lion-Handelsman
    gradients still at ~1e-3, and tightening its gradient tolerance does
    nothing because the tolerance is not what binds.

    Table 2.1 is a set of *equations*, not an objective, so it is solved as
    one: ``root`` on the same analytic gradient vector, seeded from the
    minimiser's answer.  Square system -- 4 unknowns and 4 equations per
    interior impulse, plus the two terminal times.
    """
    from scipy.optimize import root

    t1, tn = float(traj.t_imp[0]), float(traj.t_imp[-1])
    mids = [(r, float(t)) for r, t in
            zip(interior_positions(traj, fm.mu), traj.t_imp[1:-1])]
    scale_t = tn - t1
    t_ref = t1
    r_refs = [np.asarray(r, float) for r, _ in mids]

    def build(z):
        a, m, b = _unpack(z, scale_r, scale_t, t_ref, r_refs)
        return n_impulse(node_o, node_f,
                         [a] + [t for _, t in m] + [b],
                         [r for r, _ in m], fm, sc, t0=traj.t0, tf=traj.tf)

    def grad(z):
        cur = build(z)
        if cur is None:
            return np.full_like(z, 1e3)
        a = analyse(cur, fm, n_eval=2)
        g = [a.grad_t_first * scale_t]
        for gr, gt in zip(a.mid_grad_r, a.mid_grad_t):
            g.extend(list(gr * scale_r) + [gt * scale_t])
        g.append(a.grad_t_last * scale_t)
        return np.array(g)

    z0 = _pack(t1, mids, tn, scale_r, scale_t, t_ref, r_refs)
    g0 = float(np.max(np.abs(grad(z0))))
    # eps must be a real step, not machine epsilon: the Jacobian of grad J is
    # itself finite-differenced, and the default relative step reduces to noise.
    sol = root(grad, z0, method="hybr",
               options={"maxfev": maxfev, "xtol": 1e-12, "eps": 1e-6})
    best = build(sol.x)
    if best is None:
        return traj
    g1 = float(np.max(np.abs(grad(sol.x))))
    if verbose:
        print(f"      {tag}NC polish (root on grad J): "
              f"max|grad| {g0:.2e} -> {g1:.2e} m/s, "
              f"dV {traj.total_dv:.6f} -> {best.total_dv:.6f} m/s")
    # keep it only if the conditions really improved and the cost did not blow up
    if g1 < g0 and best.total_dv <= traj.total_dv + 1e-4:
        return best
    return traj


def add_impulse(node_o, node_f, fm, sc, traj, ana,
                betas=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6),
                bounds_days=None, verbose=True, maxiter=400, refine=True):
    """Insert one more impulse where ``|p| > 1``, then re-optimise everything.

    Seed by Eqs. (2.64)-(2.69).  The chapter's advice is followed literally --
    "if the resulting dJ >= 0, then decrease eps and repeat" -- and the whole
    beta ladder is tried, keeping whichever displacement gives the largest
    decrease.  The scale matters more than one might expect: beta = 0.05 (the
    chapter's suggested value) displaces a 44-minute LEO transfer by 346 km and
    *raises* the cost by 815 m/s, while beta = 3e-4 (2 km) lowers it.

    Works on an arc of any impulse count: :func:`primer.midcourse_seed` picks
    the leg carrying the largest ``|p|`` and the new impulse is spliced into
    the existing time/position list in order.
    """
    t1, tn = float(traj.t_imp[0]), float(traj.t_imp[-1])
    old_mids = [(r, float(t)) for r, t in
                zip(interior_positions(traj, fm.mu), traj.t_imp[1:-1])]

    seeded = None
    for beta in betas:
        seed = midcourse_seed(traj, ana, fm, beta=beta)
        if seed is None:
            return None
        t_m, r_m = seed
        mids = sorted(old_mids + [(r_m, t_m)], key=lambda p: p[1])
        tr = n_impulse(node_o, node_f, [t1] + [t for _, t in mids] + [tn],
                       [r for r, _ in mids], fm, sc, t0=traj.t0, tf=traj.tf)
        if verbose:
            got = "failed" if tr is None else f"dV = {tr.total_dv:10.5f} m/s"
            print(f"      seed beta = {beta:8.1e}: {got}")
        if tr is not None and tr.total_dv < traj.total_dv:
            if seeded is None or tr.total_dv < seeded[1].total_dv:
                seeded = (beta, tr, t_m)
    if seeded is None:
        if verbose:
            print(f"      no seed displacement lowered the cost -> keeping "
                  f"{traj.n_imp} impulses")
        return None
    beta, tr, t_m = seeded
    if verbose:
        print(f"      impulse {traj.n_imp + 1} seeded at t = {t_m/DAY:.6f} d "
              f"(beta = {beta:g}): dV {traj.total_dv:.4f} -> {tr.total_dv:.4f} m/s")
    if not refine:
        # the chapter's Fig. 2.8: the freshly inserted, still non-optimal arc.
        # |p| = 1 at the new impulse but pdot is discontinuous there and |p|
        # still exceeds 1 -- both NC violations that the refinement removes.
        return tr

    out = refine_n_impulse(node_o, node_f, fm, sc, tr,
                           bounds_days=bounds_days, verbose=verbose,
                           maxiter=maxiter, tag=f"n = {tr.n_imp}: ")
    return polish_nc(node_o, node_f, fm, sc, out, verbose=verbose,
                     tag=f"n = {out.n_imp}: ")


# kept for backwards compatibility
add_midcourse = add_impulse


# ---------------------------------------------------------------------------
def optimize_transfer(node_o, node_f, fm, sc, t_max_days=250.0,
                      window_index=0, half_width_days=3.0,
                      anchor_back_days=25.0,
                      coarse_step_days=0.2, fine_step_days=0.002,
                      tof_periods=(0.30, 1.20), n_tof=25,
                      n_candidates=3, allow_midcourse=True, max_impulses=6,
                      nc_tol=1e-3, seed=None, refine_after_insert=True,
                      verbose=True):
    """Full pipeline.

    Returns ``dict`` with the trajectory, the primer analysis, both scans and
    the J2 window information.
    """
    P = node_o.period
    mu = fm.mu

    # --- 1. J2 alignment window (closed form, then calibrated) ------------
    windows = raan_window(node_o, node_f, t_max_days * DAY)
    if not windows:
        raise RuntimeError("no J2 RAAN alignment inside the horizon")
    t_rough = windows[min(window_index, len(windows) - 1)]
    if verbose:
        print(f"  [1] secular J2 RAAN crossings in {t_max_days:.0f} d: "
              + ", ".join(f"{w/DAY:.3f} d" for w in windows))

    # --- 2. exact orbits; anchor early enough to allow the correction -----
    t_anchor = max(0.0, t_rough - anchor_back_days * DAY)
    cache = _window_cache_path(node_o, node_f, t_rough)
    t_win = float(np.load(cache)) if cache.exists() else None

    # if the calibration is already known, build the dense window straight to
    # its final extent instead of a wide provisional one
    t_hi = ((t_win if t_win is not None else t_rough)
            + half_width_days * DAY + tof_periods[1] * P)
    if verbose:
        print(f"  [2] exact propagation, anchor at {t_anchor/DAY:.3f} d")
    node_o.prepare(t_anchor, t_hi, verbose=verbose)
    node_f.prepare(t_anchor, t_hi, verbose=verbose)

    if t_win is not None:
        if verbose:
            print(f"      calibrated window {t_win/DAY:.4f} d loaded from cache")
    else:
        t_win = calibrate_window(node_o, node_f, t_rough,
                                 t_rough + anchor_back_days * DAY,
                                 verbose=verbose)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.array(t_win))
        t_hi = t_win + half_width_days * DAY + tof_periods[1] * P
        node_o.reprepare(t_hi, verbose=verbose)
        node_f.reprepare(t_hi, verbose=verbose)
    if verbose:
        print(f"      window: {t_rough/DAY:.4f} d (secular) -> "
              f"{t_win/DAY:.4f} d (propagated)")

    t_lo = max(t_anchor, t_win - half_width_days * DAY)

    # --- 3. departure scan on the exact orbits ----------------------------
    tof_grid = np.linspace(*tof_periods, n_tof) * P
    t_scan_hi = t_hi - tof_periods[1] * P
    if seed is not None:
        # a known (t1, ToF) supplied: skip the ~3 min pork-chop entirely.
        coarse = fine = None
        cands = [(float(seed[0]) * DAY, float(seed[1]) * 60.0, float("nan"))]
        if verbose:
            print(f"  [3] scan skipped, seeded at t1 = {seed[0]:.5f} d, "
                  f"ToF = {seed[1]:.3f} min")
    else:
        if verbose:
            print("  [3] departure scan (Lambert seeds on the exact orbits)")
        coarse = scan_departure(node_o, node_f,
                                np.arange(t_lo, t_scan_hi, coarse_step_days * DAY),
                                tof_grid, mu, exact=True, verbose=verbose)
        fine = scan_departure(node_o, node_f,
                              np.arange(t_lo, t_scan_hi, fine_step_days * DAY),
                              tof_grid, mu, exact=True, verbose=verbose)
        cands = fine.best(n_candidates, min_sep=0.5 * P)
        if not cands:
            raise RuntimeError("no feasible Lambert arc inside the window")
        if verbose:
            for c1, c2, c3 in cands:
                print(f"      candidate: t1 = {c1/DAY:10.5f} d, "
                      f"ToF = {c2/60:7.2f} min, dV = {c3:8.3f} m/s")

    # --- 4/5. converge + Lion-Handelsman terminal coasts -------------------
    bounds = [(t_lo / DAY, t_scan_hi / DAY), (t_lo / DAY, t_hi / DAY)]
    best = None
    for c1, c2, _ in cands:
        if verbose:
            print(f"  [4] converging t1 = {c1/DAY:.5f} d against {fm}")
        try:
            tr, _ = refine_terminal_times(node_o, node_f, fm, sc, c1, c1 + c2,
                                          bounds_days=bounds, verbose=verbose)
        except Exception as exc:                                 # noqa: BLE001
            if verbose:
                print(f"      failed: {exc}")
            continue
        if tr is not None and (best is None or tr.total_dv < best.total_dv):
            best = tr
    if best is None:
        raise RuntimeError("no candidate converged")

    # --- 6/7. primer necessary conditions ---------------------------------
    # --- 6. NECESSARY-CONDITION LOOP --------------------------------------
    # Repeat: evaluate the primer -> if |p| > 1 anywhere inside, add an impulse
    # there and re-optimise every free variable -> re-evaluate.  Stop when the
    # conditions of Table 2.1 hold (or when adding an impulse stops paying).
    # The terminal coasts need no separate branch: t_1 and t_n are decision
    # variables in every refinement, so dJ/dt_1 and dJ/dt_n are driven to zero
    # at each n, which is exactly the initial/final coast criterion of
    # Eqs. (2.59)-(2.60).
    if verbose:
        print("  [5] primer vector analysis")
    ana = analyse(best, fm)
    history = [(best, ana)]
    if verbose:
        print(f"      n = 2: dV = {best.total_dv:.4f} m/s, "
              f"max|p| = {ana.p_max:.5f}, "
              f"NC {'OK' if ana.optimal() else 'violated'}")

    if verbose:
        print("  [6] necessary-condition loop")
    while allow_midcourse and best.n_imp < max_impulses:
        if ana.p_max <= 1.0 + nc_tol:
            if verbose:
                ok = "satisfies" if ana.optimal() else "still VIOLATES"
                print(f"      max|p| = {ana.p_max:.5f} <= 1 + {nc_tol:g}: no "
                      f"further impulse is indicated; n = {best.n_imp} "
                      f"{ok} Table 2.1")
            break
        if verbose:
            print(f"      max|p| = {ana.p_max:.5f} > 1 -> try n = {best.n_imp + 1}")
        trn = add_impulse(node_o, node_f, fm, sc, best, ana,
                          bounds_days=bounds, verbose=verbose,
                          refine=refine_after_insert)
        # Ground truth for "does another impulse help" is the cost itself, not
        # the |p| > 1 predictor.  Once max|p| sits within the numerical floor
        # of 1, the seed degenerates to a null impulse (|dv| ~ 1e-2 m/s) that
        # buys nothing, and accepting it on a 1e-9 improvement would grow n
        # forever.  Require a materially lower cost instead.
        gain = None if trn is None else best.total_dv - trn.total_dv
        if trn is None or gain <= max(1e-4, 1e-6 * best.total_dv):
            if verbose:
                got = "no feasible insertion" if trn is None \
                    else f"gain only {gain:+.2e} m/s"
                print(f"      adding an impulse did not pay ({got}) -> "
                      f"stopping at n = {best.n_imp}")
            break
        best, ana = trn, analyse(trn, fm)
        history.append((best, ana))
        if verbose:
            print(f"      n = {best.n_imp}: dV = {best.total_dv:.4f} m/s, "
                  f"max|p| = {ana.p_max:.5f}, "
                  f"NC {'OK' if ana.optimal() else 'violated'}")
        if not refine_after_insert:
            if verbose:
                print("      --no-refine: stopping with the freshly inserted, "
                      "still non-optimal arc (chapter Fig. 2.8)")
            break
    else:
        if verbose and allow_midcourse:
            print(f"      hit max_impulses = {max_impulses}")

    return {"trajectory": best, "analysis": ana, "scan_coarse": coarse,
            "scan_fine": fine, "t_win": t_win, "t_win_secular": t_rough,
            "windows": windows, "two_impulse": history[0], "history": history,
            "refined_after_insert": refine_after_insert,
            "node_o": node_o, "node_f": node_f, "fm": fm, "sc": sc}

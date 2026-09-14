"""Impulsive transfer arcs under the full (perturbed) force model.

A Lambert arc is only a two-body seed.  Each leg is re-converged by Newton
shooting on the departure velocity so that the *perturbed* propagation hits the
required arrival position, with the Jacobian ``d r(t_b) / d v(t_a)`` taken from
the MEE state-transition matrix:

    d cart_b / d cart_a = (d cart/d mee)_b  Phi_mee  (d cart/d mee)_a^-1
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import MU
from .mee import mee_from_rv, rv_from_mee, dcart_dmee
from .dynamics import propagate, propagate_stm
from .lambert import lambert  # noqa: F401  (seed for solve_leg)

__all__ = ["Leg", "Trajectory", "solve_leg", "n_impulse", "two_impulse",
           "three_impulse", "interior_positions", "verify"]


# ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One ballistic arc between two impulses."""
    t_a: float
    t_b: float
    x_a: np.ndarray            # MEE just after the impulse at t_a
    x_b: np.ndarray            # MEE just before the impulse at t_b
    Phi: np.ndarray            # MEE STM from t_a to t_b
    v_a: np.ndarray            # ECI velocity just after t_a
    v_b: np.ndarray            # ECI velocity just before t_b


@dataclass
class Trajectory:
    """An n-impulse trajectory."""
    t_imp: np.ndarray                       # (n,) impulse times [s]
    dv: np.ndarray                          # (n,3) ECI velocity increments
    legs: list = field(default_factory=list)  # (n-1) Leg objects
    t0: float = 0.0                         # nominal mission start
    tf: float = 0.0                         # nominal mission end
    masses: np.ndarray = None               # (n+1,) mass before/after impulses
    converged: bool = True

    @property
    def dv_mag(self):
        return np.linalg.norm(self.dv, axis=1)

    @property
    def total_dv(self):
        return float(np.sum(self.dv_mag))

    @property
    def n_imp(self):
        return len(self.t_imp)

    @property
    def units(self):
        """Unit vectors along each impulse (ECI)."""
        return self.dv / self.dv_mag[:, None]

    def summary(self):
        s = [f"{self.n_imp}-impulse, total dV = {self.total_dv:8.3f} m/s"]
        t1 = self.t_imp[0]
        for i, (t, d) in enumerate(zip(self.t_imp, self.dv_mag)):
            s.append(f"    dv{i+1}: t = {t/86400:11.6f} d "
                     f"(+{(t-t1)/60:7.3f} min)   |dv| = {d:8.3f} m/s")
        s.append(f"    coast on the origin orbit from epoch to the first "
                 f"impulse: {t1/86400:.6f} d")
        return "\n".join(s)


# ---------------------------------------------------------------------------
def solve_leg(r_a, t_a, r_b, t_b, fm, mass, v_guess=None,
              tol=1e-3, maxiter=15, accept=1.0):
    """Converge a ballistic leg from ``r_a`` at ``t_a`` to ``r_b`` at ``t_b``.

    Returns ``(v_a, v_b, Phi, x_a, x_b, ok)`` where ``Phi`` is the MEE STM of
    the converged arc, ``tol`` is the target miss distance [m] and ``accept``
    the miss distance still deemed converged.

    Newton on the shooting residual converges in 3-5 steps when the Lambert
    seed is sound.  It is abandoned as soon as the miss distance stops
    decreasing: a near-degenerate transfer angle produces a seed from which the
    iteration diverges, and grinding through the full iteration budget on every
    line-search probe is what makes an outer optimisation appear to hang.
    """
    dt = t_b - t_a
    if v_guess is None:
        v_guess, _ = lambert(r_a, r_b, dt, fm.mu)
        if v_guess is None:
            return None, None, None, None, None, False
    v = np.array(v_guess, float)

    best = None
    prev_miss = np.inf
    stalls = 0
    for _ in range(maxiter):
        x_a = mee_from_rv(r_a, v, fm.mu)
        try:
            _, X, PHI = propagate_stm(x_a, (t_a, t_b), fm, mass,
                                      t_eval=[t_a, t_b])
        except RuntimeError:
            # a line search can propose a velocity whose transfer orbit is
            # degenerate (grazing / near-parabolic); the integrator then stalls.
            # Treat it as a failed trial rather than killing the optimisation.
            break
        x_b = X[:, -1]
        Phi = PHI[-1]
        r_end, v_end = rv_from_mee(x_b, fm.mu)
        miss = r_b - r_end
        mn = float(np.linalg.norm(miss))
        if best is None or mn < best[0]:
            best = (mn, v.copy(), v_end, Phi, x_a, x_b)
        if mn < tol:
            return v, v_end, Phi, x_a, x_b, True
        if mn > 0.5 * prev_miss:                # not at least halving: stalling
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
        prev_miss = mn
        Jb = dcart_dmee(x_b, fm.mu)
        Ja_inv = np.linalg.inv(dcart_dmee(x_a, fm.mu))
        Dcart = Jb @ Phi @ Ja_inv                      # d cart_b / d cart_a
        drdv = Dcart[0:3, 3:6]
        try:
            v = v + np.linalg.solve(drdv, miss)
        except np.linalg.LinAlgError:
            break

    mn, v, v_end, Phi, x_a, x_b = best
    return v, v_end, Phi, x_a, x_b, mn < accept


def n_impulse(node_o, node_f, times, mid_positions, fm, sc, t0=None, tf=None,
              v_guess=None):
    """General n-impulse rendezvous arc.

    Parameters
    ----------
    times : sequence of n epochs ``[t_1, t_m1, ..., t_mk, t_n]`` (k = n - 2).
    mid_positions : the k interior impulse positions (ECI).  Empty for the
        two-impulse case.

    The chaser coasts on the origin orbit until ``t_1`` and on the target orbit
    from ``t_n``, so the first and last impulse times *are* the terminal coasts
    of the chapter -- there is no separate coast variable.

    Each leg is Lambert-seeded and then Newton-shot against the full force
    model.  Returns ``None`` if any leg fails to converge.
    """
    times = np.asarray(times, float)
    n = times.size
    if n < 2 or len(mid_positions) != n - 2:
        raise ValueError("len(mid_positions) must be len(times) - 2")
    if np.any(np.diff(times) <= 0.0):
        return None

    r1, v_before = rv_from_mee(node_o.mee(times[0]), fm.mu)
    r_last, v_after = rv_from_mee(node_f.mee(times[-1]), fm.mu)
    way = [r1] + [np.asarray(r, float) for r in mid_positions] + [r_last]

    m = sc.wet_mass
    masses = [m]
    legs, dvs = [], []
    guess = v_guess
    for i in range(n - 1):
        va, vb, Phi, xa, xb, ok = solve_leg(way[i], times[i], way[i + 1],
                                            times[i + 1], fm, m,
                                            v_guess=guess if i == 0 else None)
        if not ok:
            return None
        dv = va - v_before
        if fm.drag is not None:
            # the coast mass is the post-burn mass; only matters through drag
            m_leg = sc.mass_after(np.linalg.norm(dv), m)
            va, vb, Phi, xa, xb, ok = solve_leg(way[i], times[i], way[i + 1],
                                                times[i + 1], fm, m_leg,
                                                v_guess=va)
            if not ok:
                return None
            dv = va - v_before
        dvs.append(dv)
        m = sc.mass_after(np.linalg.norm(dv), m)
        masses.append(m)
        legs.append(Leg(times[i], times[i + 1], xa, xb, Phi, va, vb))
        v_before = vb

    dvs.append(v_after - v_before)
    masses.append(sc.mass_after(np.linalg.norm(dvs[-1]), m))

    return Trajectory(
        t_imp=times.copy(),
        dv=np.vstack(dvs),
        legs=legs,
        t0=times[0] if t0 is None else t0,
        tf=times[-1] if tf is None else tf,
        masses=np.array(masses),
    )


def two_impulse(node_o, node_f, t1, t2, fm, sc, t0=None, tf=None,
                v_guess=None):
    """Two-impulse rendezvous (Fig. 2.3 of the chapter)."""
    return n_impulse(node_o, node_f, [t1, t2], [], fm, sc, t0, tf, v_guess)


def three_impulse(node_o, node_f, t1, tm, t2, r_m, fm, sc, t0=None, tf=None):
    """Three-impulse trajectory with a midcourse impulse at ``(r_m, tm)``."""
    return n_impulse(node_o, node_f, [t1, tm, t2], [r_m], fm, sc, t0, tf)


def interior_positions(traj, mu):
    """The interior impulse positions of an existing trajectory (ECI)."""
    return [rv_from_mee(leg.x_b, mu)[0] for leg in traj.legs[:-1]]


def verify(traj, node_o, node_f, fm, sc):
    """Independent end-to-end feasibility check of a finished trajectory.

    The solver builds the arc leg by leg, each leg shot to its own target, so
    a bookkeeping slip could in principle produce a "cheap" trajectory that is
    not actually flyable.  This routine ignores all of that: it takes the
    origin state at ``t_1``, applies the reported ECI impulse vectors, and
    propagates straight through to ``t_n``, then compares with the target
    orbit's state there.

    Returns a dict with the position/velocity miss and the propellant.
    """
    r, v = rv_from_mee(node_o.mee(traj.t_imp[0]), fm.mu)
    m = sc.wet_mass
    for i in range(traj.n_imp - 1):
        v = v + traj.dv[i]
        m = sc.mass_after(np.linalg.norm(traj.dv[i]), m)
        x = mee_from_rv(r, v, fm.mu)
        _, X = propagate(x, (traj.t_imp[i], traj.t_imp[i + 1]), fm, m,
                         t_eval=[traj.t_imp[i + 1]])
        r, v = rv_from_mee(X[:, -1], fm.mu)
    v = v + traj.dv[-1]
    m = sc.mass_after(np.linalg.norm(traj.dv[-1]), m)

    r_t, v_t = rv_from_mee(node_f.mee(traj.t_imp[-1]), fm.mu)
    return {
        "pos_miss_m": float(np.linalg.norm(r - r_t)),
        "vel_miss_mps": float(np.linalg.norm(v - v_t)),
        "final_mass_kg": float(m),
        "propellant_kg": float(sc.wet_mass - m),
        "dv_sum_mps": traj.total_dv,
    }

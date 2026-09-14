"""Costates, the primer vector and the switching function (slides 11-12).

Slide 11 gives the Hamiltonian for the MEE + mass state

    H = (1 - lam_m) Gamma
        + lam_mee^T A(x) Gamma u + lam_mee^T A(x) a_g
        + lam_mee^T A(x) a_D + lam_mee^T b(x)

so that, collecting the terms linear in the thrust,

    p(t) = -A^T(x) lam_mee(t)        (primer vector, LVLH components)
    u*   = p / |p|
    S(t) = |p| - (1 - lam_m)         (switching function, slide 12)
    Gamma* = Gamma_max  if S > 0,  0 if S < 0.

Two facts make the chapter's impulsive machinery usable unchanged:

1. ``p`` is a *physical vector*.  Because ``A = (d x_mee / d v) Q_r``, we have
   ``-A^T lam_mee = Q_r^T (-lam_v)``: rotating the MEE primer into ECI gives
   exactly Lawden's Cartesian primer ``-lam_v``.  (Verified in
   :mod:`opt_trj.validate`.)
2. On a coast arc the costate obeys ``lam_dot = -F^T lam`` with
   ``F = d xdot / d x``, i.e. ``lam(t) = Phi(t,t0)^-T lam(t0)`` -- the MEE
   analogue of Eq. (2.36)-(2.37).

Impulsive limit.  With the chapter's cost ``J = int Gamma dt = sum |dv_k|``
every impulse is weighted equally, the mass costate normalises to
``lam_m = 0``, and ``S = |p| - 1`` recovers Lawden's conditions of Table 2.1.
``lam_m`` is still integrated by :func:`lam_m_rate` for the finite-thrust (CSI)
cross-check reported by the driver.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU
from .mee import A_matrix, dcart_dmee, rv_from_mee, lvlh_frame, csqrt_norm
from .dynamics import jac_mee, mee_rhs

__all__ = ["costate_rhs", "propagate_costate", "primer_from_costate",
           "primer_history", "primer_map", "solve_primer_bvp",
           "switching_function", "lam_m_profile", "lam_m_rate",
           "hamiltonian_coast"]


# ---------------------------------------------------------------------------
# costate propagation
# ---------------------------------------------------------------------------
def costate_rhs(x, lam, fm, mass):
    """``lam_mee_dot = -F^T lam_mee`` on a coast arc."""
    return -jac_mee(x, fm, mass).T @ lam


def propagate_costate(x0, lam0, t_span, fm, mass, t_eval=None):
    """Integrate state and costate together.  Returns ``(t, LAM, X)``."""
    def rhs(t, y):
        x, lam = y[:6], y[6:]
        return np.concatenate([mee_rhs(x, fm, mass), costate_rhs(x, lam, fm, mass)])

    y0 = np.concatenate([np.asarray(x0, float), np.asarray(lam0, float)])
    atol = np.concatenate([[1e-6], np.full(5, 1e-12), np.full(6, 1e-12)])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=atol,
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[6:], sol.y[:6]


# ---------------------------------------------------------------------------
# primer vector
# ---------------------------------------------------------------------------
def primer_map(x, mu=MU):
    """3x6 matrix ``R(x)`` with ``p_eci = R(x) lam_mee``.

    ``R = -[ (d cart / d mee)^-T ]_{rows 3:6}``, i.e. the velocity block of the
    Cartesian costate, negated.  Equivalent to ``Q_r (-A^T)``.
    """
    Jt_inv = np.linalg.inv(dcart_dmee(x, mu).T)
    return -Jt_inv[3:6, :]


def primer_from_costate(x, lam, mu=MU):
    """Return ``(p, pdot)`` in ECI from the MEE state and costate.

    ``pdot = lam_r`` exactly whenever the non-thrust acceleration depends on
    position only (two-body + zonal gravity).  With drag switched on this is an
    approximation; :func:`primer_history` then differentiates numerically.
    """
    lam_cart = np.linalg.solve(dcart_dmee(x, mu).T, lam)
    return -lam_cart[3:6], lam_cart[0:3]


def primer_history(x0, lam0, t_span, fm, mass, t_eval=None):
    """Primer magnitude/rate along a coast arc.

    Returns ``(t, P, PD, X)`` with ``P``, ``PD`` of shape ``(3, n)`` in ECI.
    """
    t, LAM, X = propagate_costate(x0, lam0, t_span, fm, mass, t_eval=t_eval)
    n = t.size
    P = np.empty((3, n))
    PD = np.empty((3, n))
    for i in range(n):
        P[:, i], PD[:, i] = primer_from_costate(X[:, i], LAM[:, i], fm.mu)
    if fm.drag is not None:                      # pdot != lam_r: differentiate
        PD = np.gradient(P, t, axis=1, edge_order=2)
    return t, P, PD, X


def solve_primer_bvp(x1, x2, Phi21, u1, u2, mu=MU):
    """Initial MEE costate reproducing the two impulse directions.

    The MEE analogue of Eq. (2.43)-(2.49):

        p(t1) = u1 ,  p(t2) = u2 ,  lam(t2) = Phi21^-T lam(t1)

    gives 6 linear equations for the 6 unknowns ``lam_mee(t1)``.

    Parameters
    ----------
    x1, x2 : MEE state at the first and last impulse.
    Phi21  : state transition matrix from t1 to t2 of the coast arc.
    u1, u2 : ECI unit vectors along the two impulses.
    """
    R1 = primer_map(x1, mu)
    R2 = primer_map(x2, mu) @ np.linalg.inv(Phi21.T)
    M = np.vstack([R1, R2])
    rhs = np.concatenate([u1, u2])
    lam1, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    resid = float(np.max(np.abs(M @ lam1 - rhs)))
    cond = float(np.linalg.cond(M))
    return lam1, resid, cond


# ---------------------------------------------------------------------------
# switching function / mass costate (finite-thrust cross-check)
# ---------------------------------------------------------------------------
def switching_function(p_mag, lam_m=0.0):
    """``S(t) = |p| - (1 - lam_m)`` (slide 12)."""
    return p_mag - (1.0 - lam_m)


def lam_m_profile(traj, cost="dv"):
    """Piecewise-constant ``lam_m`` on an impulsive trajectory (one per leg).

    On a coast arc ``Gamma = 0`` and ``a_T = 0``, so ``lam_m_dot = 0`` directly
    from slide 11 -- ``lam_m`` cannot change while drifting.

    Across an impulse it does not change either.  Substituting ``a_T = Gamma u``
    with ``u = p/|p|`` and ``lam_mee^T A = -p^T`` into the slide's rate,

        (1 - lam_m) Gamma + lam_mee^T A a_T
            = Gamma [(1 - lam_m) - |p|]  =  -Gamma S ,

    and an impulse occurs precisely where ``S = 0``, so the integrand vanishes
    through the burn and the jump in ``lam_m`` is zero.

    Hence ``lam_m`` is constant over the whole trajectory.  With the chapter's
    cost ``J = int Gamma dt = sum |dv_k|`` the mass never enters the cost, the
    transversality condition on the free final mass gives ``lam_m = 0``, and the
    switching function collapses to Lawden's ``S(t) = |p(t)| - 1``.

    The ``cost`` argument is a placeholder for other normalisations (e.g. a
    maximum-final-mass objective, for which ``lam_m`` is a nonzero constant);
    only ``"dv"`` is implemented.
    """
    if cost != "dv":
        raise NotImplementedError(f"lam_m normalisation {cost!r}")
    return np.zeros(len(traj.legs))


def lam_m_rate(x, lam, mass, thrust_rtn, gamma, mu=MU):
    """``lam_m_dot = -(1/m)[(1 - lam_m) Gamma + lam_mee^T A(x) a_T]`` (slide 11).

    Zero on a coast arc (``Gamma = 0``, ``a_T = 0``), so ``lam_m`` is piecewise
    constant on an impulsive trajectory.
    """
    if gamma == 0.0:
        return 0.0
    return -(1.0 / mass) * (gamma + lam @ (A_matrix(x, mu) @ thrust_rtn))


def hamiltonian_coast(x, lam, fm, mass):
    """``H`` on a null-thrust arc: ``H = lam_mee^T [A a_pert + b]``.

    Used for the midcourse-impulse time gradient, Eq. (2.76).  Equal to
    ``pdot^T v - p^T g`` of Eq. (2.22) in the Cartesian formulation.
    """
    return float(lam @ mee_rhs(x, fm, mass))

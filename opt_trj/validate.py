"""Self-checks for the MEE / costate machinery.

Run with ``python -m opt_trj.validate``.  Every check is a physical identity,
not a regression baseline, so a failure means the model is wrong -- not merely
changed.
"""

import numpy as np

from .constants import MU, R_E, DEG
from .mee import (A_matrix, b_vector, rv_from_mee, mee_from_rv, lvlh_frame,
                  mee_from_coe, dcart_dmee)
from .dynamics import ForceModel, mee_rhs, jac_mee, propagate, propagate_stm
from .perturbations import a_zonal_rtn
from .costate import primer_from_costate, propagate_costate


def _sample_state():
    return mee_from_coe(a=R_E + 700e3, e=0.012, inc=51.6 * DEG,
                        raan=40.0 * DEG, argp=25.0 * DEG, nu=100.0 * DEG)


def check_roundtrip(tol=1e-9):
    x = _sample_state()
    r, v = rv_from_mee(x)
    x2 = mee_from_rv(r, v)
    err = np.max(np.abs(x - x2) / np.array([x[0], 1, 1, 1, 1, 1]))
    print(f"  MEE <-> Cartesian round trip      : {err:.3e}")
    assert err < tol, err


def check_A_b(tol=1e-8):
    """``A(x) a + b(x)`` must equal the MEE rate implied by Cartesian dynamics.

    Cartesian:  rdot = v,  vdot = -mu r/r^3 + Q_r a
    so          xdot_mee = (d mee / d cart) [v ; -mu r/r^3 + Q_r a].
    """
    x = _sample_state()
    r, v = rv_from_mee(x)
    Q = lvlh_frame(r, v)
    rng = np.random.default_rng(0)
    a_rtn = rng.normal(size=3) * 1e-3          # arbitrary RTN acceleration

    lhs = A_matrix(x) @ a_rtn + b_vector(x)
    cart_dot = np.concatenate([v, -MU * r / np.linalg.norm(r) ** 3 + Q @ a_rtn])
    rhs = np.linalg.solve(dcart_dmee(x), cart_dot)

    err = np.max(np.abs(lhs - rhs) / (np.abs(rhs) + 1e-12))
    print(f"  A(x)a + b(x) vs Cartesian dynamics: {err:.3e}")
    assert err < tol, err


def check_zonal(tol=1e-8):
    """J2 in the slide's (dg_n, dg_r) form vs the textbook closed form."""
    x = _sample_state()
    r, v = rv_from_mee(x)
    rn = np.linalg.norm(r)
    J2 = 1.082626683e-3
    z2 = (r[2] / rn) ** 2
    kk = 1.5 * J2 * MU * R_E ** 2 / rn ** 4
    a_eci = np.array([kk * r[0] / rn * (5 * z2 - 1),
                      kk * r[1] / rn * (5 * z2 - 1),
                      kk * r[2] / rn * (5 * z2 - 3)])
    ref = lvlh_frame(r, v).T @ a_eci
    got = a_zonal_rtn(x, degree=2)
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    print(f"  J2 (slide form) vs closed form    : {err:.3e}")
    assert err < tol, err


def check_jacobian(tol=1e-6):
    """Complex-step Jacobian vs central finite differences."""
    fm = ForceModel(zonal_degree=2)
    x = _sample_state()
    F = jac_mee(x, fm, 5000.0)
    Ffd = np.zeros((6, 6))
    scale = np.array([x[0], 1, 1, 1, 1, 1])
    for j in range(6):
        h = 1e-6 * scale[j]
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        Ffd[:, j] = (mee_rhs(xp, fm, 5000.0) - mee_rhs(xm, fm, 5000.0)) / (2 * h)
    err = np.max(np.abs(F - Ffd)) / np.max(np.abs(Ffd))
    print(f"  complex-step Jacobian vs FD       : {err:.3e}")
    assert err < tol, err


def check_stm(tol=1e-7):
    """STM vs finite differences of the flow (compared in scaled units)."""
    fm = ForceModel(zonal_degree=2)
    x0 = _sample_state()
    T = 1200.0
    _, X, PHI = propagate_stm(x0, (0.0, T), fm, 5000.0, t_eval=[0.0, T])
    Phi = PHI[-1]
    scale = np.array([x0[0], 1, 1, 1, 1, 1])
    Pfd = np.zeros((6, 6))
    for j in range(6):
        h = 1e-6 * scale[j]
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        _, Xp = propagate(xp, (0.0, T), fm, 5000.0, t_eval=[T])
        _, Xm = propagate(xm, (0.0, T), fm, 5000.0, t_eval=[T])
        Pfd[:, j] = (Xp[:, -1] - Xm[:, -1]) / (2 * h)
    S = np.diag(scale)                                   # nondimensionalise
    err = np.max(np.abs(np.linalg.solve(S, (Phi - Pfd) @ S)))
    print(f"  STM vs finite-difference flow     : {err:.3e}")
    assert err < tol, err


def check_primer_consistency(tol=1e-8):
    """p = -A^T lambda_mee (LVLH) must equal -lambda_v (ECI) after rotation.

    This is the identity that lets the whole Lion-Handelsman apparatus of the
    chapter be reused verbatim in the MEE formulation:  the MEE primer and the
    Cartesian primer are the *same physical vector*.
    """
    x = _sample_state()
    lam = np.array([1e-7, 0.4, -0.2, 0.9, 0.3, -0.5])
    r, v = rv_from_mee(x)
    Q = lvlh_frame(r, v)

    p_rtn = -A_matrix(x).T @ lam
    p_eci_mee = Q @ p_rtn

    lam_cart = np.linalg.solve(dcart_dmee(x).T, lam)     # [lambda_r ; lambda_v]
    p_eci_cart = -lam_cart[3:]

    err = np.max(np.abs(p_eci_mee - p_eci_cart)) / np.max(np.abs(p_eci_cart))
    print(f"  primer  -A^T lam  vs  -lambda_v   : {err:.3e}")
    assert err < tol, err


def check_costate_transport(tol=1e-7):
    """lambda(t) = Phi(t,t0)^-T lambda(t0) must satisfy lamdot = -F^T lam."""
    fm = ForceModel(zonal_degree=2)
    x0 = _sample_state()
    lam0 = np.array([1e-7, 0.4, -0.2, 0.9, 0.3, -0.5])
    T = 900.0
    t, X, PHI = propagate_stm(x0, (0.0, T), fm, 5000.0, t_eval=np.linspace(0, T, 5))
    lam_stm = np.array([np.linalg.solve(PHI[i].T, lam0) for i in range(len(t))]).T
    lam_ode = propagate_costate(x0, lam0, (0.0, T), fm, 5000.0, t_eval=t)[1]
    err = np.max(np.abs(lam_stm - lam_ode)) / np.max(np.abs(lam_stm))
    print(f"  costate: Phi^-T vs direct ODE     : {err:.3e}")
    assert err < tol, err


def check_two_body_primer(tol=1e-5):
    """With J2 off, the MEE primer must match Lawden's Cartesian pdd = G p."""
    from .costate import primer_history
    from .mee import period

    fm = ForceModel(zonal_degree=0)
    x0 = _sample_state()
    T = 0.35 * period(x0)
    lam0 = np.array([2e-8, 0.4, -0.2, 0.9, 0.3, -0.5])
    t, P, PD, _ = primer_history(x0, lam0, (0.0, T), fm, 5000.0,
                                 t_eval=np.linspace(0, T, 60))

    # integrate pdd = G(r) p directly in Cartesian
    from scipy.integrate import solve_ivp
    r0, v0 = rv_from_mee(x0)

    def rhs(tt, y):
        r, v, p, pd = y[0:3], y[3:6], y[6:9], y[9:12]
        rn = np.linalg.norm(r)
        G = MU * (3 * np.outer(r, r) / rn ** 5 - np.eye(3) / rn ** 3)
        return np.concatenate([v, -MU * r / rn ** 3, pd, G @ p])

    y0 = np.concatenate([r0, v0, P[:, 0], PD[:, 0]])
    s = solve_ivp(rhs, (0, T), y0, method="DOP853", rtol=1e-13, atol=1e-14,
                  t_eval=t)
    err = np.max(np.abs(s.y[6:9] - P)) / np.max(np.abs(P))
    print(f"  MEE primer vs Lawden pdd = G p    : {err:.3e}")
    assert err < tol, err


def main():
    print("opt_trj self-check")
    check_roundtrip()
    check_A_b()
    check_zonal()
    check_jacobian()
    check_stm()
    check_primer_consistency()
    check_costate_transport()
    check_two_body_primer()
    print("all checks passed")


if __name__ == "__main__":
    main()

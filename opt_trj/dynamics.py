"""MEE equations of motion, their Jacobian, and state-transition propagation.

The coast (null-thrust) dynamics are

    xdot = A(x) (a_g + a_D) + b(x)

and the associated variational equation is  Phidot = F(t,x) Phi  with
``F = d xdot / d x`` obtained by *complex-step* differentiation (machine
precision, no step-size tuning).

The same ``F`` drives the costate equation of slide 11,

    lambda_mee_dot = -F^T lambda_mee ,

so ``lambda_mee(t) = Phi(t,t0)^-T lambda_mee(t0)`` -- the MEE analogue of the
symplectic Cartesian result used in the chapter (Eq. 2.36-2.37).
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU, R_E
from .mee import A_matrix, b_vector, rv_from_mee, mee_from_rv, dcart_dmee
from .perturbations import a_zonal_rtn, a_drag_rtn

__all__ = ["ForceModel", "mee_rhs", "jac_mee", "propagate", "propagate_stm",
           "propagate_cart"]

_ATOL = np.array([1e-6, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
_RTOL = 1e-12


class ForceModel:
    """Container for the perturbations acting on a coasting spacecraft."""

    def __init__(self, zonal_degree=2, drag=None, mu=MU, re=R_E):
        self.zonal_degree = int(zonal_degree)
        self.drag = drag
        self.mu = mu
        self.re = re

    def accel_rtn(self, x, mass):
        """Non-thrust perturbing acceleration in LVLH components."""
        a = a_zonal_rtn(x, self.zonal_degree, self.mu, self.re)
        if self.drag is not None:
            a = a + a_drag_rtn(x, mass, self.drag, self.mu, self.re)
        return a

    def __repr__(self):
        return (f"ForceModel(zonal_degree={self.zonal_degree}, "
                f"drag={'on' if self.drag else 'off'})")


# ---------------------------------------------------------------------------
def mee_rhs(x, fm, mass):
    """``xdot`` for a coasting arc (thrust off)."""
    return A_matrix(x, fm.mu) @ fm.accel_rtn(x, mass) + b_vector(x, fm.mu)


def jac_mee(x, fm, mass, step=1e-30):
    """``F = d xdot / d x`` (6x6) by complex-step differentiation."""
    F = np.zeros((6, 6))
    for j in range(6):
        xp = np.array(x, dtype=complex)
        xp[j] = xp[j] + 1j * step
        F[:, j] = np.imag(mee_rhs(xp, fm, mass)) / step
    return F


# ---------------------------------------------------------------------------
def propagate(x0, t_span, fm, mass, t_eval=None, rtol=_RTOL, atol=_ATOL):
    """Propagate the MEE state over ``t_span = (t0, t1)``.

    Returns ``(t, X)`` with ``X`` of shape ``(6, n)``.  ``L`` is *not* wrapped.
    """
    sol = solve_ivp(lambda t, y: mee_rhs(y, fm, mass), t_span, np.asarray(x0, float),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval,
                    dense_output=t_eval is None)
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    return sol.t, sol.y


def propagate_dense(x0, t_span, fm, mass, rtol=_RTOL, atol=_ATOL):
    """Propagate and return the solver's dense-output callable ``x(t)``.

    Preferred over splining a sampled history: DOP853's dense output is
    accurate to the integration tolerance, so a mission point's state can be
    queried at any epoch inside the window without re-integrating.
    """
    sol = solve_ivp(lambda t, y: mee_rhs(y, fm, mass), t_span,
                    np.asarray(x0, float), method="DOP853", rtol=rtol,
                    atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    return sol.sol


def propagate_stm(x0, t_span, fm, mass, t_eval=None, rtol=_RTOL, atol=_ATOL):
    """Propagate state + 6x6 state-transition matrix.

    Returns ``(t, X, PHI)`` where ``PHI`` has shape ``(n, 6, 6)``.
    """
    x0 = np.asarray(x0, float)

    def rhs(t, y):
        x = y[:6]
        Phi = y[6:].reshape(6, 6)
        F = jac_mee(x, fm, mass)
        return np.concatenate([mee_rhs(x, fm, mass), (F @ Phi).ravel()])

    y0 = np.concatenate([x0, np.eye(6).ravel()])
    atol_full = np.concatenate([atol, np.full(36, 1e-12)])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=rtol, atol=atol_full,
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"STM propagation failed: {sol.message}")
    n = sol.t.size
    return sol.t, sol.y[:6], sol.y[6:].T.reshape(n, 6, 6)


def propagate_cart(r0, v0, t_span, fm, mass, **kw):
    """Convenience: propagate from Cartesian and return Cartesian."""
    x0 = mee_from_rv(r0, v0, fm.mu)
    t, X = propagate(x0, t_span, fm, mass, **kw)
    R = np.empty((3, X.shape[1]))
    V = np.empty((3, X.shape[1]))
    for i in range(X.shape[1]):
        R[:, i], V[:, i] = rv_from_mee(X[:, i], fm.mu)
    return t, R, V


def dmee_dcart(x, mu=MU):
    """``d[p,f,g,h,k,L] / d[r;v]`` -- inverse of :func:`mee.dcart_dmee`."""
    return np.linalg.inv(dcart_dmee(x, mu))

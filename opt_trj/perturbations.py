"""Perturbing accelerations  a = a_T + a_g + a_D  (slide 10).

Everything is returned in the **LVLH / RTN frame** ``[a_r, a_theta, a_h]``,
which is what ``A(x)`` in :mod:`opt_trj.mee` consumes.

* ``a_g`` -- Earth oblateness, zonal harmonics J2..Jn, exactly in the form of
  the slide::

      a_g   = Q_r^T dg
      dg    = dg_n i_n - dg_r i_r
      dg_n  = -(mu cos(phi) / r^2) sum_k (Re/r)^k P'_k(sin phi) J_k
      dg_r  = -(mu / r^2)          sum_k (k+1) (Re/r)^k P_k(sin phi) J_k

  with ``i_n`` the local-north direction obtained by removing the radial part
  of the inertial pole ``e_n = [0,0,1]``.

* ``a_D`` -- atmospheric drag, also as on the slide::

      a_D = [a_Dr, a_Dtheta, 0]
      a_Dr     = -0.5 rho B v v_r
      a_Dtheta = -0.5 rho B v v_theta
      v_r      = sqrt(mu/p) (f sinL - g cosL)
      v_theta  = sqrt(mu/p) (1 + f cosL + g sinL)

  ``B = C_D * S / m`` is the inverse ballistic coefficient [m^2/kg].  **The
  drag model is disabled by default**: ``S``, ``C_D`` and an atmosphere model
  were never specified for this vehicle, and inventing them would silently bias
  the answer.  Supply ``DragModel(...)`` explicitly to switch it on.

All functions are complex-step safe.
"""

import numpy as np

from .constants import MU, R_E, J_ZONAL
from .mee import csqrt_norm, rv_from_mee, lvlh_frame

__all__ = ["legendre_P", "legendre_dP", "a_zonal_rtn", "DragModel",
           "ExponentialAtmosphere", "a_drag_rtn"]


# ---------------------------------------------------------------------------
# Zonal gravity
# ---------------------------------------------------------------------------
def legendre_P(k, u):
    """Legendre polynomial ``P_k(u)`` for k = 2, 3, 4."""
    if k == 2:
        return 0.5 * (3.0 * u * u - 1.0)
    if k == 3:
        return 0.5 * (5.0 * u ** 3 - 3.0 * u)
    if k == 4:
        return (35.0 * u ** 4 - 30.0 * u * u + 3.0) / 8.0
    raise ValueError(f"zonal degree {k} not implemented (k = 2..4)")


def legendre_dP(k, u):
    """``dP_k/du`` for k = 2, 3, 4."""
    if k == 2:
        return 3.0 * u
    if k == 3:
        return 0.5 * (15.0 * u * u - 3.0)
    if k == 4:
        return 0.5 * (35.0 * u ** 3 - 15.0 * u)
    raise ValueError(f"zonal degree {k} not implemented (k = 2..4)")


def a_zonal_rtn(x, degree=2, mu=MU, re=R_E, jcoef=None):
    """Oblateness acceleration in LVLH components, from the MEE state ``x``.

    ``degree = 2`` gives J2 only; ``degree = 0`` disables the perturbation.
    """
    dt = np.result_type(x[0], 0.0)
    if degree < 2:
        return np.zeros(3, dtype=dt)
    jcoef = J_ZONAL if jcoef is None else jcoef

    r, v = rv_from_mee(x, mu)
    rn = csqrt_norm(r)
    ir = r / rn

    sphi = r[2] / rn                       # sin(geocentric latitude)
    cphi = np.sqrt(1.0 - sphi * sphi)

    # local north: pole with its radial component removed
    en = np.zeros(3, dtype=dt)
    en[2] = 1.0
    nvec = en - (en @ ir) * ir
    nn = csqrt_norm(nvec)
    i_n = nvec / nn

    dg_n = np.zeros((), dtype=dt)
    dg_r = np.zeros((), dtype=dt)
    for k in range(2, degree + 1):
        Jk = jcoef[k]
        rk = (re / rn) ** k
        dg_n = dg_n - mu * cphi / rn ** 2 * rk * legendre_dP(k, sphi) * Jk
        dg_r = dg_r - mu / rn ** 2 * (k + 1.0) * rk * legendre_P(k, sphi) * Jk

    dg = dg_n * i_n - dg_r * ir            # ECI components
    Q = lvlh_frame(r, v)
    return Q.T @ dg


# ---------------------------------------------------------------------------
# Drag  (off unless the user supplies the vehicle aerodynamics)
# ---------------------------------------------------------------------------
class ExponentialAtmosphere:
    """Piecewise exponential density model (US Standard 1976 fit).

    Only used if a :class:`DragModel` is explicitly configured.
    """

    # base altitude [km], base density [kg/m^3], scale height [km]
    _TAB = np.array([
        [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682],
        [40, 3.972e-3, 7.554], [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714],
        [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799], [90, 3.396e-6, 5.382],
        [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
        [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523],
        [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546],
        [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298], [400, 3.725e-12, 58.515],
        [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
        [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05],
        [1000, 3.019e-15, 268.00],
    ])

    def __call__(self, alt_m):
        alt_km = np.real(alt_m) / 1e3
        i = int(np.searchsorted(self._TAB[:, 0], alt_km, side="right") - 1)
        i = min(max(i, 0), len(self._TAB) - 1)
        h0, rho0, H = self._TAB[i]
        return rho0 * np.exp(-(alt_m / 1e3 - h0) / H)


class DragModel:
    """Cannonball drag.

    Parameters
    ----------
    cd : float
        Drag coefficient (dimensionless).
    area : float
        Reference cross-sectional area [m^2].
    atmosphere : callable
        ``rho = atmosphere(altitude_m)``.
    """

    def __init__(self, cd, area, atmosphere=None):
        self.cd = float(cd)
        self.area = float(area)
        self.atmosphere = atmosphere or ExponentialAtmosphere()


def a_drag_rtn(x, mass, drag, mu=MU, re=R_E):
    """Drag acceleration in LVLH components, per the slide's ``a_D`` form."""
    dt = np.result_type(x[0], 0.0)
    if drag is None:
        return np.zeros(3, dtype=dt)

    p, f, g, _, _, L = x
    cL, sL = np.cos(L), np.sin(L)
    sq = np.sqrt(mu / p)
    v_r = sq * (f * sL - g * cL)
    v_t = sq * (1.0 + f * cL + g * sL)
    v = np.sqrt(v_r * v_r + v_t * v_t)

    r, _ = rv_from_mee(x, mu)
    rho = drag.atmosphere(csqrt_norm(r) - re)
    coef = 0.5 * rho * drag.cd * drag.area / mass

    out = np.zeros(3, dtype=dt)
    out[0] = -coef * v * v_r
    out[1] = -coef * v * v_t
    return out

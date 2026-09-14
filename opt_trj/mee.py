"""Modified equinoctial elements (MEE) and the control influence matrices.

State (slide 10):

    x = [p, f, g, h, k, L]              (SI: p in metres, L in radians)
    xdot = A(x) a + b(x)

where ``a = [a_r, a_theta, a_h]`` is the *perturbing acceleration expressed in
the LVLH / RTN frame*  (radial, transverse, orbit-normal) and

    q     = 1 + f cos L + g sin L
    s^2   = 1 + h^2 + k^2
    chi   = sqrt(h^2 + k^2)

``Q_r = [i_r  i_theta  i_h]`` is the 3x3 rotation LVLH -> ECI, so an ECI
acceleration is mapped to the LVLH components used by ``A`` with ``Q_r.T``.

Sign convention note
--------------------
The entries below are the standard Walker/Betts Gauss-variational form.  They
agree with the slide except for two entries where the slide appears to carry a
typo (``a_{2,3}`` uses ``h sinL + k cosL`` and ``a_{3,1}`` is written ``+cos L``);
the signs used here are the ones that reproduce the true Cartesian dynamics and
are verified numerically by :func:`opt_trj.validate.check_A_b`.

Every routine here is written to be *complex-step safe* (no ``abs``, no
``np.linalg.norm``, no ``arctan2`` on the differentiated path) so that exact
Jacobians can be obtained with a complex perturbation.
"""

import numpy as np

from .constants import MU

__all__ = [
    "csqrt_norm", "aux", "A_matrix", "b_vector",
    "rv_from_mee", "mee_from_rv", "lvlh_frame", "mee_from_coe",
    "period", "dcart_dmee",
]


def csqrt_norm(v):
    """Euclidean norm, complex-step safe (``np.linalg.norm`` is not)."""
    return np.sqrt(np.sum(v * v))


def aux(x):
    """Return the auxiliary scalars ``(q, s2, chi2, w)`` used throughout."""
    p, f, g, h, k, L = x
    cL, sL = np.cos(L), np.sin(L)
    q = 1.0 + f * cL + g * sL
    s2 = 1.0 + h * h + k * k
    chi2 = h * h + k * k
    return q, s2, chi2, (cL, sL)


# ---------------------------------------------------------------------------
# Gauss variational equations
# ---------------------------------------------------------------------------
def A_matrix(x, mu=MU):
    """6x3 control-influence matrix ``A(x)`` (slide 10).

    Columns correspond to ``a = [a_r, a_theta, a_h]`` in the LVLH frame.
    """
    p, f, g, h, k, L = x
    q, s2, _, (cL, sL) = aux(x)
    sp = np.sqrt(p / mu)
    hk = h * sL - k * cL          # recurring combination

    A = np.zeros((6, 3), dtype=np.result_type(x[0], 0.0))
    # dp/dt
    A[0, 1] = 2.0 * p / q * sp
    # df/dt
    A[1, 0] = sp * sL
    A[1, 1] = sp / q * ((q + 1.0) * cL + f)
    A[1, 2] = -sp * g / q * hk
    # dg/dt
    A[2, 0] = -sp * cL
    A[2, 1] = sp / q * ((q + 1.0) * sL + g)
    A[2, 2] = sp * f / q * hk
    # dh/dt
    A[3, 2] = sp * s2 * cL / (2.0 * q)
    # dk/dt
    A[4, 2] = sp * s2 * sL / (2.0 * q)
    # dL/dt (perturbing part)
    A[5, 2] = sp / q * hk
    return A


def b_vector(x, mu=MU):
    """6-vector ``b(x)`` -- the unperturbed (Keplerian) drift, slide 10."""
    p = x[0]
    q, _, _, _ = aux(x)
    b = np.zeros(6, dtype=np.result_type(x[0], 0.0))
    b[5] = np.sqrt(mu * p) * (q / p) ** 2
    return b


def period(x, mu=MU):
    """Orbital period [s] of the osculating orbit."""
    p, f, g = x[0], x[1], x[2]
    a = p / (1.0 - f * f - g * g)
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


# ---------------------------------------------------------------------------
# MEE <-> Cartesian
# ---------------------------------------------------------------------------
def rv_from_mee(x, mu=MU):
    """Inertial position/velocity from MEE.  Complex-step safe."""
    p, f, g, h, k, L = x
    cL, sL = np.cos(L), np.sin(L)
    alpha2 = h * h - k * k
    s2 = 1.0 + h * h + k * k
    w = 1.0 + f * cL + g * sL
    r = p / w
    sq = np.sqrt(mu / p)

    dt = np.result_type(x[0], 0.0)
    rv = np.zeros(3, dtype=dt)
    vv = np.zeros(3, dtype=dt)

    rv[0] = r / s2 * (cL + alpha2 * cL + 2.0 * h * k * sL)
    rv[1] = r / s2 * (sL - alpha2 * sL + 2.0 * h * k * cL)
    rv[2] = 2.0 * r / s2 * (h * sL - k * cL)

    vv[0] = -sq / s2 * (sL + alpha2 * sL - 2.0 * h * k * cL
                        + g - 2.0 * f * h * k + alpha2 * g)
    vv[1] = -sq / s2 * (-cL + alpha2 * cL + 2.0 * h * k * sL
                        - f + 2.0 * g * h * k + alpha2 * f)
    vv[2] = 2.0 * sq / s2 * (h * cL + k * sL + f * h + g * k)
    return rv, vv


def mee_from_rv(r, v, mu=MU):
    """MEE from inertial position/velocity (uses ``arctan2``: real input only)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    hv = np.cross(r, v)
    hn = np.linalg.norm(hv)
    p = hn * hn / mu
    hhat = hv / hn

    h = -hhat[1] / (1.0 + hhat[2])
    k = hhat[0] / (1.0 + hhat[2])

    # equinoctial in-plane basis, read directly off :func:`rv_from_mee`
    s2 = 1.0 + h * h + k * k
    alpha2 = h * h - k * k
    fhat = np.array([1.0 + alpha2, 2.0 * h * k, -2.0 * k]) / s2
    ghat = np.array([2.0 * h * k, 1.0 - alpha2, 2.0 * h]) / s2

    evec = np.cross(v, hv) / mu - r / rn
    f = float(evec @ fhat)
    g = float(evec @ ghat)
    L = float(np.arctan2(r @ ghat, r @ fhat)) % (2.0 * np.pi)
    return np.array([p, f, g, h, k, L])


def lvlh_frame(r, v):
    """Rotation matrix ``Q_r = [i_r i_theta i_h]`` mapping LVLH -> ECI."""
    ir = r / csqrt_norm(r)
    hv = np.cross(r, v)
    ih = hv / csqrt_norm(hv)
    it = np.cross(ih, ir)
    return np.stack([ir, it, ih], axis=1)


def mee_from_coe(a, e, inc, raan, argp, nu):
    """MEE from classical elements (angles in radians, ``a`` in metres)."""
    p = a * (1.0 - e * e)
    f = e * np.cos(argp + raan)
    g = e * np.sin(argp + raan)
    h = np.tan(inc / 2.0) * np.cos(raan)
    k = np.tan(inc / 2.0) * np.sin(raan)
    L = (raan + argp + nu) % (2.0 * np.pi)
    return np.array([p, f, g, h, k, L])


# ---------------------------------------------------------------------------
# Jacobian of the MEE -> Cartesian map (needed to convert costates)
# ---------------------------------------------------------------------------
def dcart_dmee(x, mu=MU, step=1e-30):
    """``d[r;v] / d[p,f,g,h,k,L]`` (6x6) by complex-step differentiation."""
    J = np.zeros((6, 6))
    for j in range(6):
        xp = np.array(x, dtype=complex)
        xp[j] = xp[j] + 1j * step
        rr, vv = rv_from_mee(xp, mu)
        J[0:3, j] = np.imag(rr) / step
        J[3:6, j] = np.imag(vv) / step
    return J

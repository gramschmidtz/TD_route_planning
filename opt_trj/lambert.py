"""Universal-variable Lambert solver (Vallado / Battin).

Used only to *seed* the two-impulse solution; the arc is then re-converged
against the full perturbed force model in :mod:`opt_trj.twoimpulse`.
"""

import numpy as np

from .constants import MU

__all__ = ["stumpff", "lambert"]


def stumpff(psi):
    """Stumpff functions ``(c2, c3)``."""
    if psi > 1e-6:
        sp = np.sqrt(psi)
        return (1.0 - np.cos(sp)) / psi, (sp - np.sin(sp)) / sp ** 3
    if psi < -1e-6:
        sp = np.sqrt(-psi)
        return (1.0 - np.cosh(sp)) / psi, (np.sinh(sp) - sp) / sp ** 3
    # series around 0
    return (0.5 - psi / 24.0 + psi * psi / 720.0,
            1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0)


def lambert(r1, r2, dt, mu=MU, prograde=True, tol=1e-8, maxiter=200):
    """Solve Lambert's problem for the zero-revolution transfer.

    Parameters
    ----------
    r1, r2 : array_like (3,)
        Departure and arrival position vectors [m].
    dt : float
        Time of flight [s].

    Returns
    -------
    (v1, v2) : ndarray, ndarray
        Velocities on the transfer arc at ``r1`` and ``r2``.  ``(None, None)``
        if the iteration does not converge.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)

    cosdnu = np.clip(r1 @ r2 / (r1n * r2n), -1.0, 1.0)
    cross_z = np.cross(r1, r2)[2]
    if prograde:
        dm = 1.0 if cross_z >= 0.0 else -1.0
    else:
        dm = -1.0 if cross_z >= 0.0 else 1.0

    A = dm * np.sqrt(r1n * r2n * (1.0 + cosdnu))
    if abs(A) < 1e-9:
        return None, None

    psi, psi_up, psi_low = 0.0, 4.0 * np.pi ** 2, -4.0 * np.pi
    c2, c3 = 0.5, 1.0 / 6.0
    y = 0.0
    for _ in range(maxiter):
        y = r1n + r2n + A * (psi * c3 - 1.0) / np.sqrt(c2)
        if A > 0.0 and y < 0.0:
            for _ in range(100):                     # push psi_low up until y > 0
                psi_low += 0.1
                psi = 0.5 * (psi_up + psi_low)
                c2, c3 = stumpff(psi)
                y = r1n + r2n + A * (psi * c3 - 1.0) / np.sqrt(c2)
                if y >= 0.0:
                    break
            if y < 0.0:
                return None, None
        chi = np.sqrt(y / c2)
        dt_try = (chi ** 3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        if abs(dt_try - dt) < tol * max(dt, 1.0):
            break
        if dt_try <= dt:
            psi_low = psi
        else:
            psi_up = psi
        psi = 0.5 * (psi_up + psi_low)
        c2, c3 = stumpff(psi)
    else:
        return None, None

    f = 1.0 - y / r1n
    gdot = 1.0 - y / r2n
    g = A * np.sqrt(y / mu)
    if abs(g) < 1e-12:
        return None, None
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g

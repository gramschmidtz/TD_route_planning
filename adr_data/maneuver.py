"""
Transfer-cost model (only needed to RE-COMPUTE / check the shipped dV tables).

Combined 2-impulse Hohmann between circularised orbits (radius = mean a),
plane change merged vectorially into the cheaper of the two burns at the
mutual nodes N1 / N2 (prompt §4); `split` = 3-impulse comparison option.
Same formulas as external/.../edge_cost_v2.py.
"""
from __future__ import annotations

import numpy as np

MU = 398600.4418                      # km^3/s^2 (same as edge_cost_v2.py)


def hvec(inc_rad, raan_rad):
    return np.stack([np.sin(inc_rad) * np.sin(raan_rad),
                     -np.sin(inc_rad) * np.cos(raan_rad),
                     np.cos(inc_rad)], -1)


def plane_angle_deg(a_i, inc_i_deg, raan_i_deg, a_j, inc_j_deg, raan_j_deg):
    hi = hvec(np.radians(inc_i_deg), np.radians(raan_i_deg))
    hj = hvec(np.radians(inc_j_deg), np.radians(raan_j_deg))
    return np.degrees(np.arccos(np.clip(np.sum(hi * hj, -1), -1, 1)))


def transfer_dv(a_i, inc_i_deg, raan_i_deg, a_j, inc_j_deg, raan_j_deg,
                mode="combined"):
    """dV [m/s] and plane angle theta [deg]; a in km, angles in deg."""
    th = np.radians(plane_angle_deg(a_i, inc_i_deg, raan_i_deg,
                                    a_j, inc_j_deg, raan_j_deg))
    vi, vj = np.sqrt(MU / a_i), np.sqrt(MU / a_j)
    at = (a_i + a_j) / 2
    va = np.sqrt(MU * (2 / a_i - 1 / at))
    vb = np.sqrt(MU * (2 / a_j - 1 / at))
    if mode == "split":
        d = np.abs(va - vi) + np.abs(vj - vb) + 2 * np.minimum(vi, vj) * np.sin(th / 2)
    elif mode == "combined":
        dA = np.sqrt(vi**2 + va**2 - 2 * vi * va * np.cos(th)) + np.abs(vj - vb)
        dB = np.abs(va - vi) + np.sqrt(vb**2 + vj**2 - 2 * vb * vj * np.cos(th))
        d = np.minimum(dA, dB)
    else:
        raise ValueError(mode)
    return 1000.0 * d, np.degrees(th)


def phasing_dv(i: int, j: int, day: float, cfg_maneuver: dict | None = None) -> float:
    """Phasing dV [m/s] added to each debris->debris transfer.

    Currently a constant (default 0: phase is matched by coasting/waiting).
    Hook for Choi et al. (2024) eqs. (22)-(23): replace the body with the
    phasing-orbit dV as a function of (i, j, day) -- the adapter already calls
    this for every arc, so nothing else has to change."""
    return float((cfg_maneuver or {}).get("phasing_dv_mps", 0.0))


def dv_table_from_elements(elems: dict, ids: list, mode="combined"):
    """Rebuild dV[i, j, day] from debris_mean_elements_daily.csv
    (adr_data.load.load_mean_elements output)."""
    n = len(ids)
    n_days = next(iter(elems.values())).shape[0]
    M = np.full((n, n, n_days), np.nan)
    TH = np.full((n, n, n_days), np.nan)
    for p in range(n):
        Ep = elems[ids[p]]
        for q in range(n):
            if p == q:
                continue
            Eq = elems[ids[q]]
            M[p, q], TH[p, q] = transfer_dv(Ep[:, 0], Ep[:, 2], Ep[:, 3],
                                            Eq[:, 0], Eq[:, 2], Eq[:, 3], mode)
    return M, TH


def combined_impulses(a_i, inc_i_deg, raan_i_deg, a_j, inc_j_deg, raan_j_deg):
    """Impulse breakdown of the `combined` transfer (scalar inputs).
    -> dict(theta_deg, plane_change_at 'N1'|'N2', dv1_mps, dv2_mps, total_mps)
    N1 = first (departure) burn, N2 = second (arrival) burn; the plane change is
    merged into whichever gives the smaller total (same rule as transfer_dv)."""
    th = np.radians(float(plane_angle_deg(a_i, inc_i_deg, raan_i_deg,
                                          a_j, inc_j_deg, raan_j_deg)))
    vi, vj = np.sqrt(MU / a_i), np.sqrt(MU / a_j)
    at = (a_i + a_j) / 2
    va = np.sqrt(MU * (2 / a_i - 1 / at))
    vb = np.sqrt(MU * (2 / a_j - 1 / at))
    a1 = np.sqrt(vi**2 + va**2 - 2 * vi * va * np.cos(th)); a2 = abs(vj - vb)
    b1 = abs(va - vi); b2 = np.sqrt(vb**2 + vj**2 - 2 * vb * vj * np.cos(th))
    if a1 + a2 <= b1 + b2:
        where, d1, d2 = "N1", a1, a2
    else:
        where, d1, d2 = "N2", b1, b2
    return {"theta_deg": float(np.degrees(th)), "plane_change_at": where,
            "dv1_mps": float(1000 * d1), "dv2_mps": float(1000 * d2),
            "total_mps": float(1000 * (d1 + d2))}

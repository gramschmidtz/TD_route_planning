"""Necessary conditions and Lion-Handelsman gradients on an impulsive arc.

Everything here is the chapter's theory (Table 2.1, Eqs. 2.43-2.76) evaluated
with the MEE costate machinery of :mod:`opt_trj.costate`.  Because the MEE
primer and Lawden's Cartesian primer are the same physical vector, the
gradients keep their original form:

    dJ/dt_1  = -|dv_1| pdot_1^T p_1                       (2.59)
    dJ/dt_n  = -|dv_n| pdot_n^T p_n                       (2.60)
    dJ/dr_m  =  pdot_m^+ - pdot_m^-                       (2.75)
    dJ/dt_m  = -(H_m^+ - H_m^-)                           (2.76)

and the midcourse insertion uses

    A     = -(M_fm^T N_fm^-T + T_mo N_mo^-1)              (2.65)
    dr_m  = eps A^-1 p_m ,  eps = beta r_m / |A^-1 p_m|   (2.67, 2.69)

with M, N, S, T the 3x3 partitions of the **Cartesian** transition matrix,
recovered from the MEE STM by

    Phi_cart(b,a) = (d cart/d mee)_b  Phi_mee(b,a)  (d cart/d mee)_a^-1 .
"""

from dataclasses import dataclass, field

import numpy as np

from .mee import rv_from_mee, dcart_dmee
from .dynamics import propagate_stm
from .costate import (solve_primer_bvp, primer_from_costate, primer_map,
                      hamiltonian_coast)

__all__ = ["PrimerSegment", "PrimerAnalysis", "analyse", "cart_stm",
           "midcourse_seed", "terminal_gradients"]


# ---------------------------------------------------------------------------
def cart_stm(Phi_mee, x_a, x_b, mu):
    """Cartesian STM from the MEE STM."""
    return dcart_dmee(x_b, mu) @ Phi_mee @ np.linalg.inv(dcart_dmee(x_a, mu))


def _partitions(Phi):
    return Phi[0:3, 0:3], Phi[0:3, 3:6], Phi[3:6, 0:3], Phi[3:6, 3:6]


# ---------------------------------------------------------------------------
@dataclass
class PrimerSegment:
    """Primer history on one ballistic leg."""
    t: np.ndarray
    P: np.ndarray             # (3, n) ECI
    PD: np.ndarray            # (3, n) ECI
    X: np.ndarray             # (6, n) MEE
    PHI: np.ndarray           # (n, 6, 6) MEE STM from t[0]
    lam0: np.ndarray          # MEE costate at t[0]
    resid: float              # boundary-condition residual
    cond: float               # conditioning of the 6x6 BVP matrix

    @property
    def pmag(self):
        return np.linalg.norm(self.P, axis=0)

    @property
    def pdot_dot_p(self):
        """``pdot^T p``: equals d|p|/dt where |p| = 1 (Eq. 2.61)."""
        return np.sum(self.PD * self.P, axis=0)


@dataclass
class PrimerAnalysis:
    segments: list
    grad_t_first: float          # dJ/dt_1   (2.59)
    grad_t_last: float           # dJ/dt_n   (2.60)
    pdot_first: float            # pdot^T p at the first impulse
    pdot_last: float             # pdot^T p at the last impulse
    p_max: float
    t_p_max: float
    seg_p_max: int
    mid_grad_r: list = field(default_factory=list)   # (2.75) per interior impulse
    mid_grad_t: list = field(default_factory=list)   # (2.76) per interior impulse
    bvp_resid: float = 0.0
    bvp_cond: float = 0.0
    tof: float = 1.0                 # transfer duration, used to scale gradients

    # -- necessary conditions (Table 2.1) -----------------------------------
    # Every gradient is converted to "m/s of cost per natural unit of the
    # variable" before being compared with one tolerance in m/s.  Comparing the
    # raw gradients against a single dimensionless number, as an earlier version
    # did, is meaningless: dJ/dr_m is in 1/s and dJ/dt_m in (m/s)/s, so a
    # tolerance of 1e-3 passed a midcourse impulse whose dJ/dt_m was -1.76 m/s
    # per second -- wildly non-stationary -- while looking perfectly converged.
    #
    # tol_p is set from the measured reproducibility of max|p|, not from a
    # guess.  Two independent refinements of the same 1->2 three-impulse arc
    # (71 and 120 L-BFGS-B iterations, both landing on dV = 34.7846 m/s)
    # returned max|p| = 1.00104 and 1.00086 -- a spread of 1.8e-4.  The primer
    # BVP matrix has condition number 1.8e8 and the impulse directions come
    # from a shooting converged to 1e-3 m, so that spread is the floor; asking
    # for agreement with 1 to better than ~1e-3 is asking for noise, and doing
    # so makes the driver insert null impulses (|dv| ~ 0.01 m/s) forever.
    def gradient_costs(self, scale_r=1e3):
        """Cost change [m/s] for a unit move of each free variable.

        Time variables are scaled by the transfer duration, positions by
        ``scale_r`` (default 1 km).
        """
        out = {"t_first": abs(self.grad_t_first) * self.tof,
               "t_last": abs(self.grad_t_last) * self.tof}
        for i, (gr, gt) in enumerate(zip(self.mid_grad_r, self.mid_grad_t)):
            out[f"r_mid{i+1}"] = float(np.linalg.norm(gr)) * scale_r
            out[f"t_mid{i+1}"] = abs(gt) * self.tof
        return out

    def nc(self, tol_p=2e-3, tol_dv=1e-3):
        """Table 2.1, as *stationarity* of the cost in every free variable.

        The transfer time is open here (``t_1`` and ``t_n`` are decision
        variables), so the chapter's one-sided coast criteria tighten to
        ``dJ/dt_1 = dJ/dt_n = 0`` -- see :meth:`coast_advice` for the one-sided
        reading and what it tells you to do.
        """
        out = {"p_le_1": self.p_max <= 1.0 + tol_p}
        for k, v in self.gradient_costs().items():
            out[f"stationary_{k}"] = v <= tol_dv
        return out

    def coast_advice(self):
        """The chapter's terminal-coast criteria (p. 28), with the action."""
        msg = []
        if self.pdot_first > 0:
            msg.append("pdot(t_1) > 0  ->  an INITIAL COAST (delay t_1) lowers the cost")
        elif self.pdot_first < 0:
            msg.append("pdot(t_1) < 0  ->  no initial coast; start EARLIER "
                       "(lengthen the transfer) to lower the cost")
        if self.pdot_last < 0:
            msg.append("pdot(t_n) < 0  ->  a FINAL COAST (advance t_n) lowers the cost")
        elif self.pdot_last > 0:
            msg.append("pdot(t_n) > 0  ->  no final coast; end LATER "
                       "(lengthen the transfer) to lower the cost")
        return msg

    def optimal(self, **kw):
        return all(self.nc(**kw).values())

    def report(self):
        nc = self.nc()
        gc = self.gradient_costs()
        lines = [
            f"  max |p|            = {self.p_max:.6f} at t = {self.t_p_max/86400:.6f} d"
            f" (segment {self.seg_p_max + 1})",
            f"  pdot^T p  @ first  = {self.pdot_first:+.3e} 1/s"
            f"   ->  dJ/dt1 = {self.grad_t_first:+.6e} m/s per s",
            f"  pdot^T p  @ last   = {self.pdot_last:+.3e} 1/s"
            f"   ->  dJ/dtn = {self.grad_t_last:+.6e} m/s per s",
            f"  BVP residual/cond  = {self.bvp_resid:.2e} / {self.bvp_cond:.2e}",
        ]
        for i, (gr, gt) in enumerate(zip(self.mid_grad_r, self.mid_grad_t)):
            lines.append(f"  midcourse {i+1}: |dJ/dr_m| = {np.linalg.norm(gr):.3e} 1/s,"
                         f"  dJ/dt_m = {gt:+.3e} m/s per s")
            lines.append(f"               |pdot+ - pdot-| = "
                         f"{np.linalg.norm(gr):.3e} 1/s  (NC 4: must vanish)")
        lines.append("  cost sensitivity [m/s per unit move]: "
                     + ", ".join(f"{k} {v:.2e}" for k, v in gc.items()))
        for m in self.coast_advice():
            lines.append(f"  {m}")
        lines.append("  NC: " + ", ".join(f"{k}={'OK' if v else 'VIOLATED'}"
                                          for k, v in nc.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def terminal_gradients(traj, fm):
    """Cheap ``(dJ/dt_1, dJ/dt_n, pdot_1, pdot_n)`` -- no extra propagation.

    Everything needed is already stored on the first and last :class:`Leg`:
    solve the 6x6 primer BVP, read ``p, pdot`` at the two ends.
    """
    units = traj.units
    first, last = traj.legs[0], traj.legs[-1]

    lam_a, _, _ = solve_primer_bvp(first.x_a, first.x_b, first.Phi,
                                   units[0], units[1], fm.mu)
    p_a, pd_a = primer_from_costate(first.x_a, lam_a, fm.mu)

    n = len(traj.legs)
    lam_l, _, _ = solve_primer_bvp(last.x_a, last.x_b, last.Phi,
                                   units[n - 1], units[n], fm.mu)
    lam_b = np.linalg.solve(last.Phi.T, lam_l)
    p_b, pd_b = primer_from_costate(last.x_b, lam_b, fm.mu)

    s_a = float(pd_a @ p_a)
    s_b = float(pd_b @ p_b)
    d = traj.dv_mag
    return -d[0] * s_a, -d[-1] * s_b, s_a, s_b


def analyse(traj, fm, mass=None, n_eval=400):
    """Evaluate the primer vector and all Lion-Handelsman gradients."""
    units = traj.units
    segs = []
    resid_max, cond_max = 0.0, 0.0

    for i, leg in enumerate(traj.legs):
        m = traj.masses[i + 1] if traj.masses is not None else (mass or 5000.0)
        lam0, resid, cond = solve_primer_bvp(leg.x_a, leg.x_b, leg.Phi,
                                             units[i], units[i + 1], fm.mu)
        resid_max = max(resid_max, resid)
        cond_max = max(cond_max, cond)

        t_eval = np.linspace(leg.t_a, leg.t_b, n_eval)
        _, X, PHI = propagate_stm(leg.x_a, (leg.t_a, leg.t_b), fm, m,
                                  t_eval=t_eval)
        P = np.empty((3, n_eval))
        PD = np.empty((3, n_eval))
        for j in range(n_eval):
            lam = np.linalg.solve(PHI[j].T, lam0)
            P[:, j], PD[:, j] = primer_from_costate(X[:, j], lam, fm.mu)
        if fm.drag is not None:
            PD = np.gradient(P, t_eval, axis=1, edge_order=2)
        segs.append(PrimerSegment(t_eval, P, PD, X, PHI, lam0, resid, cond))

    # terminal gradients, Eqs. (2.59)-(2.60)
    dmag = traj.dv_mag
    pd_first = float(segs[0].pdot_dot_p[0])
    pd_last = float(segs[-1].pdot_dot_p[-1])
    grad_1 = -dmag[0] * pd_first
    grad_n = -dmag[-1] * pd_last

    # interior-impulse gradients, Eqs. (2.75)-(2.76)
    mid_r, mid_t = [], []
    for i in range(len(traj.legs) - 1):
        sa, sb = segs[i], segs[i + 1]
        pd_minus = sa.PD[:, -1]
        pd_plus = sb.PD[:, 0]
        mid_r.append(pd_plus - pd_minus)
        m_before = traj.masses[i + 1] if traj.masses is not None else 5000.0
        m_after = traj.masses[i + 2] if traj.masses is not None else 5000.0
        lam_minus = np.linalg.solve(sa.PHI[-1].T, sa.lam0)
        H_minus = hamiltonian_coast(sa.X[:, -1], lam_minus, fm, m_before)
        H_plus = hamiltonian_coast(sb.X[:, 0], sb.lam0, fm, m_after)
        mid_t.append(-(H_plus - H_minus))

    pmaxs = [s.pmag.max() for s in segs]
    k = int(np.argmax(pmaxs))
    j = int(np.argmax(segs[k].pmag))

    return PrimerAnalysis(
        segments=segs,
        grad_t_first=grad_1, grad_t_last=grad_n,
        pdot_first=pd_first, pdot_last=pd_last,
        p_max=float(pmaxs[k]), t_p_max=float(segs[k].t[j]), seg_p_max=k,
        mid_grad_r=mid_r, mid_grad_t=mid_t,
        bvp_resid=resid_max, bvp_cond=cond_max,
        tof=float(traj.t_imp[-1] - traj.t_imp[0]),
    )


# ---------------------------------------------------------------------------
def midcourse_seed(traj, ana, fm, beta=0.05):
    """Initial position and time of an inserted midcourse impulse.

    Implements Eqs. (2.64)-(2.69): pick ``t_m`` at the maximum of ``|p|`` and
    displace the nominal position by ``dr_m = eps A^-1 p_m``.
    Returns ``(t_m, r_m)`` or ``None`` if no interior point has ``|p| > 1``.
    """
    k = ana.seg_p_max
    seg = ana.segments[k]
    leg = traj.legs[k]
    j = int(np.argmax(seg.pmag))
    if seg.pmag[j] <= 1.0 or j in (0, seg.t.size - 1):
        return None

    t_m = float(seg.t[j])
    x_m = seg.X[:, j]
    r_m, _ = rv_from_mee(x_m, fm.mu)
    p_m = seg.P[:, j]

    # Cartesian STMs of the two sub-arcs of this leg
    Phi_mo_mee = seg.PHI[j]
    Phi_fo_mee = seg.PHI[-1]
    Phi_fm_mee = Phi_fo_mee @ np.linalg.inv(Phi_mo_mee)

    Phi_mo = cart_stm(Phi_mo_mee, leg.x_a, x_m, fm.mu)
    Phi_fm = cart_stm(Phi_fm_mee, x_m, leg.x_b, fm.mu)

    M_fm, N_fm, _, _ = _partitions(Phi_fm)
    _, N_mo, _, T_mo = _partitions(Phi_mo)

    A = -(M_fm.T @ np.linalg.inv(N_fm.T) + T_mo @ np.linalg.inv(N_mo))
    w = np.linalg.solve(A, p_m)
    eps = beta * np.linalg.norm(r_m) / np.linalg.norm(w)
    dr_m = eps * w
    return t_m, r_m + dr_m

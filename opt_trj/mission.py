"""Mission points, spacecraft data, and the propagated node orbits.

Mission points (the "Mission point [고도, RAAN, Phase]" table)::

        alt [km]   RAAN@t=0 [deg]   True anomaly@t=0 [deg]
    1(O)   500          0.0                116.6
    2      565        341.5                 54.3
    3      630        303.2                234.3
    4      700        242.3                 26.1
    5      775        158.2                192.9
    6      845         63.8                131.6
    7      900        337.8                 20.9
    8(D)   950        249.2                182.7

All eight are **circular**, so ``e = 0``, ``argp = 0`` and the tabulated "true
anomaly" is the argument of latitude at epoch.

Two quantities are NOT in the table and are taken from the project's existing
``config.yaml`` (``orbital.inclination_deg``, ``orbital.t_start``):

* inclination = 51.6 deg, shared by all nodes  -> :data:`INCLINATION_DEG`
* ``t = 0`` is a relative mission epoch; no absolute calendar date is needed
  because the only time-dependent effect modelled is J2 nodal regression.

Each :class:`NodeOrbit` carries two models of the same orbit:

* ``mean_*``  -- secular-J2 mean elements, closed form.  Used only for the
  coarse departure scan, where hundreds of thousands of evaluations are needed
  and a ~0.01 deg RAAN error is irrelevant.
* ``mee(t)``  -- the true osculating state from numerical propagation of the
  full force model, available inside a prepared window via DOP853 dense output.
  Everything downstream of the scan uses this.

Spacecraft (as specified): wet mass 5000 kg, usable propellant 1500 kg,
T_max = 490 N, Isp = 315.5 s, hence c = 3094.4 m/s and dV budget = 1103.6 m/s.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import MU, R_E, G0, DEG, DAY, J_ZONAL
from .mee import mee_from_coe, rv_from_mee, period
from .dynamics import ForceModel, propagate, propagate_dense

__all__ = ["INCLINATION_DEG", "MISSION_POINTS", "Spacecraft", "SPACECRAFT",
           "secular_rates", "raan_window", "NodeOrbit", "build_nodes"]

_CACHE_DIR = Path(__file__).with_name("_cache")

# --- taken from config.yaml (orbital.inclination_deg); NOT in the table ------
INCLINATION_DEG = 51.6

# id -> (altitude [km], RAAN@t=0 [deg], true anomaly@t=0 [deg])
MISSION_POINTS = {
    1: (500.0, 0.0, 116.6),
    2: (565.0, 341.5, 54.3),
    3: (630.0, 303.2, 234.3),
    4: (700.0, 242.3, 26.1),
    5: (775.0, 158.2, 192.9),
    6: (845.0, 63.8, 131.6),
    7: (900.0, 337.8, 20.9),
    8: (950.0, 249.2, 182.7),
}


@dataclass
class Spacecraft:
    wet_mass: float = 5000.0        # [kg]
    fuel_mass: float = 1500.0       # [kg]
    thrust_max: float = 490.0       # [N]
    isp: float = 315.5              # [s]

    @property
    def c(self):
        """Effective exhaust velocity ``Isp g0`` [m/s]."""
        return self.isp * G0

    @property
    def dv_budget(self):
        """Tsiolkovsky dV budget for the full propellant load [m/s]."""
        return self.c * np.log(self.wet_mass / (self.wet_mass - self.fuel_mass))

    @property
    def mdot(self):
        """Mass flow at full thrust [kg/s]."""
        return self.thrust_max / self.c

    def gamma_max(self, mass):
        """Maximum thrust acceleration at a given mass [m/s^2]."""
        return self.thrust_max / mass

    def burn_time(self, dv, mass):
        """Finite-burn duration [s] for an impulse ``dv`` starting at ``mass``."""
        return mass * (1.0 - np.exp(-dv / self.c)) / self.mdot

    def mass_after(self, dv, mass):
        return mass * np.exp(-dv / self.c)


SPACECRAFT = Spacecraft()


# ---------------------------------------------------------------------------
def secular_rates(a, e, inc, mu=MU, re=R_E, j2=None):
    """Secular J2 rates ``(raan_dot, argp_dot, M_dot)`` [rad/s]."""
    j2 = J_ZONAL[2] if j2 is None else j2
    n = np.sqrt(mu / a ** 3)
    p = a * (1.0 - e * e)
    kk = 1.5 * j2 * (re / p) ** 2 * n
    raan_dot = -kk * np.cos(inc)
    argp_dot = 0.5 * kk * (5.0 * np.cos(inc) ** 2 - 1.0)
    m_dot = n + 0.5 * kk * np.sqrt(1.0 - e * e) * (3.0 * np.cos(inc) ** 2 - 1.0)
    return raan_dot, argp_dot, m_dot


def raan_window(node_o, node_f, t_max):
    """Epochs in ``[0, t_max]`` where the two RAANs coincide (J2 secular).

    The relative node line drifts linearly, so the crossings are exact in the
    mean model.  Unlike the *phase*, RAAN is safe to predict this way: the mean
    and osculating node rates differ by a few parts in 1e4, which after 120 days
    is a hundredth of a degree -- a metre per second of plane change.
    """
    d0 = (node_f.raan0 - node_o.raan0 + np.pi) % (2 * np.pi) - np.pi
    ddot = node_f.raan_dot - node_o.raan_dot
    if abs(ddot) < 1e-14:
        return []
    out = []
    k = int(np.floor(-abs(t_max * ddot) / (2 * np.pi))) - 1
    while True:
        t = (-d0 + 2.0 * np.pi * k) / ddot
        k += 1
        if t > t_max:
            break
        if 0.0 <= t <= t_max:
            out.append(float(t))
        if k > 10000:
            break
    return sorted(out)


# ---------------------------------------------------------------------------
class NodeOrbit:
    """A mission point: a circular reference orbit the chaser departs/arrives on."""

    def __init__(self, node_id, fm, mass=5000.0,
                 inclination_deg=INCLINATION_DEG):
        alt_km, raan_deg, nu_deg = MISSION_POINTS[node_id]
        self.node_id = node_id
        self.fm = fm
        self.mass = mass
        self.alt = alt_km * 1e3
        self.a = R_E + self.alt
        self.inc = inclination_deg * DEG
        self.raan0 = raan_deg * DEG
        self.u0 = nu_deg * DEG                    # argument of latitude at epoch
        self.x0 = mee_from_coe(self.a, 0.0, self.inc, self.raan0, 0.0, self.u0)
        self.period = period(self.x0, fm.mu)

        rd, ad, md = secular_rates(self.a, 0.0, self.inc, fm.mu, fm.re)
        self.raan_dot = rd                        # [rad/s]
        self.u_dot = ad + md                      # argument-of-latitude rate

        self._t_lo = None
        self._t_hi = None
        self._dense = None
        self._anchor = None

    # -- analytic (secular J2) model, for the coarse scan --------------------
    def mean_mee(self, t):
        raan = self.raan0 + self.raan_dot * t
        u = self.u0 + self.u_dot * t
        return mee_from_coe(self.a, 0.0, self.inc, raan, 0.0, u)

    def mean_rv(self, t):
        return rv_from_mee(self.mean_mee(t), self.fm.mu)

    def mean_raan_deg(self, t):
        return np.degrees((self.raan0 + self.raan_dot * t) % (2 * np.pi))

    # -- exact osculating model ---------------------------------------------
    def _cache_path(self, t_lo):
        key = (f"n{self.node_id}_i{self.inc:.10f}_z{self.fm.zonal_degree}"
               f"_d{int(self.fm.drag is not None)}_t{t_lo:.6f}")
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        return _CACHE_DIR / f"anchor_{self.node_id}_{h}.npy"

    def prepare(self, t_lo, t_hi, verbose=False, use_cache=True):
        """Numerically propagate 0 -> ``t_lo`` once, then keep dense output.

        After this, :meth:`mee` is exact and cheap anywhere in
        ``[t_lo, t_hi]``.  The long 0 -> ``t_lo`` leg (~0.4 s per simulated day)
        is cached on disk, so re-running a study is fast.
        """
        path = self._cache_path(t_lo)
        x_lo = None
        if use_cache and path.exists():
            x_lo = np.load(path)
            if verbose:
                print(f"      node {self.node_id}: anchor at {t_lo/DAY:.3f} d "
                      f"loaded from cache", flush=True)
        if x_lo is None:
            if verbose:
                print(f"      node {self.node_id}: propagating 0 -> "
                      f"{t_lo/DAY:.3f} d ...", end="", flush=True)
            if t_lo > 0.0:
                _, X = propagate(self.x0, (0.0, t_lo), self.fm, self.mass,
                                 t_eval=[t_lo])
                x_lo = X[:, -1]
            else:
                x_lo = self.x0.copy()
            if use_cache:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(path, x_lo)
            if verbose:
                print(" done", flush=True)
        self._anchor = (t_lo, x_lo)
        self._dense = propagate_dense(x_lo, (t_lo, t_hi), self.fm, self.mass)
        self._t_lo, self._t_hi = t_lo, t_hi
        if verbose:
            print(f"      node {self.node_id}: dense window "
                  f"[{t_lo/DAY:.3f}, {t_hi/DAY:.3f}] d ready", flush=True)
        return self

    def reprepare(self, t_hi, verbose=False):
        """Re-run only the dense window, reusing the anchor already computed."""
        if self._anchor is None:
            raise RuntimeError("call prepare() first")
        t_lo, x_lo = self._anchor
        self._dense = propagate_dense(x_lo, (t_lo, t_hi), self.fm, self.mass)
        self._t_hi = t_hi
        if verbose:
            print(f"      node {self.node_id}: dense window extended to "
                  f"{t_hi/DAY:.3f} d", flush=True)
        return self

    def mee(self, t):
        """Exact osculating MEE state at ``t`` (window must be prepared)."""
        if self._dense is None:
            raise RuntimeError("call NodeOrbit.prepare(t_lo, t_hi) first")
        if not (self._t_lo - 1e-6 <= t <= self._t_hi + 1e-6):
            raise ValueError(f"t = {t/DAY:.4f} d outside prepared window "
                             f"[{self._t_lo/DAY:.4f}, {self._t_hi/DAY:.4f}] d")
        return np.asarray(self._dense(t), float)

    def rv(self, t):
        return rv_from_mee(self.mee(t), self.fm.mu)

    def raan_deg(self, t):
        x = self.mee(t)
        return float(np.degrees(np.arctan2(x[4], x[3]) % (2 * np.pi)))

    # backwards-compatible alias used by the transfer solver
    def refine(self, t, mass=None):
        return self.mee(t)

    def __repr__(self):
        return (f"NodeOrbit({self.node_id}: alt = {self.alt/1e3:.0f} km, "
                f"T = {self.period/60:.2f} min, "
                f"RAANdot = {np.degrees(self.raan_dot)*86400:+.4f} deg/day)")


def build_nodes(ids, fm, **kw):
    """``{id: NodeOrbit}`` for the requested mission points."""
    return {i: NodeOrbit(i, fm, **kw) for i in ids}

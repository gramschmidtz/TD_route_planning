"""
Time-dependent ORBITAL transfer network (space analogue of network.py).

This is the space-domain twin of the road ``TDNetwork``: it exposes the same
two time-dependent arc functions the solvers consume,

    travel_time(i, j, t)  ->  Time of Flight (ToF) of the transfer i -> j
    preference(i, j, t)   ->  -Delta V   (less propellant is better)

but the underlying physics is orbital mechanics instead of a road-speed model.

Scenario (confirmed modelling choices)
--------------------------------------
* Each NODE is a circular orbit ("point") defined by an ALTITUDE and a RAAN
  (Omega); all nodes share one INCLINATION, and circularity means the argument
  of perigee is undefined.  A point is "reached" when the chaser's orbit
  matches it in (altitude, RAAN) -- phase is ignored (Sec. 2.2).
* Semi-analytic arc cost (Edelbaum + J2), NOT full MEE numerical propagation.
* The chaser WAITS at the origin node's own orbit; the natural J2 nodal
  precession makes the two planes' relative RAAN drift, and departing at the
  right clock time t makes the residual plane change small.  "Waiting" is thus
  encoded purely by the choice of departure time t (exactly like waiting at a
  node in the road model) -- there is no separate phasing orbit.
* Pure orbit-to-orbit rendezvous: no ADR de-orbit / disposal phase.

Time-dependence -- why departure time matters
---------------------------------------------
Under J2 the RAAN of a circular orbit precesses at

    Omega_dot(a) = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i)          (e=0)

which depends on the semi-major axis a = Re + altitude.  Two nodes at different
altitudes precess at DIFFERENT rates, so their relative RAAN

    dOmega(t) = Omega_j(t) - Omega_i(t),   Omega_x(t) = Omega_x0 + Omega_dot_x * t

sweeps through 0 with synodic period 360 deg / |Omega_dot_j - Omega_dot_i|.
The plane angle the maneuver must rotate through is

    cos(theta) = cos^2(i) + sin^2(i) * cos(dOmega)                         (>=0)

(theta = angle between the two orbit-normal vectors), and Edelbaum's optimal
low-thrust circle-to-circle transfer with a combined plane change costs

    dV = sqrt(V_i^2 + V_j^2 - 2 V_i V_j cos( (pi/2) * theta ) ),  V = sqrt(mu/a).

So dV(t) is minimal (theta -> 0 => dV -> |V_i - V_j|, pure altitude change) at
the RAAN-alignment windows and maximal (theta max) half a synodic period away.
Waiting for a window is what saves propellant.

ToF and propellant
------------------
The continuous-thrust burn time that delivers dV follows the rocket equation
with constant exhaust speed c = Isp * g0 and constant mass flow mdot = T/c:

    dm  = m0 * (1 - exp(-dV / c))          propellant burned
    ToF = dm / mdot = (m0 * c / T) * (1 - exp(-dV / c))

dV is mass-independent, so preference = -dV is history-free.  ToF depends on the
current mass m0; for a stateless arc cost we evaluate it at a fixed reference
mass ``ref_mass_kg`` (default = wet mass) exactly as the road model's travel
time is history-free.  The true burn time at any point in a mission scales with
the actual current mass.

Clock convention
----------------
``t`` is elapsed time from the mission epoch, in DAYS (orbital drift and
low-thrust transfers both live on day-to-month timescales).  ToF is returned in
days as well.  dV is in metres per second.

Modelling simplification
------------------------
The plane change is driven by the relative RAAN at DEPARTURE, dOmega(t).  Both
orbits keep drifting during the (long) transfer; accounting for that lead angle
would make ToF and dOmega mutually dependent (a fixed point).  The
departure-time model keeps arc cost an explicit function of t and makes the
alignment windows directly readable on the plots; drift-during-transfer is left
as a refinement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

# --------------------------------------------------------------------------
# Physical constants (Earth, SI-ish: km, s, kg)
# --------------------------------------------------------------------------
MU = 398600.4418          # km^3 / s^2   Earth gravitational parameter
RE = 6378.137             # km           Earth equatorial radius
J2 = 1.08262668e-3        # -            second zonal harmonic
G0 = 9.80665              # m / s^2      standard gravity (for Isp)

SECONDS_PER_DAY = 86400.0
DEG = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def wrap180(deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    x = (deg + 180.0) % 360.0 - 180.0
    return x + 360.0 if x <= -180.0 else x


@dataclass
class OrbitalNetwork:
    """A time-dependent orbital transfer network (space twin of TDNetwork).

    Nodes carry ``alt`` (altitude, km) and ``raan0`` (RAAN at epoch, deg).
    The graph is directed; a fully connected instance is built by
    :func:`td_route.orbital_instance.build_orbital_network`.
    """

    # --- spacecraft (chaser) -------------------------------------------------
    thrust_N: float = 490.0           # max thrust, N
    isp_s: float = 315.5              # specific impulse, s
    wet_mass_kg: float = 5000.0       # initial (wet) mass, kg
    fuel_mass_kg: float = 1500.0      # usable propellant, kg
    ref_mass_kg: float | None = None  # reference mass for ToF; None -> wet mass
    # "impulsive"  -> 2-impulse Hohmann + plane change, TRUE-ANOMALY matched
    # "low_thrust" -> continuous Edelbaum spiral (the legacy model)
    propulsion: str = "impulsive"
    phase_tol_deg: float = 0.05       # |phase error| accepted as "matched"

    # --- constellation -------------------------------------------------------
    inclination_deg: float = 51.6     # shared orbital inclination

    # --- horizon (days from epoch) ------------------------------------------
    t_start: float = 0.0
    t_end: float = 730.0              # 2 years
    dt_days: float = 0.5              # discretisation step (see choose_dt)
    transit_drift: bool = True        # account for RAAN drift DURING the transfer

    def __post_init__(self):
        self.G = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # Derived spacecraft quantities
    # ------------------------------------------------------------------ #
    @property
    def dry_mass_kg(self) -> float:
        return self.wet_mass_kg - self.fuel_mass_kg

    @property
    def exhaust_speed(self) -> float:
        """Effective exhaust velocity c = Isp * g0  [m/s]."""
        return self.isp_s * G0

    @property
    def mdot(self) -> float:
        """Propellant mass flow rate  mdot = T / c  [kg/s] (constant)."""
        return self.thrust_N / self.exhaust_speed

    @property
    def ref_mass(self) -> float:
        return self.ref_mass_kg if self.ref_mass_kg is not None else self.wet_mass_kg

    @property
    def dv_budget(self) -> float:
        """Total mission dV available from the full propellant load [m/s]
        (Tsiolkovsky, wet -> dry)."""
        return self.exhaust_speed * math.log(self.wet_mass_kg / self.dry_mass_kg)

    # ------------------------------------------------------------------ #
    # Per-node orbital quantities
    # ------------------------------------------------------------------ #
    def sma(self, v) -> float:
        """Semi-major axis a = Re + altitude  [km]."""
        return RE + self.G.nodes[v]["alt"]

    def circ_speed(self, v) -> float:
        """Circular orbital speed V = sqrt(mu / a)  [km/s]."""
        return math.sqrt(MU / self.sma(v))

    def mean_motion(self, v) -> float:
        """Mean motion n = sqrt(mu / a^3)  [rad/s]."""
        a = self.sma(v)
        return math.sqrt(MU / a**3)

    def raan_rate(self, v) -> float:
        """J2 nodal precession rate Omega_dot  [deg/day]  (circular, e=0).

        Omega_dot = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i).
        Negative for prograde orbits (i < 90 deg): the node regresses westward.
        """
        a = self.sma(v)
        i = self.inclination_deg * DEG
        rate_rad_s = -1.5 * J2 * math.sqrt(MU) * RE**2 / a**3.5 * math.cos(i)
        return rate_rad_s * SECONDS_PER_DAY * RAD2DEG

    def raan_at(self, v, t: float) -> float:
        """RAAN of node v's orbit at elapsed time t (days)  [deg]."""
        return self.G.nodes[v]["raan0"] + self.raan_rate(v) * t

    # ------------------------------------------------------------------ #
    # In-plane phase (true anomaly / argument of latitude), impulsive model
    # ------------------------------------------------------------------ #
    def phase_rate(self, v) -> float:
        """Mean motion in deg/day (circular => true anomaly advances at n)."""
        return self.mean_motion(v) * SECONDS_PER_DAY * RAD2DEG

    def phase_at(self, v, t: float) -> float:
        """Argument of latitude u_v(t) [deg], measured from the ascending node.
        Circular orbit => u = theta0 + n t (the AOP is undefined, so this single
        angle fixes the object's position in its plane)."""
        return self.G.nodes[v].get("theta0", 0.0) + self.phase_rate(v) * t

    # ------------------------------------------------------------------ #
    # Impulsive transfer: 2-impulse Hohmann + plane change
    # ------------------------------------------------------------------ #
    def hohmann(self, i, j) -> dict:
        """Coplanar Hohmann elements between the two circular orbits.

        Returns speeds at both ends plus the transfer time (HALF the period of
        the transfer ellipse), all independent of the plane change and of t:
            {"dv1", "dv2" [m/s, coplanar], "tof_days", "vi","vj","vti","vtj"}.
        """
        ri, rj = self.sma(i), self.sma(j)
        at = 0.5 * (ri + rj)
        vi = math.sqrt(MU / ri) * 1000.0                 # m/s
        vj = math.sqrt(MU / rj) * 1000.0
        vti = math.sqrt(MU * (2.0 / ri - 1.0 / at)) * 1000.0   # on transfer @ ri
        vtj = math.sqrt(MU * (2.0 / rj - 1.0 / at)) * 1000.0   # on transfer @ rj
        tof = math.pi * math.sqrt(at**3 / MU) / SECONDS_PER_DAY  # days
        return {"dv1": abs(vti - vi), "dv2": abs(vj - vtj), "tof_days": tof,
                "vi": vi, "vj": vj, "vti": vti, "vtj": vtj,
                "ri": ri, "rj": rj, "at": at}

    def _impulsive_dv(self, i, j, theta: float) -> tuple:
        """(dv1, dv2) [m/s] of the 2-impulse transfer whose plane change is
        theta [rad].  The rotation is folded into the burn at the LARGER radius
        (lower speed), which is where a plane change is cheapest."""
        h = self.hohmann(i, j)
        if h["rj"] >= h["ri"]:            # ascending: apoapsis at rj -> 2nd burn
            dv1 = abs(h["vti"] - h["vi"])
            dv2 = math.sqrt(max(0.0, h["vtj"]**2 + h["vj"]**2
                                - 2.0 * h["vtj"] * h["vj"] * math.cos(theta)))
        else:                             # descending: apoapsis at ri -> 1st burn
            dv1 = math.sqrt(max(0.0, h["vti"]**2 + h["vi"]**2
                                - 2.0 * h["vti"] * h["vi"] * math.cos(theta)))
            dv2 = abs(h["vj"] - h["vtj"])
        return dv1, dv2

    def phase_delay(self, i, j, t: float) -> float:
        """Wait [days] from t until the FIRST instant the phase permits a
        rendezvous, i.e. the Hohmann arrival lands exactly on target j.

        A Hohmann transfer sweeps 180 deg, so departing at tau puts the chaser
        at u_i(tau) + 180 on arrival at tau + ToF, and rendezvous needs

            u_i(tau) + 180  ==  u_j(tau + ToF)   (mod 360).

        With u_x(t) = theta0_x + n_x t this is linear in tau, so the phase error
        sweeps through zero every 360/|n_j - n_i| days (hours for LEO pairs) and
        the smallest non-negative solution is returned.  ToF does NOT depend on
        the plane change, so there is no fixed point to iterate.
        """
        tof = self.hohmann(i, j)["tof_days"]
        ni, nj = self.phase_rate(i), self.phase_rate(j)
        k = nj - ni                                       # deg/day
        # phase error at tau:  A + k*tau  (mod 360)
        A = (self.G.nodes[j].get("theta0", 0.0) + nj * tof
             - self.G.nodes[i].get("theta0", 0.0) - 180.0)
        if abs(k) < 1e-12:                                # co-altitude: never drifts
            return 0.0 if abs(wrap180(A)) <= self.phase_tol_deg else math.inf
        psi = -(A + k * t)                                # need k*delta == psi
        return ((psi if k > 0 else -psi) % 360.0) / abs(k)

    def phase_error(self, i, j, t: float) -> float:
        """Rendezvous phase error [deg] for a departure exactly at t (0 = the
        arrival lands on the target)."""
        tof = self.hohmann(i, j)["tof_days"]
        return wrap180(self.phase_at(j, t + tof) - self.phase_at(i, t) - 180.0)

    def mean_drift_rate(self, i, j) -> float:
        """Effective RAAN drift rate the chaser feels DURING the i->j transfer
        [deg/day].

        While spiralling the semi-major axis from a_i to a_j the chaser's J2
        rate Omega_dot(a) = K * a^(-7/2) varies.  Assuming a varies linearly
        with time over the transfer, the time-average is the a-average

            <Omega_dot> = (1/(a_j - a_i)) * integral_{a_i}^{a_j} K a^(-7/2) da
                        = K * (-2/5) * (a_j^(-5/2) - a_i^(-5/2)) / (a_j - a_i).

        (For a_i == a_j it reduces to the single-orbit rate.)
        """
        ai, aj = self.sma(i), self.sma(j)
        inc = self.inclination_deg * DEG
        K = -1.5 * J2 * math.sqrt(MU) * RE**2 * math.cos(inc)      # rad/s * km^3.5
        if abs(aj - ai) < 1e-9:
            rate_rad_s = K * ai**-3.5
        else:
            rate_rad_s = K * (-0.4) * (aj**-2.5 - ai**-2.5) / (aj - ai)
        return rate_rad_s * SECONDS_PER_DAY * RAD2DEG

    # ------------------------------------------------------------------ #
    # Transfer geometry
    # ------------------------------------------------------------------ #
    def rel_raan(self, i, j, t: float) -> float:
        """Relative RAAN at DEPARTURE, dOmega = Omega_j(t) - Omega_i(t),
        wrapped to (-180, 180]."""
        return wrap180(self.raan_at(j, t) - self.raan_at(i, t))

    def _plane_angle(self, draan_deg: float) -> float:
        """Angle theta between two shared-inclination planes whose RAANs differ
        by draan_deg [rad].  cos(theta) = cos^2(i) + sin^2(i) cos(draan)."""
        inc = self.inclination_deg * DEG
        c = math.cos(inc)**2 + math.sin(inc)**2 * math.cos(draan_deg * DEG)
        return math.acos(max(-1.0, min(1.0, c)))

    def _edelbaum_dv(self, i, j, theta: float) -> float:
        """Edelbaum combined altitude + plane-change dV [m/s] for plane angle
        theta.  dV = sqrt(V_i^2 + V_j^2 - 2 V_i V_j cos((pi/2) theta))."""
        vi = self.circ_speed(i) * 1000.0
        vj = self.circ_speed(j) * 1000.0
        val = vi*vi + vj*vj - 2.0*vi*vj*math.cos(0.5*math.pi*theta)
        return math.sqrt(max(0.0, val))

    def _burn_days(self, dv: float) -> float:
        """Continuous-thrust burn time [days] delivering dV at the reference
        mass.  ToF = (m0 c / T)(1 - exp(-dV/c))."""
        dm = self.ref_mass * (1.0 - math.exp(-dv / self.exhaust_speed))
        return (dm / self.mdot) / SECONDS_PER_DAY

    def plane_angle(self, i, j, t: float) -> float:
        """Plane angle theta the maneuver actually performs at departure t
        [rad] (departure-only geometry; see transfer() for transit drift)."""
        return self._plane_angle(self.rel_raan(i, j, t))

    def _transfer_impulsive(self, i, j, t: float) -> dict:
        """Impulsive arc cost: wait for the phase, then a 2-impulse Hohmann whose
        plane change absorbs whatever RAAN offset the departure instant leaves.

        The leg is  [t -> t+delta]  waiting on orbit i for the rendezvous phase,
        then [t+delta -> t+delta+ToF] on the transfer ellipse.  Planes are matched
        at ARRIVAL, so the rotation the burn must perform follows from
        dOmega = Omega_j(tau+ToF) - Omega_i(tau) -- J2 keeps drifting the two
        nodes apart, so WHEN you leave sets the plane-change bill (waiting for a
        window is cheap; leaving at the wrong time is not).
        """
        h = self.hohmann(i, j)
        tof = h["tof_days"]
        delta = self.phase_delay(i, j, t)
        if not math.isfinite(delta):                # phase can never align
            return {"dv": self.dv_budget, "tof": tof, "theta": math.nan,
                    "draan_thrust": math.nan, "draan_dep": math.nan,
                    "phase_wait": math.inf, "hohmann_tof": tof,
                    "dv1": math.nan, "dv2": math.nan, "feasible": False}
        tau = t + delta                             # phase-matched departure
        draan = self.raan_at(j, tau + tof) - self.raan_at(i, tau)
        theta = self._plane_angle(draan)
        dv1, dv2 = self._impulsive_dv(i, j, theta)
        dv = dv1 + dv2
        return {"dv": dv, "tof": delta + tof, "theta": theta,
                "draan_thrust": wrap180(draan), "draan_dep": wrap180(draan),
                "phase_wait": delta, "hohmann_tof": tof,
                "dv1": dv1, "dv2": dv2, "feasible": dv <= self.dv_budget}

    # ------------------------------------------------------------------ #
    # Arc cost solver (dV, ToF) with optional transit-drift self-consistency
    # ------------------------------------------------------------------ #
    def transfer(self, i, j, t: float, transit_drift: bool | None = None,
                 grid: int = 400) -> dict:
        """Solve one transfer i -> j departing at elapsed time t.

        Returns {"dv" [m/s], "tof" [days], "theta" [rad],
                 "draan_thrust" [deg], "draan_dep" [deg], "feasible": bool}.

        transit_drift (default = self.transit_drift):
          * False -- the plane change closes the DEPARTURE relative RAAN
            dOmega(t); ToF is then an explicit function of t.
          * True  -- rendezvous requires the planes to match at ARRIVAL
            (Sec. 2.2: Omega_SC(t_f) = Omega_j(t_f)).  During the transfer the
            target precesses at Omega_dot_j while the chaser precesses at its
            transit mean rate <Omega_dot>_ij (mean_drift_rate), so the RAAN the
            thrust must actually close is

                dOmega_thrust(ToF) = dOmega(t) + (Omega_dot_j - <Omega_dot>_ij)*ToF,

            and, with continuous max thrust, ToF is itself the burn time of the
            resulting dV -- a self-consistency  ToF = burn(dV(theta(dOmega_thrust))).
            Because a low-thrust ToF (tens-hundreds of days) is a large fraction
            of the synodic period, the drift term is NOT a small correction: it
            wraps through many cycles, so the map has several fixed points.  We
            take the SHORTEST self-consistent ToF within the propellant budget
            (the most direct feasible transfer) by bracketing the first zero of
            g(ToF) = burn(dV(ToF)) - ToF on [ToF_floor, ToF_budget] and
            bisecting.  If no such ToF exists the arc is INFEASIBLE at t (the
            required plane change exceeds the whole dV budget); we return the
            budget dV as a sentinel with feasible=False.  Transit drift shifts
            each window earlier (the "lead angle") and narrows it; the dV floor
            (pure altitude, theta=0) is unchanged.
        """
        if self.propulsion == "impulsive":
            return self._transfer_impulsive(i, j, t)

        td = self.transit_drift if transit_drift is None else transit_drift
        draan_dep = self.raan_at(j, t) - self.raan_at(i, t)          # deg, unwrapped

        if not td:
            theta = self._plane_angle(draan_dep)
            dv = self._edelbaum_dv(i, j, theta)
            return {"dv": dv, "tof": self._burn_days(dv), "theta": theta,
                    "draan_thrust": wrap180(draan_dep),
                    "draan_dep": wrap180(draan_dep), "feasible": True}

        # net differential drift rate between target and chaser during transfer
        k = self.raan_rate(j) - self.mean_drift_rate(i, j)          # deg/day
        floor = abs(self.circ_speed(i) - self.circ_speed(j)) * 1000.0

        def resid(tf):
            """g(tf) = burn(dV) - tf ; >=0 at ToF_floor, root => self-consistent."""
            dr = wrap180(draan_dep + k * tf)
            dv = self._edelbaum_dv(i, j, self._plane_angle(dr))
            return self._burn_days(dv) - tf, dv, dr

        tof_lo = self._burn_days(floor)                 # shortest possible (theta=0)
        tof_hi = self._burn_days(self.dv_budget)        # feasibility horizon
        # bracket the first sign change of g from >=0 to <0
        prev_tf, prev_g = tof_lo, resid(tof_lo)[0]
        root = None
        for n in range(1, grid + 1):
            tf = tof_lo + (tof_hi - tof_lo) * n / grid
            g = resid(tf)[0]
            if prev_g >= 0.0 > g:
                lo, hi = prev_tf, tf                    # bisect
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if resid(mid)[0] >= 0.0:
                        lo = mid
                    else:
                        hi = mid
                root = 0.5 * (lo + hi)
                break
            prev_tf, prev_g = tf, g

        if root is not None:
            _, dv, dr = resid(root)
            return {"dv": dv, "tof": root, "theta": self._plane_angle(dr),
                    "draan_thrust": dr, "draan_dep": wrap180(draan_dep),
                    "feasible": True}
        # infeasible within budget: no feasible transfer departs at this time
        return {"dv": self.dv_budget, "tof": tof_hi, "theta": math.nan,
                "draan_thrust": math.nan, "draan_dep": wrap180(draan_dep),
                "feasible": False}

    # ------------------------------------------------------------------ #
    # Arc cost:  dV (m/s) and ToF (days)
    # ------------------------------------------------------------------ #
    def delta_v(self, i, j, t: float, transit_drift: bool | None = None) -> float:
        """Edelbaum combined altitude + plane-change dV  [m/s]."""
        return self.transfer(i, j, t, transit_drift)["dv"]

    def tof(self, i, j, t: float, transit_drift: bool | None = None) -> float:
        """Continuous-thrust Time of Flight  [days] at the reference mass."""
        return self.transfer(i, j, t, transit_drift)["tof"]

    def propellant(self, i, j, t: float, m0: float | None = None) -> float:
        """Propellant burned on the arc  [kg] for a spacecraft of mass m0
        (default: reference mass).  dm = m0 (1 - exp(-dV/c))."""
        m0 = self.ref_mass if m0 is None else m0
        return m0 * (1.0 - math.exp(-self.delta_v(i, j, t) / self.exhaust_speed))

    # ------------------------------------------------------------------ #
    # Road-model-compatible aliases (so solvers can be shared)
    # ------------------------------------------------------------------ #
    def travel_time(self, i, j, t: float) -> float:
        """Time-dependent travel time c_ijt := ToF(i, j, t)  [days]."""
        return self.tof(i, j, t)

    def preference(self, i, j, t: float) -> float:
        """Time-dependent preference u_ijt := -dV(i, j, t)  [m/s, negated]."""
        return -self.delta_v(i, j, t)

    # ------------------------------------------------------------------ #
    # Route evaluation (sequential, time-dependent, mass-tracking)
    # ------------------------------------------------------------------ #
    def evaluate_route(self, nodes: list, t0: float | None = None,
                       dep_times: list | None = None) -> dict:
        """Walk a node sequence, accumulating ToF, dV and propellant.

        ``dep_times`` (elapsed-day departure time per leg) enables waiting at a
        node; otherwise every leg departs as soon as the previous arrives (no
        waiting).  Mass is tracked so ToF/propellant reflect the shrinking
        spacecraft.  Returns totals plus feasibility (propellant within budget
        and every arc present).
        """
        if t0 is None:
            t0 = self.t_start
        t = t0
        total_tof = 0.0
        total_dv = 0.0
        m = self.wet_mass_kg
        feasible = True
        legs = []
        for leg, (i, j) in enumerate(zip(nodes, nodes[1:])):
            if not self.G.has_edge(i, j):
                feasible = False
                break
            t_dep = t if dep_times is None else max(t, dep_times[leg])
            wait = t_dep - t
            dv = self.delta_v(i, j, t_dep)
            dm = m * (1.0 - math.exp(-dv / self.exhaust_speed))
            tof = (dm / self.mdot) / SECONDS_PER_DAY
            legs.append({"i": i, "j": j, "t_dep": t_dep, "wait": wait,
                         "dv": dv, "tof": tof, "prop": dm})
            total_dv += dv
            total_tof += tof
            m -= dm
            t = t_dep + tof
        prop_used = self.wet_mass_kg - m
        if prop_used > self.fuel_mass_kg + 1e-9:
            feasible = False
        return {"tof": total_tof, "delta_v": total_dv, "propellant": prop_used,
                "arrival": t, "feasible": feasible, "legs": legs}

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def n_arcs(self) -> int:
        return self.G.number_of_edges()

    def synodic_period(self, i, j) -> float:
        """RAAN re-alignment period 360 / |Omega_dot_i - Omega_dot_j| [days]
        (inf if the two nodes share an altitude and never realign)."""
        drate = abs(self.raan_rate(i) - self.raan_rate(j))
        return math.inf if drate == 0.0 else 360.0 / drate

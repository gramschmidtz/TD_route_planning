"""
Adapter that runs the exact epsilon-constraint method (td_route/eecm) on the
ORBITAL network.

Three things the road EECM assumes but the orbital network violates, and how
this adapter reconciles them:

1. Integer time on a grid.  The EECM propagates the time index as t + c_ijt and
   steps waiting arcs by +1, so travel times must be integers on a fixed grid.
   Orbital ToF is a real number of days, so we work in STEP units of dt (default
   1 day): step index t <-> absolute day (t_start + t*dt), and
   travel_time = ceil(ToF / dt) steps.

2. Infeasible arcs.  A departure time at which the required plane change exceeds
   the whole dV budget has no valid transfer.  transfer()["feasible"] flags
   these, and arc_feasible() lets TimeExpanded drop them (never create x_ijt),
   so "that arc is unusable at that time" is honoured exactly.

3. Propellant budget.  Written as in the space-mission formulation as a cap on
   the number of maneuvers,  sum_{i,j,t} x_ijt < N, with
       N = floor( dV_budget / dV_per_day ),   dV_per_day = (Tmax / m0) * 86400,
   i.e. the total dV budget divided by the dV one day of thrust delivers at the
   wet mass (~4.536 m/s/day) -> N = 1243.  Passed to the solver as
   arc_cap = N - 1  (encoding the strict "< N" for an integer arc count).

transfer() results are cached per (i, j, step) because arc_feasible /
travel_time / preference each need them and the root-find is not free.
"""
from __future__ import annotations

import json
import math

from .orbital import OrbitalNetwork


class OrbitalTD:
    """Presents an OrbitalNetwork to the EECM in integer dt-day time steps.

    Exposes exactly what TimeExpanded / solve_single read: G, n_nodes, t_start,
    t_end, horizons, travel_time(i,j,t), preference(i,j,t), arc_feasible(i,j,t).
    All t are STEP indices (absolute day = onet.t_start + t*dt_days).
    """

    def __init__(self, onet: OrbitalNetwork, dt_days: float = 1.0,
                 t_end_days: float | None = None, unit_pref: bool = False,
                 mission=None, cost_dv: bool = False):
        self.onet = onet
        self.dt = float(dt_days)
        # Three preference (f1) modes:
        #   mission set  -> u_ijt = v_j(t + c_ijt) = mission value AT ARRIVAL
        #                   (max f1 => max total collected mission value);
        #   unit_pref    -> u_ijt = 1 (max f1 => max visited-node count);
        #   otherwise    -> u_ijt = -dV.
        self.unit_pref = unit_pref
        self.mission = mission
        # cost_dv=True exposes arc_cost(i,j,t)=dv_ijt, which switches the EECM's
        # SECOND objective to  min sum dv x  (waiting free) instead of elapsed
        # time -- the space-mission "min dv" formulation.  Defined as an
        # instance attribute so TimeExpanded's getattr hook only sees it when
        # the mode is on.
        if cost_dv:
            self.arc_cost = lambda i, j, t: self._tr(i, j, t)["dv"]
        self.G = onet.G
        self.t_start = 0
        end = onet.t_end if t_end_days is None else t_end_days
        self.n_steps = int(round((end - onet.t_start) / self.dt))
        self.t_end = self.n_steps
        self.horizons = [(0, self.n_steps)]        # for _safe_time_budget only
        self.dv_budget = onet.dv_budget
        self._cache: dict = {}

    @property
    def maneuver_cap(self) -> int:
        """The old maneuver-COUNT bound N in  sum x_ijt < N, i.e.
        N = floor(dV_budget / dV_per_day_at_wet_mass)  (= 1243 for the default
        chaser).  Kept for reference; the flight-time budget below is preferred."""
        dv_per_day = self.onet.thrust_N / self.onet.wet_mass_kg * 86400.0
        return int(self.onet.dv_budget / dv_per_day)

    @property
    def burn_budget_days(self) -> float:
        """Total FLIGHT (burn) time that expends the whole propellant load, in
        days: fuel / mdot = fuel * c / T.  Because mdot = T/c is constant, this
        is the EXACT propellant budget written as a cap on sum tof x
        (= 1081 d for 100 kg -- vs the ~15%-loose 1243 that divided total dV by
        the wet-mass-only dV/day rate)."""
        return self.onet.fuel_mass_kg / self.onet.mdot / 86400.0

    @property
    def n_nodes(self) -> int:
        return self.onet.n_nodes

    def _tr(self, i, j, t):
        key = (i, j, t)
        r = self._cache.get(key)
        if r is None:
            day = self.onet.t_start + t * self.dt
            r = self.onet.transfer(i, j, day)
            self._cache[key] = r
        return r

    # ---- interface consumed by the EECM -------------------------------- #
    def arc_feasible(self, i, j, t) -> bool:
        return self._tr(i, j, t)["feasible"]

    def travel_time(self, i, j, t) -> int:
        """ToF rounded UP to whole dt steps (>= 1)."""
        return max(1, math.ceil(self._tr(i, j, t)["tof"] / self.dt))

    def preference(self, i, j, t) -> float:
        """u_ijt: mission value at arrival v_j(t+c) / 1 per arc / -dV."""
        if self.mission is not None:
            arr = (t + self.travel_time(i, j, t)) * self.dt      # arrival day
            return self.mission.value(j, arr)
        return 1.0 if self.unit_pref else -self._tr(i, j, t)["dv"]

    # ---- reporting helpers --------------------------------------------- #
    def day(self, step) -> float:
        return self.onet.t_start + step * self.dt

    def route_delta_v(self, sol) -> float:
        """Total dV [m/s] of a solution's route, recomputed from its timeline
        (works regardless of which objective was optimised)."""
        route, tl = sol.route, sol.meta.get("timeline")
        if not tl:
            return -sol.f1 if not self.unit_pref else float("nan")
        dv = 0.0
        for m in range(len(route) - 1):
            dep = (tl[m]["departure"] or 0) * self.dt
            dv += self.onet.transfer(route[m], route[m + 1], dep)["dv"]
        return dv

    def route_elapsed_days(self, sol) -> float:
        """Elapsed mission time [days] of a solution, from its timed schedule.
        (sol.f2 is NOT always time: in dv mode it is total dV.)"""
        tl = sol.meta.get("timeline")
        if tl:
            return float(tl[-1]["arrival"]) * self.dt
        el = sol.meta.get("elapsed")
        return float(el) * self.dt if el is not None else float(sol.f2) * self.dt

    def route_value(self, sol) -> float:
        """Total mission value collected = sum of v_j(arrival) over the route's
        arrivals (0 if no mission model)."""
        if self.mission is None:
            return float("nan")
        route, tl = sol.route, sol.meta.get("timeline")
        if not tl:
            return float("nan")
        return sum(self.mission.value(route[m + 1], tl[m + 1]["arrival"] * self.dt)
                   for m in range(len(route) - 1))

    def describe(self, sol) -> str:
        """Human-readable route summary: mission value (if any), visited-node
        count, total dV, fuel, elapsed time (from the timed schedule)."""
        dv = self.route_delta_v(sol)
        m_final = self.onet.wet_mass_kg * math.exp(-dv / self.onet.exhaust_speed)
        fuel = self.onet.wet_mass_kg - m_final
        val = "" if self.mission is None else f"value={self.route_value(sol):5.2f}  "
        return (f"route {sol.route}  {val}visits={len(sol.route)}  "
                f"sum dV={dv:.0f} m/s  fuel={fuel:.1f}/{self.onet.fuel_mass_kg:.0f} kg"
                f"  elapsed={self.route_elapsed_days(sol):.0f} d")


# ========================================================================== #
# Persisting the Pareto front, so the (slow) solve runs ONCE
# ========================================================================== #
def save_front(front, td: OrbitalTD, path: str, objective: str = "mission_value",
               f2_mode: str = "time"):
    """Write the solved Pareto front to JSON.

    Stores each solution's route and full timed schedule (arrival/departure per
    node, in dt STEPS) plus its derived metrics, and the context needed to
    replay it (objective, f2 mode, dt_days, O/D).  The network and the mission
    model are NOT stored -- both are rebuilt deterministically from config.yaml.
    NOTE: raw f2 is elapsed STEPS in time mode but total dV [m/s] in dv mode;
    elapsed_days is therefore always taken from the timed schedule.
    """
    sols = []
    for s in sorted(front, key=lambda z: z.f2):
        sols.append({
            "route": [int(v) for v in s.route],
            "f1": float(s.f1), "f2": float(s.f2),
            "elapsed_days": float(td.route_elapsed_days(s)),
            "value": (None if td.mission is None else float(td.route_value(s))),
            "delta_v": float(td.route_delta_v(s)),
            "visits": len(s.route),
            "optimal": bool(s.meta.get("optimal", False)),
            "timeline": [{"node": int(e["node"]), "arrival": e["arrival"],
                          "departure": e["departure"]}
                         for e in s.meta.get("timeline", [])],
            "dep_times": s.meta.get("dep_times"),
        })
    data = {"meta": {"objective": objective, "f2": f2_mode, "dt_days": td.dt,
                     "t_end_days": td.n_steps * td.dt,
                     "origin": td.G.graph.get("origin"),
                     "destination": td.G.graph.get("destination"),
                     "n_solutions": len(sols)},
            "solutions": sols}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"saved -> {path}  ({len(sols)} solutions)")
    return data


def load_front(path: str) -> dict:
    """Read a front written by save_front() -> {"meta": ..., "solutions": [...]}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def solve_orbital_eecm(onet: OrbitalNetwork, O=None, D=None, *, dt_days=1.0,
                       t_end_days=None, waiting=True, origin_wait=True,
                       objective="max_visits", f2="time", mission=None,
                       enforce_budget=True, tof_budget_days=None,
                       dv_budget=None, delta=None,
                       time_limit=None, exact_cfg=None, verbose=True):
    """Run the exact epsilon-constraint Pareto tracer on an orbital network.

    objective (f1):
      * "mission_value" -- f1 = sum v_j(t_arr) (needs mission);
      * "max_visits"    -- f1 = visited-node count (u_ijt = 1);
      * "min_dv"        -- f1 = -sum dV.

    f2 (second objective, minimised):
      * "time" -- elapsed time  sum tof x + sum w  (f2 in dt steps);
      * "dv"   -- total propellant  sum dv_ijt x_ijt  (waiting free; f2 in m/s).
        This is the space-mission formulation  max sum v x / min sum dv x.

    Propellant budget (enforce_budget):
      * f2="dv"   -> sum dv x <= dv_budget (default onet.dv_budget = 5642.4 m/s
                     = Isp*ge*ln(m0/(m0-fuel)), the exact 100 kg cut);
      * f2="time" -> sum tof x <= tof_budget_days (default burn_budget_days =
                     fuel/mdot = 1081 d).
    """
    from .eecm.core import epsilon_constraint
    from .eecm.solve import configure_solver

    O = onet.G.graph.get("origin") if O is None else O
    D = onet.G.graph.get("destination") if D is None else D
    configure_solver(exact_cfg or {})

    unit_pref = (objective == "max_visits")
    use_mission = (objective == "mission_value")
    if use_mission and mission is None:
        raise ValueError("objective='mission_value' requires a mission model")
    if delta is None:
        delta = 0.3 if use_mission else (1 if unit_pref else 100.0)
    cost_dv = (f2 == "dv")
    td = OrbitalTD(onet, dt_days=dt_days, t_end_days=t_end_days,
                   unit_pref=unit_pref, mission=mission if use_mission else None,
                   cost_dv=cost_dv)
    tof_cap = cost_cap = None
    if enforce_budget:
        if cost_dv:
            cost_cap = onet.dv_budget if dv_budget is None else dv_budget
        else:
            bd = td.burn_budget_days if tof_budget_days is None else tof_budget_days
            tof_cap = bd / td.dt                  # days -> steps (matches c_ijt)
    if verbose:
        cut = ("no propellant cut" if not enforce_budget else
               (f"dv cut: sum dv x < {cost_cap:.1f} m/s" if cost_dv else
                f"flight-time cut: sum tof x <= {tof_cap * td.dt:.0f} d"))
        print(f"objective: {objective} / f2: "
              f"{'sum dv (waiting free)' if cost_dv else 'elapsed time'}   {cut}")
    front = epsilon_constraint(td, O, D, t0=0, delta=delta, waiting=waiting,
                               origin_wait=origin_wait, budget=td.n_steps,
                               tof_cap=tof_cap, cost_cap=cost_cap,
                               time_limit=time_limit, verbose=verbose)
    return front, td

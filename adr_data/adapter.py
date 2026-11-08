"""
Adapter: FY-1C ADR tables  ->  the input interface of the EXISTING exact
TD-graph MILP (td_route/eecm, unchanged).

What td_route.eecm.TimeExpanded / solve_single read from a "net":
    G (nx.DiGraph), n_nodes, t_start, t_end, horizons,
    travel_time(i,j,t) [int steps], preference(i,j,t) [f1, maximised],
    arc_feasible(i,j,t) [optional], arc_cost(i,j,t) [optional -> f2 = sum cost x]
and the model has ONE origin O (optionally allowed to wait, origin_wait=True)
and ONE destination D that the unit of flow must reach.

Mapping (prompt §6):
    node 0         = S, virtual launch source (O).  S waits on its own wait arcs
                     and takes ONE arc S->j at step t with day(t) in the launch
                     window [0, 180]  =>  launch day t_L and first target are
                     decision variables; first rendezvous dV = 0, f1 += score_j(t_L).
    nodes 1..n     = debris D01..D10 ; node v <-> ids[v-1].
                     Wait arcs (j,t)->(j,t+1) cost 0 (core-generated).
    node n+1       = T, virtual sink (D).  Arc j->T only at the last step H, so
                     the servicer simply idles at its last debris until the end.
    debris arc     (i,t)->(j,t+k), k = max(min_transfer_steps(=1),
                                         ceil((tof_phys + tau_s)/dt))   (A2)
                   cost = dV_ij(day t) + phasing_dv  [departure day]
                   pref = score_j(day(t + k))        [GRAPH arrival day]  (P1)
                   (optionally integerised: round(K * score), K = pareto.score_scale)
    re-entered debris: no arcs into / out of it, no waits, no sink arc  (P6)
    node-once      = core constraint (8): each debris visited at most once.

Time: step t <-> day(t) = t * dt_days; coarse grids use the grid-day values.
f2 is total dV [m/s] (arc_cost present => waiting is free in the core).
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from .load import ADRData
from .maneuver import phasing_dv


def dv_max_mps(cfg: dict) -> float:
    """Tsiolkovsky budget Isp*g0*ln(m0/m_dry) - margin  (= 2161.6 m/s)."""
    s = cfg["servicer"]
    return (s["isp_s"] * s["g0"] * math.log(s["m0_kg"] / s["m_dry_kg"])
            - float(s.get("dv_margin_mps", 0.0)))


class ADRNetwork:
    """Presents the ADR data to td_route.eecm in integer grid steps."""

    def __init__(self, data: ADRData, cfg: dict, dv_cap: float | None = None):
        self.data = data
        self.cfg = cfg
        m = cfg["mission"]
        mv = cfg["maneuver"]
        self.dt = float(m.get("dt_days", 1))
        self.horizon_days = int(m["horizon_days"])          # days 0..horizon-1
        self.H = int((self.horizon_days - 1) // self.dt)    # last step index
        lw = m["launch_window_days"]
        self.launch_lo, self.launch_hi = float(lw[0]), float(lw[1])
        self.tau_s = float(mv.get("tau_s_days", 0.0))
        self.tof_phys = float(mv.get("tof_phys_days", 0.07))
        # A2: graph arrival >= physical completion (transfer + stay), and >= 1
        # step so the time index strictly increases on debris arcs (Q1)
        self.k_transfer = max(int(mv.get("min_transfer_steps", 1)),
                              int(math.ceil((self.tof_phys + self.tau_s) / self.dt
                                            - 1e-12)))
        K = (cfg.get("pareto") or {}).get("score_scale")
        self.score_scale = None if K in (None, 0) else float(K)
        self.mv = mv
        self.dv_cap = dv_max_mps(cfg) if dv_cap is None else float(dv_cap)

        n = data.n
        self.S, self.T = 0, n + 1
        self.debris = list(range(1, n + 1))
        G = nx.DiGraph()
        G.add_nodes_from([self.S, *self.debris, self.T])
        for j in self.debris:
            G.add_edge(self.S, j)
            G.add_edge(j, self.T)
            for k in self.debris:
                if k != j:
                    G.add_edge(j, k)
        G.graph.update(origin=self.S, destination=self.T)
        self.G = G
        # attributes read by TimeExpanded
        self.t_start = 0
        self.t_end = self.H
        self.horizons = [(0, self.H)]

    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()

    def day(self, t) -> float:
        return t * self.dt

    def _didx(self, day: float) -> int:
        return int(round(day))

    def name(self, v) -> str:
        if v == self.S:
            return "S"
        if v == self.T:
            return "T"
        return self.data.ids[v - 1]

    def score(self, j, day) -> float:
        """score of debris NODE j at a (possibly fractional -> rounded) day."""
        return float(self.data.score[j - 1, self._didx(day)])

    def value(self, j, day) -> float:
        """objective coefficient: score, or round(K*score) when integerised."""
        v = self.score(j, day)
        return float(round(self.score_scale * v)) if self.score_scale else v

    def reentered(self, j, day) -> bool:
        if j in (self.S, self.T):
            return False
        r = self.data.reentry_day(j - 1)
        return r is not None and day >= r

    def wait_feasible(self, v, t) -> bool:
        """P6: no waiting on a re-entered debris (core hook)."""
        return not self.reentered(v, self.day(t + 1))

    def raw_dv(self, i, j, day) -> float:
        return float(self.data.dv[i - 1, j - 1, self._didx(day)])

    def transfer_dv(self, i, j, t) -> float:
        d = self.day(t)
        return self.raw_dv(i, j, d) + phasing_dv(i - 1, j - 1, d, self.mv)

    # ---- interface consumed by td_route.eecm --------------------------- #
    def arc_feasible(self, i, j, t) -> bool:
        if i == self.S:                                  # launch arc
            d = self.day(t)
            return (self.launch_lo <= d <= self.launch_hi
                    and self.score(j, d) > 0.0 and not self.reentered(j, d))
        if j == self.T:                                  # sink only at the end
            return t == self.H and not self.reentered(i, self.day(t))
        if self.reentered(i, self.day(t)) or \
                self.reentered(j, self.day(t + self.k_transfer)):
            return False
        dv = self.transfer_dv(i, j, t)                   # NaN = re-entered
        return bool(np.isfinite(dv) and dv <= self.dv_cap)

    def travel_time(self, i, j, t) -> int:
        if i == self.S or j == self.T:
            return 0                                     # S/T never lie on a cycle
        return self.k_transfer

    def preference(self, i, j, t) -> float:
        if j == self.T:
            return 0.0
        if i == self.S:
            return self.value(j, self.day(t))
        return self.value(j, self.day(t + self.k_transfer))   # P1: graph arrival

    def arc_cost(self, i, j, t) -> float:
        if i == self.S or j == self.T:
            return 0.0
        return self.transfer_dv(i, j, t)

    # ---- reporting ------------------------------------------------------ #
    def describe_solution(self, sol) -> dict:
        """Turn a core Solution (route + timeline in steps) into physical days."""
        tl = sol.meta["timeline"]                 # [{node, arrival, departure}]
        route = sol.route
        assert route[0] == self.S and route[-1] == self.T
        visits, legs = [], []
        launch_day = self.day(tl[0]["departure"])
        for m in range(1, len(route) - 1):
            j = route[m]
            prev = route[m - 1]
            dep_prev = self.day(tl[m - 1]["departure"])
            if prev == self.S:
                arr_day = dep_prev
                dv = 0.0
                tof = 0.0
            else:
                arr_day = self.day(tl[m - 1]["departure"] + self.k_transfer)
                dv = self.transfer_dv(prev, j, tl[m - 1]["departure"])
                tof = arr_day - dep_prev
            nxt = tl[m]["departure"]
            dep_day = self.day(nxt) if route[m + 1] != self.T else None
            visits.append({
                "id": self.name(j), "from": self.name(prev),
                "depart_prev_day": dep_prev, "arrival_day": arr_day,
                "score": self.score(j, arr_day), "dv": dv, "tof_days": tof,
                "score_scaled": (int(round(self.score_scale * self.score(j, arr_day)))
                                 if self.score_scale else None),
                "departure_day": dep_day,
                "wait_days": (dep_day - arr_day) if dep_day is not None else None,
            })
            legs.append(dv)
        return {"launch_day": launch_day, "visits": visits,
                "total_dv": float(sum(legs)),
                "total_score": float(sum(v["score"] for v in visits)),
                "n_visits": len(visits)}

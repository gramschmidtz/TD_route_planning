"""
Time-dependent road network for the bi-objective preference-driven route
planning problem (Gao, Chu & Chen, IEEE TETCI 2026).

This module holds the SHARED problem layer used by both solvers (the exact
epsilon-constraint method and the problem-specific NSGA-II):

    - the directed graph G = (V, A)
    - the time-dependent travel time   c_ijt = ceil(d_ij / v_ijt)
    - the time-dependent preference    u_ijt = bp_ij * td_ij(t)
    - the rectangular region partition used by the NSGA-II encoding

Clock convention
----------------
`t` is ALWAYS a wall-clock time in minutes from midnight (e.g. 7am = 420).
The paper's model index "t = 1" simply corresponds to the departure clock
time t0 at the origin; we keep wall-clock minutes everywhere because the
speed / preference profiles are defined against time-of-day.

External references
-------------------
* Speed model [11] (Verbeeck et al., 2014): the four unequal time periods and
  the five arc categories use the EXACT speed matrix of Table 1, and travel
  time is computed with their FIFO piecewise scheme (Algorithm 1) in
  `travel_time` -- the speed changes when a period boundary is crossed mid-arc.
* Preference factor: each preference arc carries its OWN piecewise multiplier
  m_ij(t) in {1, 1-beta, 1+beta}, with beta ~ U[0.2,0.8] per arc and states
  lasting U[60,240] min (see build_pref_schedule / instance.py). Non-preference
  arcs have preference score 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

# --------------------------------------------------------------------------
# Time-of-day structure: 7am(420) .. 9pm(1260), split into 4 horizons.
# --------------------------------------------------------------------------
DEFAULT_HORIZONS = [(420, 540), (540, 1020), (1020, 1140), (1140, 1260)]
HORIZON_NAMES = ["07-09", "09-17", "17-19", "19-21"]

# Speed matrix v_{c,k} -- the EXACT Table 1 of Verbeeck et al. (2014), EJOR
# 236(2):419-432, the speed model cited as ref [11] by the planning paper.
# Columns are the four time periods below; the average speed is 1 (= the
# time-independent baseline), so these are dimensionless ratios. The network's
# `free_flow_speed` rescales them to distance-per-minute.
#   k=0: 7-9am Morning peak | k=1: 9am-5pm Normal
#   k=2: 5-7pm Evening peak | k=3: 7-9pm Normal
SPEED_CATEGORIES = {
    "always_busy":     [0.5, 0.81, 0.5, 0.81],
    "morning_peak":    [0.5, 0.7,  1.0, 1.5],
    "two_peaks":       [0.5, 1.5,  0.5, 1.5],
    "evening_peak":    [1.0, 1.5,  0.5, 0.7],
    "seldom_traveled": [1.5, 1.5,  1.5, 1.5],
}

# Time-dependent preference model.
#   u_ijt = ceil( bp_ij * m_ij(t) )         (0 on non-preference arcs)
# where m_ij(t) is a PER-ARC piecewise-constant multiplier taking values in
# {1, 1-beta, 1+beta}. Each preference arc draws its own beta ~ U[0.2, 0.8] and
# its own schedule of states, each state lasting U[60, 240] minutes. The
# schedule is stored on the arc as attribute 'pref_schedule' (see instance.py);
# beta is stored as 'beta'. This replaces the old fixed PREF_PROFILES table.
PREF_STATE_MIN_DUR = 60      # minutes
PREF_STATE_MAX_DUR = 240
BETA_RANGE = (0.2, 0.8)


def build_pref_schedule(rng, beta, t_start, t_end,
                        min_dur=PREF_STATE_MIN_DUR, max_dur=PREF_STATE_MAX_DUR):
    """Build one arc's preference-multiplier schedule over [t_start, t_end).

    Returns a list of (seg_start, seg_end, multiplier) segments that tile the
    day. Each segment lasts U[min_dur, max_dur] minutes and takes a multiplier
    from {1, 1-beta, 1+beta}; consecutive segments differ so the profile
    actually varies.
    """
    levels = [1.0, 1.0 - beta, 1.0 + beta]
    segs, t, prev = [], t_start, None
    while t < t_end:
        end = min(t + rng.randint(min_dur, max_dur), t_end)
        choices = [m for m in levels if m != prev] or levels
        m = rng.choice(choices)
        segs.append((t, end, m))
        prev, t = m, end
    return segs


@dataclass
class TDNetwork:
    """A directed, time-dependent road network."""

    free_flow_speed: float = 2.0          # distance units per minute
    t_start: int = 420                    # 7:00
    t_end: int = 1260                     # 21:00
    horizons: list = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    speed_categories: dict = field(default_factory=lambda: dict(SPEED_CATEGORIES))
    # legacy fallback for arcs still tagged with a named 'pref_profile'
    pref_profiles: dict = field(default_factory=lambda: {
        "constant": [1.0, 1.0, 1.0, 1.0]})

    def __post_init__(self):
        self.G = nx.DiGraph()
        self.regions: dict = {}           # region_id -> {"nodes": set, "bbox": (..)}
        self.node_region: dict = {}       # node -> region_id
        self.region_grid: dict | None = None

    # ------------------------------------------------------------------ #
    # Time horizon lookup
    # ------------------------------------------------------------------ #
    def horizon_index(self, t: float) -> int:
        for k, (a, b) in enumerate(self.horizons):
            if a <= t < b:
                return k
        if t < self.horizons[0][0]:
            return 0
        return len(self.horizons) - 1     # clamp at/after t_end

    # ------------------------------------------------------------------ #
    # Speed and travel time
    # ------------------------------------------------------------------ #
    def speed(self, i, j, t: float) -> float:
        """Instantaneous speed v_ijt at departure time t (= v_{c,k} rescaled)."""
        cat = self.G[i][j]["speed_cat"]
        factor = self.speed_categories[cat][self.horizon_index(t)]
        return self.free_flow_speed * factor

    def travel_time(self, i, j, t: float) -> int:
        """Time-dependent travel time c_ijt via Algorithm 1 of Verbeeck et al.
        (2014, ref [11]): the FIFO piecewise scheme.

        The vehicle starts at speed v_{c,k} of the departure period k. If it
        reaches the end of period k before arriving, the remaining distance is
        continued at the next period's speed v_{c,k+1}, and so on -- so the
        speed changes WHEN A PERIOD BOUNDARY IS CROSSED MID-ARC. The continuous
        travel time (t_a - t_d) is then rounded up to whole minutes, matching
        the planning paper's discretisation c_ijt = ceil(.) with mu = 1 min.
        """
        cat = self.G[i][j]["speed_cat"]
        speeds = self.speed_categories[cat]
        v0 = self.free_flow_speed

        td = t
        d = self.G[i][j]["length"]
        k = self.horizon_index(t)
        cur = t
        v = v0 * speeds[k]
        ta = cur + d / v
        # cross period boundaries while the arrival overruns the current period
        while k < len(self.horizons) - 1 and ta > self.horizons[k][1]:
            end = self.horizons[k][1]
            d -= v * (end - cur)              # distance still to cover
            cur = end
            k += 1
            v = v0 * speeds[k]
            ta = cur + d / v
        return max(1, math.ceil(ta - td))

    # ------------------------------------------------------------------ #
    # Preference:  u_ijt = ceil( bp_ij * m_ij(t) )
    # ------------------------------------------------------------------ #
    def td_factor(self, i, j, t: float) -> float:
        """Preference multiplier m_ij(t) in {1, 1-beta, 1+beta} from the arc's
        per-arc schedule. Falls back to the old PREF_PROFILES table (or 1.0)
        when an arc has no schedule."""
        sched = self.G[i][j].get("pref_schedule")
        if sched:
            if t <= sched[0][0]:
                return sched[0][2]
            for a, b, m in sched:
                if a <= t < b:
                    return m
            return sched[-1][2]                     # clamp after day end
        prof = self.G[i][j].get("pref_profile")
        if prof is not None:
            return self.pref_profiles[prof][self.horizon_index(t)]
        return 1.0

    def preference(self, i, j, t: float) -> int:
        bp = self.G[i][j].get("bp", 0)
        if bp <= 0:
            return 0
        return math.ceil(bp * self.td_factor(i, j, t))

    # ------------------------------------------------------------------ #
    # Route evaluation (sequential, time-dependent)
    # ------------------------------------------------------------------ #
    def evaluate_route(self, nodes: list, t0: float | None = None) -> dict:
        """Walk a node sequence from t0, accumulating travel time & preference.

        Returns {"travel_time": f2, "preference": f1, "arrival": t,
                 "feasible": bool}. Infeasible if an arc is missing.
        """
        if t0 is None:
            t0 = self.t_start
        t = t0
        f1 = 0.0
        f2 = 0
        for i, j in zip(nodes, nodes[1:]):
            if not self.G.has_edge(i, j):
                return {"travel_time": f2, "preference": f1,
                        "arrival": t, "feasible": False}
            f1 += self.preference(i, j, t)
            c = self.travel_time(i, j, t)
            f2 += c
            t += c
        return {"travel_time": f2, "preference": f1,
                "arrival": t, "feasible": True}

    # ------------------------------------------------------------------ #
    # Region partition (for the NSGA-II region-based encoding)
    # ------------------------------------------------------------------ #
    def partition_regions(self, n_cols: int, n_rows: int) -> None:
        """Split the bounding box into n_cols x n_rows equal rectangles.

        Regions are numbered row-major from the TOP-LEFT, exactly like Fig. 3
        of the paper:  top row = 1..n_cols, next row = n_cols+1.., etc.
        An edge spanning two regions is counted in the region of its tail
        node (the "first region"), as stated in the paper.
        """
        xs = [d["x"] for _, d in self.G.nodes(data=True)]
        ys = [d["y"] for _, d in self.G.nodes(data=True)]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        cw = (xmax - xmin) / n_cols or 1.0
        ch = (ymax - ymin) / n_rows or 1.0

        self.region_grid = {
            "n_cols": n_cols, "n_rows": n_rows,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "cw": cw, "ch": ch,
        }
        self.regions = {}
        self.node_region = {}

        for v, d in self.G.nodes(data=True):
            col = min(int((d["x"] - xmin) / cw), n_cols - 1)
            # row 0 at the TOP -> measure from ymax downward
            row = min(int((ymax - d["y"]) / ch), n_rows - 1)
            rid = row * n_cols + col + 1
            self.node_region[v] = rid
            self.regions.setdefault(rid, {"nodes": set()})["nodes"].add(v)

        # tag every arc with the region of its tail node
        for i, j in self.G.edges():
            self.G[i][j]["region"] = self.node_region[i]

    def region_adjacency(self) -> dict:
        """4-neighbour (up/down/left/right) adjacency between region ids.

        Matches the paper: e.g. for a 4x3 grid, A_10 = {6, 9, 11}.
        """
        if self.region_grid is None:
            raise RuntimeError("call partition_regions() first")
        nc, nr = self.region_grid["n_cols"], self.region_grid["n_rows"]
        adj: dict = {}
        for rid in range(1, nc * nr + 1):
            r0, c0 = (rid - 1) // nc, (rid - 1) % nc
            nbrs = set()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = r0 + dr, c0 + dc
                if 0 <= r < nr and 0 <= c < nc:
                    nbrs.add(r * nc + c + 1)
            adj[rid] = nbrs
        return adj

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def n_arcs(self) -> int:
        return self.G.number_of_edges()

    @property
    def n_pref_arcs(self) -> int:
        return sum(1 for _, _, d in self.G.edges(data=True) if d.get("bp", 0) > 0)

    def pref_density(self) -> float:
        return self.n_pref_arcs / self.n_arcs if self.n_arcs else 0.0
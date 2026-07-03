"""
Time-expanded variable set for the exact epsilon-constraint method.

Faithful to model P (eqs. 1-10): the binary variables x_ijt are built only for
time-states (i, t) REACHABLE from the origin at the departure time t0. The time
index propagates deterministically through constraint (9)
    x_ijt <= sum_k x_jk(t + c_ijt),
so an arc taken at t arrives at j at t + c_ijt and must leave j at exactly that
time. Unreachable x_ijt are fixed to 0 in the paper, so dropping them changes
nothing while keeping the program tractable. Constraints (5)/(6) -- no arc into
the origin, none out of the destination -- are enforced here by simply never
creating those variables.
"""
from __future__ import annotations


class TimeExpanded:
    """Reachable (i, j, t) tuples with their travel time and preference."""

    def __init__(self, net, O, D, t0=None, max_horizon=None, budget=None):
        self.net = net
        self.O, self.D = O, D
        self.t0 = net.t_start if t0 is None else t0
        self.t_end = net.t_end if max_horizon is None else max_horizon

        # A simple route uses at most |V|-1 arcs. Summing the worst-case travel
        # time of the (|V|-1) slowest arcs gives a SAFE upper bound on the
        # travel time of any feasible (simple) route -- so pruning states past
        # t0 + budget cannot remove any Pareto-optimal route (exactness kept),
        # while it kills the cyclic state blow-up of the time expansion.
        if budget is None:
            budget = self._safe_time_budget()
        self.budget = budget
        self.t_horizon = min(self.t_end, self.t0 + budget)

        # arcs[(i, j, t)] = (c_ijt, u_ijt)
        self.arcs: dict = {}
        self._build()

    def _safe_time_budget(self) -> int:
        net = self.net
        worst = []
        for i, j in net.G.edges():
            cmax = max(net.travel_time(i, j, t)
                       for t in (a for a, _b in net.horizons))
            worst.append(cmax)
        worst.sort(reverse=True)
        k = max(1, net.n_nodes - 1)
        return int(sum(worst[:k]))

    def _build(self):
        net, O, D = self.net, self.O, self.D
        seen = {(O, self.t0)}
        queue = [(O, self.t0)]
        while queue:
            i, t = queue.pop()
            if i == D:
                # (6): no arc leaves the destination -- never create x_{D,j,t}.
                continue
            for j in net.G.successors(i):
                if j == O:
                    # (5): no arc enters the origin -- never create x_{i,O,t}.
                    continue
                c = net.travel_time(i, j, t)
                tp = t + c
                if tp > self.t_horizon:
                    continue
                u = net.preference(i, j, t)
                self.arcs[(i, j, t)] = (c, u)
                if j != D and (j, tp) not in seen:
                    seen.add((j, tp))
                    queue.append((j, tp))

    # ---- bookkeeping helpers -------------------------------------------- #
    def in_arcs(self, node):
        return [k for k in self.arcs if k[1] == node]

    def out_arcs(self, node):
        return [k for k in self.arcs if k[0] == node]

    @property
    def nodes(self):
        s = set()
        for (i, j, _t) in self.arcs:
            s.add(i)
            s.add(j)
        return s

    @property
    def reachable_destination(self) -> bool:
        return any(k[1] == self.D for k in self.arcs)

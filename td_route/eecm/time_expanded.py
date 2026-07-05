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

    def __init__(self, net, O, D, t0=None, max_horizon=None, budget=None,
                 waiting=False):
        self.net = net
        self.O, self.D = O, D
        self.t0 = net.t_start if t0 is None else t0
        self.t_end = net.t_end if max_horizon is None else max_horizon
        self.waiting = waiting

        # arcs[(i, j, t)] = (c_ijt, u_ijt)   -- travel arcs, used by both models
        self.arcs: dict = {}

        # The time-expansion horizon is t0 + budget (a cap on elapsed time).
        #   * budget = None          -> the whole day (t_end - t0), both modes.
        #   * budget = "safe"/"auto" -> a SAFE upper bound = sum of the (|V|-1)
        #     slowest arc travel times; no simple route can exceed it, so pruning
        #     past it keeps exactness while taming the state blow-up (no-wait).
        #   * budget = <int>         -> used as given (minutes).
        if budget is None:
            budget = self.t_end - self.t0
        elif isinstance(budget, str):
            budget = self._safe_time_budget()
        self.budget = int(budget)
        self.t_horizon = min(self.t_end, self.t0 + self.budget)

        if waiting:
            self.waits: set = set()          # wait arcs (v, tau) : (v,tau)->(v,tau+1)
            self.time_nodes: set = set()     # reachable (v, tau)
            self.arrivals: dict = {}         # (v, tau) -> [travel arcs ending here]
            self.departures: dict = {}       # (v, tau) -> [travel arcs leaving here]
            self._build_waiting()
        else:
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

    def _build_waiting(self):
        """Time-expanded graph WITH waiting: besides travel arcs, a node may sit
        idle via wait arcs (v, tau) -> (v, tau+1). The origin cannot wait (it
        departs only at t0, constraint (3)); the destination absorbs the flow."""
        net, O, D = self.net, self.O, self.D
        H = self.t_horizon
        seen = {(O, self.t0)}
        queue = [(O, self.t0)]
        self.time_nodes.add((O, self.t0))
        while queue:
            i, t = queue.pop()
            if i == D:
                continue                              # (6) destination absorbs
            # travel arcs
            for j in net.G.successors(i):
                if j == O:
                    continue                          # (5) nothing enters O
                c = net.travel_time(i, j, t)
                tp = t + c
                if tp > H:
                    continue
                self.arcs[(i, j, t)] = (c, net.preference(i, j, t))
                self.departures.setdefault((i, t), []).append((i, j, t))
                self.arrivals.setdefault((j, tp), []).append((i, j, t))
                self.time_nodes.add((j, tp))
                if j != D and (j, tp) not in seen:
                    seen.add((j, tp))
                    queue.append((j, tp))
            # wait arc (i, t) -> (i, t+1), for intermediate nodes only
            if i != O and t + 1 <= H:
                self.waits.add((i, t))
                self.time_nodes.add((i, t + 1))
                if (i, t + 1) not in seen:
                    seen.add((i, t + 1))
                    queue.append((i, t + 1))

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

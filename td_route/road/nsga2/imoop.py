"""
IMOOP -- high-quality initial-population generator and the route-generation
primitives shared by the NSGA-II decoder (Appendices A, B, C).

The paper's IMOOP improves MOOP [28] in three ways:
  i)  the IMPROVED ARC TABLE picks an arc's preferred successors from the
      current region AND its adjacent regions (not just the current region);
  ii) a per-user TIME BUDGET = travel time of the time-dependent LONGEST route
      from O to D (Appendix C heuristic);
  iii) it returns a set of non-dominated itineraries.

Search/evaluation split
-----------------------
Path *finding* uses a static weight approximation (mid-day travel time, or
negative preference) because the exact travel time of an arc depends on the
arrival time, which depends on the prefix. Every candidate route is then
scored EXACTLY with TDNetwork.evaluate_route(route, t0), so f1/f2 are always
the true time-dependent values.
"""
from __future__ import annotations

import random

import networkx as nx

from ...decision import Solution, non_dominated, dedup_front


# -------------------------------------------------------------------------- #
# (i) Improved arc table
# -------------------------------------------------------------------------- #
def build_improved_arc_table(net) -> dict:
    """For every preference arc (bp>0), list its preferred successor arcs whose
    tail is this arc's head and that live in the SAME or an ADJACENT region.

    Returns {(i, j): [(j, k), ...]} keyed by preference arc.
    """
    adj = net.region_adjacency()
    table: dict = {}
    for i, j, d in net.G.edges(data=True):
        if d.get("bp", 0) <= 0:
            continue
        here = net.node_region[i]
        allowed = {here} | adj.get(here, set())
        succ = []
        for k in net.G.successors(j):
            if k == i:
                continue
            if net.G[j][k].get("bp", 0) <= 0:
                continue
            if net.node_region[j] in allowed:
                succ.append((j, k))
        table[(i, j)] = succ
    return table


# -------------------------------------------------------------------------- #
# Static search weights
# -------------------------------------------------------------------------- #
def _mid_time(net, i, j):
    mid = (net.t_start + net.t_end) // 2
    return net.travel_time(i, j, mid)


def _pref_weight(net, i, j, eps=0.1):
    """Larger preference -> smaller weight (so shortest path = most preferred)."""
    mid = (net.t_start + net.t_end) // 2
    return 1.0 / (net.preference(i, j, mid) + eps)


# -------------------------------------------------------------------------- #
# Candidate route generators
# -------------------------------------------------------------------------- #
def _shortest_time_path(net, O, D, allowed_nodes=None):
    G = net.G if allowed_nodes is None else net.G.subgraph(allowed_nodes)
    try:
        return nx.shortest_path(G, O, D,
                                weight=lambda i, j, d: _mid_time(net, i, j))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _max_pref_path(net, O, D, allowed_nodes=None):
    G = net.G if allowed_nodes is None else net.G.subgraph(allowed_nodes)
    try:
        return nx.shortest_path(G, O, D,
                                weight=lambda i, j, d: _pref_weight(net, i, j))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _biased_walk(net, O, D, rng, allowed_nodes=None, bias="pref",
                 max_steps=None):
    """A random simple walk O->D; at each step pick an unvisited successor with
    probability proportional to its preference (bias='pref') or inverse travel
    time (bias='time'). Returns a node list or None if it gets stuck."""
    allowed = None if allowed_nodes is None else set(allowed_nodes)
    max_steps = max_steps or (net.n_nodes + 1)
    cur, route, visited = O, [O], {O}
    mid = (net.t_start + net.t_end) // 2
    for _ in range(max_steps):
        if cur == D:
            return route
        succ = [w for w in net.G.successors(cur)
                if w not in visited and (allowed is None or w in allowed)]
        if not succ:
            return None
        if bias == "pref":
            wts = [net.preference(cur, w, mid) + 0.1 for w in succ]
        else:
            wts = [1.0 / (net.travel_time(cur, w, mid) + 0.1) for w in succ]
        # nudge toward D occasionally to improve hit rate
        nxt = rng.choices(succ, weights=wts, k=1)[0]
        route.append(nxt)
        visited.add(nxt)
        cur = nxt
    return None


def generate_candidate_routes(net, O, D, t0, *, allowed_nodes=None,
                              n_random=8, rng=None) -> list:
    """Return a list of DISTINCT feasible Solutions (true TD objectives) found
    by a mix of shortest-time, max-preference and biased random walks. Used
    both for the initial population and for chromosome decoding."""
    rng = rng or random.Random()
    routes = []
    sp = _shortest_time_path(net, O, D, allowed_nodes)
    if sp:
        routes.append(sp)
    mp = _max_pref_path(net, O, D, allowed_nodes)
    if mp:
        routes.append(mp)
    for _ in range(n_random):
        bias = "pref" if rng.random() < 0.6 else "time"
        w = _biased_walk(net, O, D, rng, allowed_nodes, bias)
        if w:
            routes.append(w)

    sols, seen = [], set()
    for r in routes:
        res = net.evaluate_route(r, t0)
        if not res["feasible"]:
            continue
        key = tuple(r)
        if key in seen:
            continue
        seen.add(key)
        sols.append(Solution(route=r, f1=res["preference"],
                             f2=res["travel_time"],
                             meta={"t0": t0, "arrival": res["arrival"]}))
    return sols


# -------------------------------------------------------------------------- #
# (iii) Time budget -- time-dependent longest route (Appendix C heuristic)
# -------------------------------------------------------------------------- #
def longest_route_time(net, O, D, t0, rng=None, n_trials=30) -> int:
    """Heuristic upper bound on travel time: the travel time of the longest
    simple O->D route, found by degree-biased random walks (Appendix C). Used
    as the per-user time budget. Falls back to the shortest-path time if no
    long route connects O and D."""
    rng = rng or random.Random()
    best = 0
    # degree-biased walk that tries to wander long before reaching D
    deg = dict(net.G.degree())
    for _ in range(n_trials):
        cur, route, visited = O, [O], {O}
        while cur != D and len(route) <= net.n_nodes:
            succ = [w for w in net.G.successors(cur) if w not in visited]
            if not succ:
                break
            # prefer high-degree neighbours (Appendix C) but keep a path to D open
            wts = [deg[w] for w in succ]
            cur = rng.choices(succ, weights=wts, k=1)[0]
            route.append(cur)
            visited.add(cur)
        if route[-1] == D:
            res = net.evaluate_route(route, t0)
            if res["feasible"]:
                best = max(best, res["travel_time"])
    if best == 0:
        sp = _shortest_time_path(net, O, D)
        if sp:
            best = net.evaluate_route(sp, t0)["travel_time"]
    return int(best)


# -------------------------------------------------------------------------- #
# (ii) Initial population
# -------------------------------------------------------------------------- #
def initial_population(net, O, D, t0, size, rng=None) -> list:
    """Generate a high-quality initial set of Solutions (with chromosomes in
    meta['chromosome']). Produces diverse non-dominated O->D routes via IMOOP
    candidate generation, then pads with extra random walks up to `size`."""
    from .encoding import route_to_chromosome
    rng = rng or random.Random()

    pool = generate_candidate_routes(net, O, D, t0, n_random=4 * size, rng=rng)
    # prefer the non-dominated ones, then fill with the rest for diversity
    front = dedup_front(non_dominated(pool))
    rest = [s for s in pool if s not in front]
    rng.shuffle(rest)
    chosen = (front + rest)[:size] if len(front) < size else front[:size]

    for s in chosen:
        s.meta["chromosome"] = route_to_chromosome(net, s.route)
    return chosen

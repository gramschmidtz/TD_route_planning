"""
Random road-network generator reproducing the experimental setup of the paper
(Section V.A, "Randomly generated road networks").

Spec faithfully implemented:
  - |V| nodes placed at random positions
  - each node connected to n nearby nodes, n ~ U{1,2,3,4} (nearest neighbours)
  - arc length d_ij = Euclidean distance
  - preference-arc density = 50% (configurable)
  - basic preference score bp_ij ~ U{1..10} on preference arcs (0 otherwise)
  - per-arc speed category (5 day-categories) and temporal preference profile
  - the 12 size settings |V| in {10,20,30,40,50,60,70,80,90,100,150,200}

Notes
-----
* Arcs are added BIDIRECTIONALLY by default (each direction gets its OWN
  random time-dependent attributes, so u_ijt != u_jit in general -- this
  matches the paper's remark that the SFC beauty score depends on direction).
  Set bidirectional=False for a strictly out-degree<=4 directed graph.
* O/D assumptions (no arc into O, none out of D) are NOT baked into the data;
  the exact model enforces them via constraints (5)-(6). Use
  enforce_od_assumptions() if you prefer to clean the data instead.
"""
from __future__ import annotations

import math
import random

from .network import TDNetwork, SPEED_CATEGORIES, BETA_RANGE, build_pref_schedule

SIZE_SETTINGS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _arc_attrs(rng, length, speed_cats, bp_range, beta_range, is_pref,
               t_start, t_end, pref_state_dur):
    """Time-dependent attributes for one arc.

    Preference arcs get a basic score bp ~ U{bp_range}, a per-arc
    beta ~ U[beta_range], and their own piecewise preference schedule
    m_ij(t) in {1, 1-beta, 1+beta} with states lasting U{pref_state_dur} min.
    Non-preference arcs have bp = 0 (preference score 0 at all times).
    """
    attrs = {
        "length": length,
        "speed_cat": rng.choice(speed_cats),
        "bp": 0,
        "beta": 0.0,
    }
    if is_pref:
        beta = rng.uniform(*beta_range)
        attrs["bp"] = rng.randint(*bp_range)
        attrs["beta"] = beta
        attrs["pref_schedule"] = build_pref_schedule(
            rng, beta, t_start, t_end,
            min_dur=pref_state_dur[0], max_dur=pref_state_dur[1])
    return attrs


def generate_random_network(
    n_nodes: int,
    *,
    seed: int | None = None,
    area: float = 100.0,
    free_flow_speed: float = 2.0,
    pref_density: float = 0.5,
    bp_range: tuple = (1, 10),
    beta_range: tuple = BETA_RANGE,
    pref_state_dur: tuple = (60, 240),
    bidirectional: bool = True,
    n_region_cols: int = 4,
    n_region_rows: int = 3,
    speed_cats: list | None = None,
    speed_categories: dict | None = None,
    t_start: int = 420,
    t_end: int = 1260,
    horizons: list | None = None,
) -> TDNetwork:
    """Generate one random time-dependent road network.

    pref_density = rho, the fraction of arcs that are preference arcs. The
    network structure (t_start/t_end, horizons, speed_categories) can be
    overridden to match a config file; otherwise module defaults are used.
    """
    rng = random.Random(seed)
    speed_categories = speed_categories or dict(SPEED_CATEGORIES)
    speed_cats = speed_cats or list(speed_categories)

    net_kwargs = dict(free_flow_speed=free_flow_speed,
                      t_start=t_start, t_end=t_end,
                      speed_categories=dict(speed_categories))
    if horizons is not None:
        net_kwargs["horizons"] = [tuple(h) for h in horizons]
    net = TDNetwork(**net_kwargs)

    # 1) random node positions
    pos = {}
    for v in range(1, n_nodes + 1):
        pos[v] = (rng.uniform(0, area), rng.uniform(0, area))
        net.G.add_node(v, x=pos[v][0], y=pos[v][1])

    # 2) connect each node to its n nearest neighbours, n ~ U{1,4}
    candidate_pairs = set()
    for v in range(1, n_nodes + 1):
        k = rng.randint(1, min(4, n_nodes - 1))
        nearest = sorted(
            (w for w in range(1, n_nodes + 1) if w != v),
            key=lambda w: _dist(pos[v], pos[w]),
        )[:k]
        for w in nearest:
            candidate_pairs.add((v, w))
            if bidirectional:
                candidate_pairs.add((w, v))

    # 3) assign time-dependent attributes; mark ~pref_density arcs as preferred
    def _add_arc(i, j):
        if net.G.has_edge(i, j):
            return
        is_pref = rng.random() < pref_density
        net.G.add_edge(i, j, **_arc_attrs(
            rng, _dist(pos[i], pos[j]), speed_cats, bp_range, beta_range,
            is_pref, net.t_start, net.t_end, pref_state_dur))

    for (i, j) in candidate_pairs:
        _add_arc(i, j)

    # 4) guarantee STRONG connectivity: from any node one must be able to reach
    #    every other node (no isolated nodes / unreachable subnetworks). We
    #    repeatedly bridge the two closest strongly-connected components with a
    #    pair of arcs (both directions) until the whole graph is one component.
    _ensure_strongly_connected(net, pos, _add_arc)

    # 5) region partition for the NSGA-II encoding
    net.partition_regions(n_region_cols, n_region_rows)
    return net


def _ensure_strongly_connected(net, pos, add_arc):
    """Bridge strongly-connected components until the digraph is one SCC.

    Each iteration links the LARGEST component to the nearest node outside it
    with arcs in BOTH directions (merging the two components), so the number of
    components strictly decreases and the result is strongly connected.
    """
    import networkx as nx
    while True:
        comps = list(nx.strongly_connected_components(net.G))
        if len(comps) <= 1:
            return
        comps.sort(key=len, reverse=True)
        inside = comps[0]
        outside = [w for c in comps[1:] for w in c]
        best = None                       # (dist, u in largest, w outside)
        for u in inside:
            for w in outside:
                d = _dist(pos[u], pos[w])
                if best is None or d < best[0]:
                    best = (d, u, w)
        _, u, w = best
        add_arc(u, w)
        add_arc(w, u)


def pick_od(net: TDNetwork, seed: int | None = None,
            strategy: str = "farthest") -> tuple:
    """Pick an (origin, destination) pair.

    strategy="farthest": the two nodes whose straight-line distance is largest
    (gives long routes with room for trade-offs). strategy="random": uniform.
    """
    rng = random.Random(seed)
    nodes = list(net.G.nodes)
    if strategy == "random":
        o, d = rng.sample(nodes, 2)
        return o, d
    pos = {v: (net.G.nodes[v]["x"], net.G.nodes[v]["y"]) for v in nodes}
    best, pair = -1.0, (nodes[0], nodes[-1])
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            dd = _dist(pos[a], pos[b])
            if dd > best:
                best, pair = dd, (a, b)
    return pair


def enforce_od_assumptions(net: TDNetwork, O, D) -> None:
    """Remove arcs entering O and arcs leaving D (paper assumptions 3)."""
    for p in list(net.G.predecessors(O)):
        net.G.remove_edge(p, O)
    for s in list(net.G.successors(D)):
        net.G.remove_edge(D, s)


def build_benchmark(seeds_per_size: int = 10, base_seed: int = 0,
                    **kwargs) -> dict:
    """Build the 120-network benchmark (12 sizes x 10 instances)."""
    bench: dict = {}
    s = base_seed
    for n in SIZE_SETTINGS:
        bench[n] = []
        for _ in range(seeds_per_size):
            bench[n].append(generate_random_network(n, seed=s, **kwargs))
            s += 1
    return bench
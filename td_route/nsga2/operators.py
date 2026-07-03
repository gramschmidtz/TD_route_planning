"""
Genetic operators on region chromosomes (Sections IV.C.3-5):
    * 1-point crossover            (endpoints region(O)/region(D) preserved)
    * mutation                     (replace one interior gene by a fresh region)
    * feasibility check + repair   (insert region-shortest-path between a gap)
    * region-based decoding        (one chromosome -> a SET of feasible routes)
"""
from __future__ import annotations

import random

import networkx as nx

from ..decision import Solution
from .encoding import (route_to_chromosome, is_feasible, first_infeasible_gap,
                       dedup_consecutive)
from .imoop import generate_candidate_routes


# -------------------------------------------------------------------------- #
# Crossover (Section IV.C.3)
# -------------------------------------------------------------------------- #
def crossover_one_point(c1: list, c2: list, rng: random.Random):
    """1-point crossover. Both parents start at region(O) and end at region(D),
    so children keep those endpoints. A cut point is drawn in each parent."""
    if len(c1) < 2 or len(c2) < 2:
        return list(c1), list(c2)
    cp1 = rng.randint(1, len(c1) - 1)
    cp2 = rng.randint(1, len(c2) - 1)
    child1 = dedup_consecutive(c1[:cp1] + c2[cp2:])
    child2 = dedup_consecutive(c2[:cp2] + c1[cp1:])
    return child1, child2


# -------------------------------------------------------------------------- #
# Mutation (Section IV.C.3)
# -------------------------------------------------------------------------- #
def mutate(chromo: list, all_regions: set, rng: random.Random) -> list:
    """Replace one INTERIOR gene by a region not currently in the chromosome
    (endpoints region(O)/region(D) are never mutated)."""
    if len(chromo) <= 2:
        return list(chromo)
    pos = rng.randint(1, len(chromo) - 2)
    choices = list(all_regions - set(chromo))
    if not choices:
        return list(chromo)
    new = list(chromo)
    new[pos] = rng.choice(choices)
    return dedup_consecutive(new)


# -------------------------------------------------------------------------- #
# Feasibility + repair (Section IV.C.4)
# -------------------------------------------------------------------------- #
def _region_graph(adjacency: dict) -> nx.Graph:
    g = nx.Graph()
    for a, nbrs in adjacency.items():
        g.add_node(a)
        for b in nbrs:
            g.add_edge(a, b)
    return g


def repair(chromo: list, adjacency: dict, rng: random.Random,
           region_graph: nx.Graph | None = None, max_passes: int = 50) -> list:
    """Make a chromosome feasible by inserting, between any two non-adjacent
    consecutive genes, the regions on the shortest region-path linking them."""
    rg = region_graph or _region_graph(adjacency)
    c = dedup_consecutive(chromo)
    for _ in range(max_passes):
        gap = first_infeasible_gap(c, adjacency)
        if gap is None:
            return c
        a, b = c[gap], c[gap + 1]
        try:
            bridge = nx.shortest_path(rg, a, b)          # [a, ..., b]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return c                                     # cannot repair
        c = dedup_consecutive(c[:gap] + bridge + c[gap + 2:])
    return c


def is_chromosome_feasible(chromo: list, adjacency: dict) -> bool:
    return is_feasible(chromo, adjacency)


# -------------------------------------------------------------------------- #
# Region-based decoding (Section IV.C.5): chromosome -> set of feasible routes
# -------------------------------------------------------------------------- #
def decode_chromosome(net, chromo: list, O, D, t0, rng: random.Random,
                      max_solutions: int = 6, n_random: int = 10) -> list:
    """Decode a (feasible) chromosome into a set of feasible Solutions.

    Routes are confined to the nodes of the chromosome's regions and generated
    by the IMOOP candidate generator (shortest-time, max-preference and biased
    random walks), then scored with the true time-dependent objectives. Each
    returned Solution carries its own (possibly shorter) chromosome.
    """
    region_set = set(chromo)
    allowed = [v for v in net.G.nodes if net.node_region[v] in region_set]
    allowed_set = set(allowed)
    if O not in allowed_set or D not in allowed_set:
        return []

    sols = generate_candidate_routes(net, O, D, t0, allowed_nodes=allowed,
                                     n_random=n_random, rng=rng)
    for s in sols:
        s.meta["chromosome"] = route_to_chromosome(net, s.route)

    # keep the most diverse / best: sort by preference desc then time asc and cap
    sols.sort(key=lambda s: (-s.f1, s.f2))
    return sols[:max_solutions]

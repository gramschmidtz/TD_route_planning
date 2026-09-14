"""
Region-based chromosome encoding (Section IV.C.2).

A chromosome is a sequence of REGION ids (genes). A partial route inside one
region is one gene; a whole route is the ordered list of regions it crosses,
with consecutive duplicates collapsed. An edge spanning two regions is counted
in the region of its TAIL node ("the first region"), which is already how
TDNetwork.partition_regions tags arcs.

This module only deals with the discrete region layer (encode / feasibility).
Turning a chromosome back into concrete routes lives in nsga2.decode (it needs
the arc table and the actual graph).
"""
from __future__ import annotations


def route_to_chromosome(net, route: list) -> list:
    """Map a node sequence to its region sequence, collapsing repeats."""
    chromo = []
    for v in route:
        r = net.node_region[v]
        if not chromo or chromo[-1] != r:
            chromo.append(r)
    return chromo


def is_feasible(chromosome: list, adjacency: dict) -> bool:
    """Feasibility condition (Section IV.C.4): every immediate successor gene
    must be region-adjacent to its predecessor, i.e. j in A_i."""
    for a, b in zip(chromosome, chromosome[1:]):
        if a == b:
            continue
        if b not in adjacency.get(a, set()):
            return False
    return True


def first_infeasible_gap(chromosome: list, adjacency: dict):
    """Return the index i of the first pair (chromo[i], chromo[i+1]) that
    violates adjacency, or None if the chromosome is feasible."""
    for i in range(len(chromosome) - 1):
        a, b = chromosome[i], chromosome[i + 1]
        if a != b and b not in adjacency.get(a, set()):
            return i
    return None


def dedup_consecutive(chromosome: list) -> list:
    """Remove consecutive duplicate genes (a route never 'stays' as two genes)."""
    out = []
    for g in chromosome:
        if not out or out[-1] != g:
            out.append(g)
    return out


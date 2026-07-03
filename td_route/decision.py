"""
Pareto machinery + fuzzy-logic preferred-solution selection.

Objective convention (used EVERYWHERE in this project):
    f1 = total preference score   -> MAXIMISED
    f2 = total travel time        -> MINIMISED

A `Solution` is the common currency exchanged between the two solvers and the
evaluation/decision layers. The exact method and NSGA-II both emit Solutions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    """A feasible route together with its two objective values."""
    route: list = field(default_factory=list)   # node sequence O..D
    f1: float = 0.0                              # preference score (max)
    f2: float = 0.0                              # travel time      (min)
    meta: dict = field(default_factory=dict)     # arrival time, profile info...

    def objectives(self) -> tuple:
        return (self.f1, self.f2)


# -------------------------------------------------------------------------- #
# Dominance  (f1 max, f2 min)
# -------------------------------------------------------------------------- #
def dominates(a: Solution, b: Solution) -> bool:
    """True if `a` Pareto-dominates `b` (a no worse on both, strictly better
    on at least one)."""
    not_worse = (a.f1 >= b.f1) and (a.f2 <= b.f2)
    strictly_better = (a.f1 > b.f1) or (a.f2 < b.f2)
    return not_worse and strictly_better


def non_dominated(solutions: list) -> list:
    """Return the non-dominated subset (the Pareto front) of `solutions`."""
    front = []
    for s in solutions:
        if any(dominates(o, s) for o in solutions if o is not s):
            continue
        front.append(s)
    return front


def dedup_front(front: list, decimals: int = 6) -> list:
    """Drop solutions with duplicate (f1, f2) objective vectors."""
    seen, out = set(), []
    for s in front:
        key = (round(s.f1, decimals), round(s.f2, decimals))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# -------------------------------------------------------------------------- #
# NSGA-II selection primitives
# -------------------------------------------------------------------------- #
def fast_non_dominated_sort(pop: list) -> list:
    """Deb's fast non-dominated sort.

    Returns a list of fronts S1, S2, ... (each a list of Solutions), where
    S1 is the set of non-dominated solutions of `pop`, S2 of pop - S1, etc.
    """
    S = [[] for _ in pop]          # solutions dominated by i
    n = [0] * len(pop)             # domination count of i
    fronts = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()                   # drop trailing empty front
    return [[pop[idx] for idx in f] for f in fronts]


def crowding_distance(front: list) -> dict:
    """Crowding distance for every Solution in one front (eq. 14).

    Returns {id(solution): distance}. Boundary solutions get +inf.
    """
    dist = {id(s): 0.0 for s in front}
    if len(front) <= 2:
        for s in front:
            dist[id(s)] = float("inf")
        return dist

    for getter in (lambda s: s.f1, lambda s: s.f2):
        ordered = sorted(front, key=getter)
        lo, hi = getter(ordered[0]), getter(ordered[-1])
        span = hi - lo
        dist[id(ordered[0])] = float("inf")
        dist[id(ordered[-1])] = float("inf")
        if span == 0:
            continue
        for k in range(1, len(ordered) - 1):
            dist[id(ordered[k])] += (getter(ordered[k + 1]) -
                                     getter(ordered[k - 1])) / span
    return dist


def crowded_compare_key(front: list):
    """Sort key implementing the crowded-comparison operator within a front:
    larger crowding distance is preferred."""
    cd = crowding_distance(front)
    return lambda s: -cd[id(s)]


# -------------------------------------------------------------------------- #
# Fuzzy-logic preferred-solution selection (eqs. 15-17, Section IV.D)
# -------------------------------------------------------------------------- #
def _membership_max(f_k, f_best, f_worst):
    """Membership for a MAXIMISED objective (preference, eq. 15).
    f_best = f1^I (upper), f_worst = f1^N (lower)."""
    if f_best == f_worst:
        return 1.0
    if f_k >= f_best:
        return 1.0
    if f_k <= f_worst:
        return 0.0
    return (f_k - f_worst) / (f_best - f_worst)


def _membership_min(f_k, f_best, f_worst):
    """Membership for a MINIMISED objective (travel time, eq. 16).
    f_best = f2^I (lower), f_worst = f2^N (upper)."""
    if f_best == f_worst:
        return 1.0
    if f_k <= f_best:
        return 1.0
    if f_k >= f_worst:
        return 0.0
    return (f_worst - f_k) / (f_worst - f_best)


def fuzzy_select(front: list, w1: float = 0.5):
    """Select the preferred solution from a Pareto front by fuzzy logic.

    w1 = weight of the preference score (omega_f1); the travel-time weight is
    w2 = 1 - w1 (omega_f1 + omega_f2 = 1). Returns (best_solution, details)
    where details lists per-solution membership degrees delta_k (eq. 17).
    """
    if not front:
        raise ValueError("empty Pareto front")
    w2 = 1.0 - w1

    f1_I = max(s.f1 for s in front)      # best preference (upper limit)
    f1_N = min(s.f1 for s in front)      # worst preference (lower limit)
    f2_I = min(s.f2 for s in front)      # best travel time (lower limit)
    f2_N = max(s.f2 for s in front)      # worst travel time (upper limit)

    details = []
    best, best_delta = None, -1.0
    for s in front:
        d1 = _membership_max(s.f1, f1_I, f1_N)
        d2 = _membership_min(s.f2, f2_I, f2_N)
        delta = (w1 * d1 + w2 * d2) / (w1 + w2)
        details.append({"solution": s, "delta1": d1, "delta2": d2,
                        "delta": delta})
        if delta > best_delta:
            best, best_delta = s, delta
    return best, details

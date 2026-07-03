"""
Multi-objective performance indicators (Section V.A.2):

    * Cardinality (F)        -- number of Pareto points (bigger is better)
    * Hypervolume ratio (HR) -- HR = HV_A / HV_R   (>1 means A beats R)
    * e-dominance (D)        -- average closeness of front R to front A
                                (D close to 1 => similar performance)

All fronts are lists of decision.Solution with .f1 (max) and .f2 (min).
"""
from __future__ import annotations


def cardinality(front: list) -> int:
    """Number of Pareto points in the front."""
    return len(front)


# -------------------------------------------------------------------------- #
# Hypervolume (2D).  We minimise g = (-f1, f2); the reference (nadir) point is
# the worst value on each axis. HV is the area dominated by the front, bounded
# by the reference point.
# -------------------------------------------------------------------------- #
def _to_min_points(front):
    """Map (f1 max, f2 min) -> (g1 min, g2 min) with g1 = -f1, g2 = f2."""
    return [(-s.f1, s.f2) for s in front]


def nadir_point(*fronts):
    """Worst-on-each-axis point over the union of fronts, in (f1, f2) space.
    Returns (f1_worst, f2_worst) = (min f1, max f2)."""
    pts = [s for fr in fronts for s in fr]
    if not pts:
        raise ValueError("no points to derive a nadir from")
    return (min(s.f1 for s in pts), max(s.f2 for s in pts))


def hypervolume(front: list, reference) -> float:
    """2D hypervolume of `front` w.r.t. a reference (nadir) point.

    `reference` = (f1_worst, f2_worst) in the ORIGINAL (f1 max, f2 min) space,
    i.e. small f1 and large f2. Points not dominating the reference contribute
    nothing.
    """
    if not front:
        return 0.0
    r1, r2 = -reference[0], reference[1]          # reference in g-space
    pts = [(g1, g2) for (g1, g2) in _to_min_points(front)
           if g1 <= r1 and g2 <= r2]
    if not pts:
        return 0.0

    # keep only non-dominated points in g-space (minimisation)
    pts.sort(key=lambda p: (p[0], p[1]))
    nd = []
    best_g2 = float("inf")
    for g1, g2 in pts:
        if g2 < best_g2:
            nd.append((g1, g2))
            best_g2 = g2
    # nd is sorted by g1 ascending, g2 strictly descending
    hv = 0.0
    for i, (g1, g2) in enumerate(nd):
        next_g1 = nd[i + 1][0] if i + 1 < len(nd) else r1
        hv += (next_g1 - g1) * (r2 - g2)
    return hv


def hypervolume_ratio(front_a: list, front_r: list, reference=None) -> float:
    """HR = HV_A / HV_R. A common reference point is required; if not given it
    is derived as the shared nadir over both fronts (with a 1-unit margin so
    boundary points contribute positive area)."""
    if reference is None:
        f1w, f2w = nadir_point(front_a, front_r)
        reference = (f1w - 1.0, f2w + 1.0)
    hv_r = hypervolume(front_r, reference)
    hv_a = hypervolume(front_a, reference)
    if hv_r == 0:
        return float("inf") if hv_a > 0 else 1.0
    return hv_a / hv_r


# -------------------------------------------------------------------------- #
# e-dominance (eqs. 18-19)
# -------------------------------------------------------------------------- #
def e_dominance(front_a: list, front_r: list) -> float:
    """Average e-dominance D of front R relative to reference front A.

    e(x') = min_{x in A} max( f1(x)/f1(x'), f2(x)/f2(x') )   (eq. 18)
    D     = mean over x' in R of e(x')                        (eq. 19)
    D close to 1 => the two fronts are close. e(x')<1 => x' is dominated by
    some point of A.
    """
    if not front_r or not front_a:
        return float("nan")

    def ratio(num, den):
        if den == 0:
            return float("inf") if num > 0 else 1.0
        return num / den

    total = 0.0
    for xp in front_r:
        e_xp = min(
            max(ratio(x.f1, xp.f1), ratio(x.f2, xp.f2))
            for x in front_a
        )
        total += e_xp
    return total / len(front_r)


def summary(front_a: list, front_r: list, reference=None) -> dict:
    """Convenience bundle comparing algorithm A against reference R."""
    return {
        "card_A": cardinality(front_a),
        "card_R": cardinality(front_r),
        "HR": hypervolume_ratio(front_a, front_r, reference),
        "D": e_dominance(front_a, front_r),
    }

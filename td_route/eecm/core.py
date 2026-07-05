"""
Full exact epsilon-constraint Pareto tracer (Section IV.B).

The Pareto front is traced with P2(eps_k) (eq. 12): minimise total travel time
subject to total preference >= eps_k, raising eps_k = f1^{k-1} + delta until the
program becomes infeasible. Using P2 (the shortest-path-like direction) avoids
the NP-hard longest-path solves, which is why the paper prefers P2 over P1
(Fig. 10). The minimum-travel-time solution seeds the lower end of the front.
"""
from __future__ import annotations

from ..decision import dominates
from .time_expanded import TimeExpanded
from .solve import solve_single


def epsilon_constraint(net, O, D, t0=None, *, delta=1, time_limit=None,
                       max_iter=1000, waiting=False, budget=None, verbose=False):
    """Trace the exact Pareto front of model P. Returns a list of non-dominated
    Solutions sorted by travel time (empty if D is unreachable from O).

    waiting=True allows waiting at nodes (constraint 9'); then f2 is the total
    elapsed time arrival(D) - t0, which INCLUDES waiting. budget overrides the
    time-expansion horizon (t0 + budget); None = auto."""
    te = TimeExpanded(net, O, D, t0=t0, waiting=waiting, budget=budget)
    if not te.reachable_destination:
        if verbose:
            print("destination unreachable from origin")
        return []

    s_minf2 = solve_single(te, "min_f2", time_limit=time_limit)
    if s_minf2 is None:
        return []
    if verbose:
        print(f"min-time endpoint: f1={s_minf2.f1:.0f}, f2={s_minf2.f2} "
              f"(vars={len(te.arcs)}, budget={te.budget})")

    solutions = [s_minf2]
    eps = s_minf2.f1 + delta
    it = 0
    while it < max_iter:
        it += 1
        s = solve_single(te, "min_f2", eps=eps, time_limit=time_limit)
        if s is None:
            break                     # eps > f1^I -> Pareto front complete
        solutions.append(s)
        if verbose:
            print(f"  k={it}: eps={eps:.0f} -> f1={s.f1:.0f}, f2={s.f2}")
        eps = s.f1 + delta

    front = [s for s in solutions
             if not any(dominates(o, s) for o in solutions if o is not s)]
    uniq, seen = [], set()
    for s in sorted(front, key=lambda z: (z.f2, -z.f1)):
        key = (round(s.f1, 6), s.f2)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

"""
Endpoint computation for the epsilon-constraint method (Section IV.B).

To determine the epsilon interval the paper solves four single-objective
problems, in order:

    P(f1^I): f1^I = max f1                                  s.t. (3)-(10)
    P(f2^I): f2^I = min f2                                  s.t. (3)-(10)
    P(f1^N): f1^N = max f1   s.t. (3)-(10) and  f2 = f2^I
    P(f2^N): f2^N = min f2   s.t. (3)-(10) and  f1 = f1^I

which yield the two extreme Pareto points
    (f1^N, f2^I)   -- the minimum-travel-time end, and
    (f1^I, f2^N)   -- the maximum-preference end,
the epsilon interval [f1^N, f1^I], and (eq. 13) the first epsilon
    eps_1 = f1^N + delta      (delta = 1).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..decision import Solution
from .time_expanded import TimeExpanded
from .solve import solve_single


@dataclass
class Endpoints:
    f1_I: float          # max preference               (P(f1^I))
    f2_I: int            # min travel time               (P(f2^I))
    f1_N: float          # max preference at f2 = f2^I   (P(f1^N))
    f2_N: int            # min travel time at f1 = f1^I  (P(f2^N))
    sol_f1I: Solution
    sol_f2I: Solution
    sol_f1N: Solution
    sol_f2N: Solution
    delta: int = 1

    @property
    def point_max_pref(self):
        """Extreme Pareto point (f1^I, f2^N)."""
        return (self.f1_I, self.f2_N)

    @property
    def point_min_time(self):
        """Extreme Pareto point (f1^N, f2^I)."""
        return (self.f1_N, self.f2_I)

    @property
    def eps_interval(self):
        """The epsilon interval [f1^N, f1^I]."""
        return (self.f1_N, self.f1_I)

    @property
    def first_eps(self):
        """eps_1 = f1^N + delta (eq. 13)."""
        return self.f1_N + self.delta


def compute_endpoints(te: TimeExpanded, *, delta=1, time_limit=None,
                      verbose=False) -> Endpoints:
    """Solve P(f1^I), P(f2^I), P(f1^N), P(f2^N) in order and assemble the
    endpoints, epsilon interval and first epsilon. Returns None if the
    destination is unreachable or a solve fails."""
    if not te.reachable_destination:
        if verbose:
            print("destination unreachable from origin")
        return None

    # P(f1^I): maximise total preference
    sol_f1I = solve_single(te, "max_f1", time_limit=time_limit)
    # P(f2^I): minimise total travel time
    sol_f2I = solve_single(te, "min_f2", time_limit=time_limit)
    if sol_f1I is None or sol_f2I is None:
        return None
    f1_I, f2_I = sol_f1I.f1, sol_f2I.f2
    if verbose:
        print(f"P(f1^I): f1^I = {f1_I:.0f}")
        print(f"P(f2^I): f2^I = {f2_I}")

    # P(f1^N): maximise preference among the minimum-travel-time solutions
    sol_f1N = solve_single(te, "max_f1", fix_f2=f2_I, time_limit=time_limit)
    # P(f2^N): minimise travel time among the maximum-preference solutions
    sol_f2N = solve_single(te, "min_f2", fix_f1=f1_I, time_limit=time_limit)
    if sol_f1N is None or sol_f2N is None:
        return None
    f1_N, f2_N = sol_f1N.f1, sol_f2N.f2
    if verbose:
        print(f"P(f1^N): f1^N = {f1_N:.0f}  -> point (f1^N, f2^I) = "
              f"({f1_N:.0f}, {f2_I})")
        print(f"P(f2^N): f2^N = {f2_N}  -> point (f1^I, f2^N) = "
              f"({f1_I:.0f}, {f2_N})")

    return Endpoints(f1_I=f1_I, f2_I=f2_I, f1_N=f1_N, f2_N=f2_N,
                     sol_f1I=sol_f1I, sol_f2I=sol_f2I,
                     sol_f1N=sol_f1N, sol_f2N=sol_f2N, delta=delta)

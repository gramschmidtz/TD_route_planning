"""
Exact epsilon-constraint method (EECM) for the bi-objective TD route planning
problem (Section IV.B), using PuLP + CBC. Targets SMALL instances; use
td_route.nsga2 for large ones.

Modules:
    time_expanded : the reachable time-expanded variable set (model P)
    solve         : one transformed single-objective ILP (P_k)
    endpoints     : P(f1^I), P(f2^I), P(f1^N), P(f2^N) + epsilon interval
    core          : the full P2(eps_k) Pareto-front tracer
"""
from .time_expanded import TimeExpanded
from .solve import solve_single, extract_solution
from .endpoints import compute_endpoints, Endpoints
from .core import epsilon_constraint

__all__ = [
    "TimeExpanded",
    "solve_single",
    "extract_solution",
    "compute_endpoints",
    "Endpoints",
    "epsilon_constraint",
]

"""
Backward-compatibility shim. The exact epsilon-constraint method now lives in
the `td_route.eecm` package (organised by module, mirroring `td_route.road.nsga2`).

    from td_route.eecm import epsilon_constraint, compute_endpoints, TimeExpanded

This module re-exports the public API (and the old private names) so existing
imports keep working.
"""
from __future__ import annotations

from ..eecm import (TimeExpanded, epsilon_constraint, compute_endpoints,
                   Endpoints, solve_single, extract_solution)

# legacy private aliases used by older code/tests
_solve = solve_single
_extract = extract_solution

__all__ = [
    "TimeExpanded", "epsilon_constraint", "compute_endpoints", "Endpoints",
    "solve_single", "extract_solution", "_solve", "_extract",
]

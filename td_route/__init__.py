"""
td_route -- Bi-objective time-dependent preference-driven route planning.

Reference implementation of:
    L. Gao, F. Chu, C. Chen, "Bi-Objective Optimization for Time-Dependent
    Preference-Driven Route Planning", IEEE TETCI, vol. 10, no. 1, 2026.

Two solvers share one problem layer (network.py / instance.py):
    * exact epsilon-constraint method   -> model_exact.py   (small instances)
    * problem-specific NSGA-II          -> nsga2/           (large instances)

f1 = total preference score   (maximised)
f2 = total travel time        (minimised)
"""
from .road.network import TDNetwork
from .road.instance import generate_random_network, pick_od, build_benchmark

__all__ = [
    "TDNetwork",
    "generate_random_network",
    "pick_od",
    "build_benchmark",
]

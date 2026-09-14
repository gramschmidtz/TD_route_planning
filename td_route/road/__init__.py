"""
Road domain -- time-dependent preference-driven route planning (Gao et al.,
IEEE TETCI 2026).

    network      time-dependent graph: c_ijt (travel time), u_ijt (preference)
    instance     random road-network generator + 120-network benchmark
    dataset      JSON/PNG I/O for the fixed benchmark in dataset/
    model_exact  thin re-export of the shared exact solver (td_route.eecm)
    nsga2/       problem-specific NSGA-II (region chromosome encoding)
    simulate     agent journey sampling for animations
    demo         network sanity check
"""

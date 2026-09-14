"""
Orbital domain -- the space twin of the road network.

Nodes are circular orbits; an arc's cost depends on the departure time through
J2 RAAN drift (alignment windows) and rendezvous phasing.

    network   OrbitalNetwork: impulsive Hohmann / low-thrust Edelbaum transfers
    instance  the 8-orbit mission instance + JSON I/O
    planner   exact epsilon-constraint on the orbital net (adapter to td_route.eecm)
    mission   time-dependent mission value v_j(t)
    sim       timed trajectory sampling of a solved route
    anim      Three.js HTML / MP4 animation (vendor/ holds the JS libraries)
    viz       3D orbit and RAAN-drift figures
"""

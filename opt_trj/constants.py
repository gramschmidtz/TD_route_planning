"""Physical constants (SI, ECI/J2000-like inertial frame).

All modules in :mod:`opt_trj` work in SI units internally
(metres, seconds, kilograms, radians).
"""

import numpy as np

# --- Earth -----------------------------------------------------------------
MU = 3.986004418e14          # gravitational parameter [m^3/s^2]
R_E = 6378137.0              # equatorial radius [m]
OMEGA_E = 7.2921150e-5       # sidereal rotation rate [rad/s] (drag only)

# Zonal harmonics J_k, k = 2..4  (EGM96 / WGS84).  Index k -> J[k].
J_ZONAL = {2: 1.082626683e-3,
           3: -2.532435346e-6,
           4: -1.619331205e-6}

# --- Misc ------------------------------------------------------------------
G0 = 9.80665                 # standard gravity used in Isp [m/s^2]
DAY = 86400.0                # [s]

DEG = np.pi / 180.0
RAD = 180.0 / np.pi

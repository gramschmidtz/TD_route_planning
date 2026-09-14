"""Optimal impulsive trajectory design by primer vector theory (MEE + mass).

Implements Prussing, *Primer Vector Theory and Applications* (chapter 2 of
Conway, ed., "Spacecraft Trajectory Optimization", CUP 2010) with the state
extended from Cartesian ``[r; v]`` to modified equinoctial elements plus mass,
following the derivation on slides 10-12:

    x     = [p, f, g, h, k, L],   xdot = A(x) a + b(x)
    a     = a_T + a_g + a_D
    p(t)  = -A^T(x) lam_mee(t)
    S(t)  = |p| - (1 - lam_m),    Gamma* bang-bang on sign(S)

Modules
-------
constants       physical constants
mee             MEE algebra, A(x), b(x), conversions
perturbations   a_g (zonal J2..J4) and a_D (drag, off by default)
dynamics        equations of motion, Jacobian, STM propagation
costate         costates, primer vector, switching function
lambert         universal-variable Lambert solver
twoimpulse      fixed-time two-impulse rendezvous under the full force model
primer          necessary-condition evaluation and Lion-Handelsman gradients
optimize        the iteration driver (terminal coasts, midcourse impulses)
mission         mission points and spacecraft data
plots           figures
validate        physical self-checks
"""

__all__ = ["constants", "mee", "perturbations", "dynamics", "costate",
           "lambert", "twoimpulse", "primer", "optimize", "mission",
           "plots", "validate"]

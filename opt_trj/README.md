# `opt_trj` — optimal impulsive trajectories by primer vector theory (MEE + mass)

Implementation of J. E. Prussing, *Primer Vector Theory and Applications*
(chapter 2 of B. A. Conway, ed., **Spacecraft Trajectory Optimization**, CUP
2010), with the state extended from Cartesian `[r; v]` to **modified
equinoctial elements plus mass**, following the derivation on slides 10–12.

```
x      = [p, f, g, h, k, L]                 xdot = A(x) a + b(x)
a      = a_T + a_g + a_D
p(t)   = -A^T(x) lam_mee(t)                 u* = p / |p|
S(t)   = |p| - (1 - lam_m)                  Gamma* bang-bang on sign(S)
```

## The one identity everything rests on

Because `A = (∂x_mee/∂v) Q_r`, the MEE primer and Lawden's Cartesian primer are
the **same physical vector**:

```
-A^T(x) lam_mee  =  Q_r^T (-lam_v)          i.e.   Q_r (-A^T lam_mee) = -lam_v
```

so the whole Lion–Handelsman apparatus (cost gradients with respect to impulse
times and midcourse positions) carries over unchanged, and on a coast arc

```
lam_mee(t) = Phi(t,t0)^-T lam_mee(t0)       Phi = MEE state-transition matrix
```

is the MEE analogue of the chapter's symplectic Cartesian result (Eq. 2.36–37).
`validate.py` checks both numerically, plus `pdd = G p` in the two-body limit.

## Modules

| file | contents |
|---|---|
| `constants.py` | mu, R_E, zonal J2–J4, g0 |
| `mee.py` | MEE algebra, `A(x)`, `b(x)`, conversions, `d cart/d mee` |
| `perturbations.py` | `a_g` (zonal, in the slide's `dg_n / dg_r` form), `a_D` (drag, **off by default**) |
| `dynamics.py` | equations of motion, complex-step Jacobian, STM and dense propagation |
| `costate.py` | costate ODE, primer vector, primer BVP, switching function, `lam_m` |
| `lambert.py` | universal-variable Lambert (seeds only) |
| `mission.py` | the mission-point table, spacecraft, node orbits (secular + exact) |
| `twoimpulse.py` | perturbed two-/three-impulse arcs by Newton shooting |
| `primer.py` | Table 2.1 necessary conditions, Eqs. 2.59/2.60/2.65/2.75/2.76 |
| `optimize.py` | window location + calibration, scan, refinement driver |
| `plots.py` | RAAN drift, pork-chop, primer/switching/thrust, 3D trajectory |
| `run_1to2.py` | CLI entry point |
| `validate.py` | physical self-checks |

## Running

```bash
python -m opt_trj.validate                 # physics self-checks
python -m opt_trj.run_1to2                 # mission point 1 -> 2
python -m opt_trj.run_1to2 --origin 1 --target 8 --horizon 1200
python -m opt_trj.run_1to2 --zonal 4       # J2..J4
python -m opt_trj.run_1to2 --drag 2.2,15.0 # enable drag (C_D, area m^2)
python -m opt_trj.run_1to2 --seed 124.21370,46.13   # skip the ~3 min scan
python -m opt_trj.run_1to2 --no-midcourse           # stop at n = 2
```

The first run numerically propagates each mission-point orbit from epoch to the
transfer window (~0.4 s per simulated day); that anchor state is cached under
`opt_trj/_cache/`, so later runs start in seconds.

## Pipeline, and why it is not just "run the primer iteration"

Primer-vector gradients are **local**. For 1 → 2 the cost is dominated by the
plane: the two RAANs are 18.5° apart at epoch and only J2 nodal regression
(0.154 °/day of *relative* drift) closes them. At ~132 m/s per degree, a
departure at t = 0 costs about 2.4 km/s against a 1104 m/s budget. So:

1. **locate the J2 window** — closed-form secular crossing;
2. **calibrate it** — the secular rate uses the osculating semi-major axis and
   so carries an O(J2) ≈ 0.1 % error per node. Harmless on each rate, but the
   window is set by the *difference* of two nearly equal rates, where the same
   absolute error is ≈ 8 % — several days of window displacement. The crossing
   is therefore re-found on the numerically propagated node lines;
3. **propagate both orbits exactly** over the window (dense output). The
   secular mean-element model is fine for RAAN but useless for **phase**: over
   ~1800 revolutions the osculating/mean mean-motion difference accumulates
   into a phase error of order a full revolution;
4. **Lambert pork-chop** over (departure epoch, time of flight). Note the two
   very different time scales — the window moves over ~100 days, the rendezvous
   phasing repeats every orbital period (94.6 min = 0.066 d) — so the scan is
   run coarse *and* fine;
5. **re-converge** the arc against the perturbed dynamics by Newton shooting;
6. then the **necessary-condition loop**, starting at `n = 2`:

   ```
   repeat
       solve the primer BVP:  p(t_1) = dv_1/|dv_1|,  p(t_n) = dv_n/|dv_n|
                              -> 6 linear equations for lam_mee(t_1)
       refine every free variable to zero gradient:
           t_1                (2.59)   <- this IS the initial-coast criterion
           (r_m_i, t_m_i)     (2.75-76) for each interior impulse
           t_n                (2.60)   <- this IS the final-coast criterion
       if max|p| <= 1 anywhere inside:  stop, Table 2.1 holds
       else: splice a new impulse in at argmax|p| (2.64-2.69) and n <- n+1
   until converged or adding an impulse stops paying
   ```

   The terminal coasts need no separate branch: `t_1` and `t_n` are decision
   variables at every `n`, so `pdot(t_1) > 0` / `pdot(t_n) < 0` can never
   survive a converged refinement.

The refinement is cheap in the right place: the 6×6 primer BVP reuses the STM
the shooting step already produced, so the analytic gradient costs nothing
extra. It agrees with central finite differences of the converged cost to six
significant figures.

**What the loop does and does not prove.** Table 2.1 lists *first-order
necessary* conditions. They are also sufficient only for a **linear** system
(chapter ref [5], Prussing 1995); the perturbed two-body problem is not linear.
So a converged `n` is an extremal that satisfies the NC — enough to *rule out*
smaller `n` (a violated NC is a disproof), not enough to prove global
optimality. `twoimpulse.verify()` provides the separate, independent check that
the reported arc is actually flyable: it applies the reported ECI impulse
vectors to the origin state and propagates straight through, with no shooting,
and compares against the target orbit.

## Result for mission point 1 → 2 (J2, no drag)

The necessary-condition loop over `n`:

| n | ΔV [m/s] | max &#124;p&#124; | ToF [min] | Table 2.1 |
|---|---|---|---|---|
| 2 | 35.7524 | **1.74521** | 44.19 | **VIOLATED** |
| 3 | **34.7841** | 1.00086 | 86.25 | satisfied → loop stops |

The three-impulse solution: departure at t = 124.1987 d, impulses of
7.858 / 17.673 / 9.254 m/s at +0 / +44.56 / +86.25 min, 55.90 kg of 1500 kg
propellant, 3.15 % of the ΔV budget.

Reference: the ideal coplanar Hohmann between the two *mean* circles is
35.717 m/s with a 47.64 min half-period. The residual RAAN at departure is
−0.003°, so the plane change is essentially free — the whole cost is the
65 km altitude raise. The converged cost sits slightly *below* the Keplerian
Hohmann reference, which is expected: under J2 the osculating radius of a
"circular" orbit breathes by several km, worth a few m/s either way.

The two-impulse primer history is textbook Fig. 2.7 — |p| = 1 at both impulses,
`pdot·p = 0` at both terminals (no terminal coast helps), and a hump to 1.745
in between, which is precisely the criterion for adding a midcourse impulse.

Independent end-to-end check of the converged arc (`twoimpulse.verify`): apply
the reported ECI Δv vectors to the origin state and propagate straight through
with no shooting — position miss **1.3e-5 m**, velocity miss **2.1e-8 m/s**.
The arc is genuinely flyable and the ΔV is real.

Burn durations at Γ_max ≈ 0.098 m/s² are 80 s, 179 s and 94 s against a 95 min
orbit — 1.4 %, 3.2 % and 1.6 % of a revolution — which is what justifies the
impulsive approximation (chapter §2.2.2).

Residual gradients at n = 3: `dJ/dt_1` = 2.0e-5, `dJ/dt_n` = −1.8e-5 m/s per
second, `dJ/dt_m` = 6.0e-3 m/s per second. The interior-time direction is the
least converged (the refinement stops at its 120-iteration cap), but a
one-second shift of the midcourse epoch is worth 0.006 m/s, i.e. 0.02 % of the
cost. `max|p| = 1.00086` is 9e-4 above unity, which is inside the numerical
resolution here — the primer BVP matrix has condition number 2.3e8, so 1.00086
and exactly 1 are not distinguishable.

Figures are **shown, never written to disk** — `plot_all` builds them all and
calls `plt.show()` once, so every window opens together:

| figure | contents |
|---|---|
| RAAN drift | relative RAAN vs time — why departure is at 124 d |
| pork-chop | ΔV over (departure epoch, ToF), coarse and fine (skipped with `--seed`) |
| primer, n = 2 | **the Lambert two-impulse solution's \|p\|** — the hump to 1.745 that triggers the impulse insertion (cf. chapter Figs. 2.5, 2.7) |
| primer, n = 3 | the same after inserting and re-converging |

Each primer figure has three panels: `|p|`, `pdot·p` (Eq. 2.61), and the thrust
profile at `Gamma_max = T_max/m` with the true finite-burn widths. The
switching function of slide 12 is not plotted: `lam_m` is identically zero for
the cost `J = sum |dv_k|` (proof in `costate.lam_m_profile`), so
`S = |p| - 1` would just be the first panel shifted down by one.

| trajectory | 3D view plus the altitude profile |

`plots.plot_primer_stages(res)` draws one figure per stage of the
necessary-condition loop; `plots.plot_primer(res, traj=..., ana=...)` draws any
single one. Numeric results still go to `--json` (default
`outputs/opt_trj/opt_trj_1to2.json`).

## Three things that bite, and are worth knowing before running another pair

1. **Secular mean elements cannot predict phase.** Over the ~1800 revolutions
   to the window, the osculating/mean mean-motion difference accumulates into a
   phase error of order a full revolution — a mean-element Lambert scan for
   1 → 2 returns 10.8 km/s where the truth is 36 m/s. Everything past the
   window search runs on numerically propagated dense output.
2. **The secular RAAN window is also wrong, by four days.** The closed-form
   nodal rate uses the osculating semi-major axis, giving each node ~0.1 %
   error. Harmless individually — but the window is set by the *difference* of
   two nearly equal rates (0.154 of 4.7 °/day), where the same absolute error
   is ~8 %. For 1 → 2 the crossing moves from **120.21 d** (secular) to
   **124.28 d** (propagated), further than the whole scan window is wide.
   `optimize.calibrate_window()` re-finds it on the propagated node lines; the
   result is cached.
3. **The chapter's midcourse seed size is far too large here.** `beta = 0.05`
   displaces this 44-minute transfer by 346 km and *raises* the cost by
   815 m/s. The ladder has to go down to ~3e-4 (2 km) before `dJ < 0`. The
   chapter anticipates exactly this ("if the resulting dJ >= 0, then decrease
   eps and repeat"); the point is that the working scale is two to three orders
   of magnitude below the suggested value for a short LEO transfer.

## Modelling choices that were *not* given, and what was assumed

| quantity | value used | source |
|---|---|---|
| inclination | 51.6°, shared | project `config.yaml` (`orbital.inclination_deg`) — the mission-point table has only altitude / RAAN / phase |
| epoch | relative, `t = 0` | no absolute date is needed: the only time-dependent effect modelled is J2 nodal regression |
| eccentricity | 0 (circular) | stated with the table; the tabulated "true anomaly" is therefore the argument of latitude |
| perturbations | J2 only | J2–J4 selectable |
| drag | **disabled** | `C_D`, reference area and an atmosphere model were never specified for this vehicle; inventing them would silently bias the answer. `--drag CD,AREA` switches on the slide's `a_D` with a piecewise-exponential atmosphere |
| terminal condition | rendezvous (all six elements matched) | consistent with the phase column of the table |
| transfer time | free (minimum-ΔV) | `--fixed-tof` gives the chapter's fixed-time problem |

## Two typos on the slides

The self-check `validate.check_A_b` compares `A(x) a + b(x)` against the MEE
rate implied by the true Cartesian dynamics. It agrees to 3e-11 with the
standard Walker/Betts signs, and fails with the two slide entries below, which
are therefore transcription errors:

| slide | slide value | value used |
|---|---|---|
| `a_{2,3}` | `-sqrt(p/mu)(g/q)(h sinL + k cosL)` | `-sqrt(p/mu)(g/q)(h sinL - k cosL)` |
| `a_{3,1}` | `+sqrt(p/mu) cos L` | `-sqrt(p/mu) cos L` |

Similarly `v_theta` in the drag block is written `sqrt(mu/p)(1 + f sinL + g cosL)`;
it must be `sqrt(mu/p)(1 + f cosL + g sinL) = sqrt(mu/p) q`.

## The mass costate

For the chapter's cost `J = int Gamma dt = sum |dv_k|` every impulse is
weighted equally, `lam_m` normalises to 0 and `S = |p| - 1` recovers Lawden's
conditions of Table 2.1 — that is the impulsive limit used by the driver.
`costate.lam_m_rate` implements the slide's `lam_m_dot` for the finite-thrust
(CSI) case, where `lam_m` is piecewise constant on an impulsive trajectory
(`Gamma = 0` on every coast). The report prints the true finite-burn durations
against the coast length, which is what justifies the impulsive approximation
(chapter §2.2.2).

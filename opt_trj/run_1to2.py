"""Optimal impulsive transfer from mission point 1 to mission point 2.

    python -m opt_trj.run_1to2 [--zonal 2] [--drag CD,AREA] [--json out.json]

Method (Prussing, *Primer Vector Theory and Applications*, extended to the
MEE + mass state of slides 10-12):

1. locate the J2 nodal-alignment window between the two orbits;
2. propagate both mission-point orbits with the full force model;
3. Lambert pork-chop over (departure epoch, time of flight) for the basin;
4. re-converge the arc against the perturbed dynamics by Newton shooting;
5. solve the 6x6 primer boundary-value problem  p(t1) = dv1/|dv1|,
   p(t2) = dv2/|dv2|  for lam_mee(t1), with p = -A^T lam_mee;
6. drive dJ/dt1 and dJ/dtn (Eqs. 2.59-2.60) to zero -- i.e. optimise the
   initial and final coasts;
7. if max|p| > 1, insert a midcourse impulse (Eqs. 2.63-2.69) and drive
   dJ/dr_m, dJ/dt_m (Eqs. 2.75-2.76) to zero;
8. verify the necessary conditions of Table 2.1.
"""

import argparse
import json
import time

import numpy as np

from .constants import DAY, G0
from .dynamics import ForceModel
from .perturbations import DragModel
from .mission import NodeOrbit, Spacecraft, MISSION_POINTS, INCLINATION_DEG
from .twoimpulse import verify
from .optimize import optimize_transfer
from . import plots


def report(res):
    tr, ana, sc = res["trajectory"], res["analysis"], res["sc"]
    no, nf = res["node_o"], res["node_f"]
    L = []
    L.append("=" * 74)
    L.append(f" OPTIMAL IMPULSIVE TRANSFER   point {no.node_id} -> {nf.node_id}")
    L.append("=" * 74)
    L.append(f" force model            : {res['fm']}")
    L.append(f" origin  {no.node_id}: alt {no.alt/1e3:6.1f} km, "
             f"RAAN0 {np.degrees(no.raan0):7.2f} deg, u0 {np.degrees(no.u0):6.1f} deg")
    L.append(f" target  {nf.node_id}: alt {nf.alt/1e3:6.1f} km, "
             f"RAAN0 {np.degrees(nf.raan0):7.2f} deg, u0 {np.degrees(nf.u0):6.1f} deg")
    L.append(f" inclination            : {INCLINATION_DEG} deg (from config.yaml)")
    L.append(f" J2 alignment window    : t = {res['t_win']/DAY:.4f} d "
             f"(secular estimate {res['t_win_secular']/DAY:.4f} d)")
    L.append("")
    if "history" in res:
        L.append(" necessary-condition loop  (n increased while max|p| > 1)")
        L.append("      n      dV [m/s]     max|p|    ToF [min]   NC")
        for trn, anan in res["history"]:
            L.append(f"      {trn.n_imp}   {trn.total_dv:11.4f}  {anan.p_max:10.5f}"
                     f"  {(trn.t_imp[-1]-trn.t_imp[0])/60:10.3f}   "
                     f"{'satisfied' if anan.optimal() else 'VIOLATED'}")
        L.append("    NOTE: these are the FIRST-ORDER NECESSARY conditions of")
        L.append("    Table 2.1.  They are also sufficient only for a LINEAR system")
        L.append("    (chapter ref [5]); this one is nonlinear, so the converged n is")
        L.append("    an extremal satisfying the NC, not a proven global optimum.")
        L.append("")
    L.append(" trajectory")
    L.append(" " + tr.summary().replace("\n", "\n "))
    L.append(f"    transfer time (t_n - t_1) = "
             f"{(tr.t_imp[-1]-tr.t_imp[0])/60:.3f} min "
             f"= {(tr.t_imp[-1]-tr.t_imp[0])/no.period:.4f} orbits")
    L.append(f"    total mission time from epoch = {tr.t_imp[-1]/DAY:.5f} d")
    L.append("")
    L.append(" reference values")
    r1, r2 = no.a, nf.a
    mu = res["fm"].mu
    at = 0.5 * (r1 + r2)
    hoh = (abs(np.sqrt(mu * (2 / r1 - 1 / at)) - np.sqrt(mu / r1))
           + abs(np.sqrt(mu / r2) - np.sqrt(mu * (2 / r2 - 1 / at))))
    L.append(f"    ideal coplanar Hohmann (mean circles)  = {hoh:8.4f} m/s, "
             f"half-period {np.pi*np.sqrt(at**3/mu)/60:.3f} min")
    L.append(f"    impulsive plane change at {no.alt/1e3:.0f} km  "
             f"= {2*np.sqrt(mu/r1)*np.sin(np.radians(0.5)):7.1f} m/s per degree")
    L.append(f"    residual RAAN at departure             = "
             f"{(nf.raan_deg(tr.t_imp[0]) - no.raan_deg(tr.t_imp[0]) + 180) % 360 - 180:+8.4f} deg")
    L.append("    (the Hohmann figure is a Keplerian mean-circle reference; under")
    L.append("     J2 the osculating radius of a 'circular' orbit breathes by a few")
    L.append("     km, which is worth a few m/s either way, so the converged cost")
    L.append("     is expected to straddle it rather than sit above it.)")
    L.append("")
    L.append(" propellant")
    m = tr.masses
    L.append(f"    m0 = {m[0]:8.2f} kg  ->  mf = {m[-1]:8.2f} kg    "
             f"used {m[0]-m[-1]:7.2f} kg of {sc.fuel_mass:.0f} kg")
    L.append(f"    dV used {tr.total_dv:.3f} m/s of budget {sc.dv_budget:.1f} m/s "
             f"({100*tr.total_dv/sc.dv_budget:.2f} %)")
    for i, dv in enumerate(tr.dv_mag):
        bt = sc.burn_time(dv, m[i])
        L.append(f"    burn {i+1}: {dv:8.3f} m/s -> {bt:7.2f} s at "
                 f"Gamma_max = {sc.gamma_max(m[i]):.4f} m/s^2 "
                 f"({100*bt/no.period:.3f} % of an orbit)")
    L.append("    impulsive approximation is justified when the burn arcs are")
    L.append("    short against the coast between them (chapter section 2.2.2).")
    L.append("")
    L.append(" independent end-to-end check")
    L.append("    (start from the origin state at t1, apply the reported ECI")
    L.append("     impulse vectors, propagate straight through -- no shooting)")
    v = res["verify"]
    L.append(f"    position miss at t_n = {v['pos_miss_m']:10.3e} m")
    L.append(f"    velocity miss at t_n = {v['vel_miss_mps']:10.3e} m/s")
    L.append(f"    propellant           = {v['propellant_kg']:10.4f} kg")
    L.append("")
    L.append(" primer vector necessary conditions (Table 2.1)")
    L.append(ana.report())
    L.append("=" * 74)
    return "\n".join(L)


def to_json(res):
    tr, ana = res["trajectory"], res["analysis"]
    return {
        "origin": res["node_o"].node_id,
        "destination": res["node_f"].node_id,
        "force_model": repr(res["fm"]),
        "t_window_days": res["t_win"] / DAY,
        "n_impulses": int(tr.n_imp),
        "total_dv_mps": tr.total_dv,
        "impulse_times_days": (tr.t_imp / DAY).tolist(),
        "impulse_dv_eci_mps": tr.dv.tolist(),
        "impulse_dv_mag_mps": tr.dv_mag.tolist(),
        "masses_kg": tr.masses.tolist(),
        "transfer_time_s": float(tr.t_imp[-1] - tr.t_imp[0]),
        "primer": {
            "p_max": ana.p_max,
            "t_p_max_days": ana.t_p_max / DAY,
            "pdot_first": ana.pdot_first,
            "pdot_last": ana.pdot_last,
            "grad_t_first": ana.grad_t_first,
            "grad_t_last": ana.grad_t_last,
            "nc": {k: bool(v) for k, v in ana.nc().items()},
            "optimal": bool(ana.optimal()),
        },
        "nc_loop": [
            {"n": int(t.n_imp), "dv_mps": t.total_dv, "p_max": a.p_max,
             "tof_s": float(t.t_imp[-1] - t.t_imp[0]),
             "nc_satisfied": bool(a.optimal())}
            for t, a in res.get("history", [])
        ],
        "verification": res.get("verify", {}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", type=int, default=1, choices=sorted(MISSION_POINTS))
    ap.add_argument("--target", type=int, default=2, choices=sorted(MISSION_POINTS))
    ap.add_argument("--zonal", type=int, default=2,
                    help="highest zonal degree (0 = two-body, 2 = J2, up to 4)")
    ap.add_argument("--drag", type=str, default=None, metavar="CD,AREA",
                    help="enable drag, e.g. --drag 2.2,15.0 "
                         "(C_D and reference area in m^2; OFF by default because "
                         "neither was specified for this vehicle)")
    ap.add_argument("--horizon", type=float, default=250.0, help="scan horizon [days]")
    ap.add_argument("--window", type=int, default=0, help="which J2 window to use")
    ap.add_argument("--no-midcourse", action="store_true",
                    help="stop at two impulses (skip the NC loop)")
    ap.add_argument("--max-impulses", type=int, default=6,
                    help="cap on the necessary-condition loop")
    ap.add_argument("--no-refine", action="store_true",
                    help="insert the midcourse impulse at argmax|p| along p "
                         "(Eqs. 2.64-2.69) and STOP -- shows the freshly "
                         "inserted, still non-optimal arc, i.e. the chapter's "
                         "Fig. 2.8, without the slow joint refinement")
    ap.add_argument("--seed", type=str, default=None, metavar="T1_DAYS,TOF_MIN",
                    help="skip the departure scan and start from this "
                         "(departure epoch, time of flight), e.g. "
                         "--seed 124.21370,46.13")
    ap.add_argument("--json", type=str, default="opt_trj_1to2.json")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    drag = None
    if args.drag:
        cd, area = (float(v) for v in args.drag.split(","))
        drag = DragModel(cd, area)
    fm = ForceModel(zonal_degree=args.zonal, drag=drag)
    sc = Spacecraft()

    print(f"opt_trj: point {args.origin} -> {args.target}, {fm}")
    print(f"         spacecraft {sc.wet_mass:.0f} kg wet / {sc.fuel_mass:.0f} kg fuel, "
          f"T = {sc.thrust_max:.0f} N, Isp = {sc.isp:.1f} s "
          f"-> c = {sc.c:.1f} m/s, dV budget = {sc.dv_budget:.1f} m/s")

    no = NodeOrbit(args.origin, fm)
    nf = NodeOrbit(args.target, fm)
    t0 = time.time()
    seed = None
    if args.seed:
        seed = tuple(float(v) for v in args.seed.split(","))
    res = optimize_transfer(no, nf, fm, sc, t_max_days=args.horizon,
                            window_index=args.window,
                            max_impulses=args.max_impulses,
                            seed=seed,
                            refine_after_insert=not args.no_refine,
                            allow_midcourse=not args.no_midcourse)
    res["verify"] = verify(res["trajectory"], no, nf, fm, sc)
    print(f"  [done] {time.time()-t0:.1f} s\n")
    print(report(res))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(res), fh, indent=2)
        print(f"\n      wrote {args.json}")
    if not args.no_plots:
        plots.plot_all(res)
    return res


if __name__ == "__main__":
    main()

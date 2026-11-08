"""
Exact epsilon-constraint Pareto front for the ADR MILP, with checkpoint / resume.

    python -m adr_data.pareto_exact                      # config pareto.mode (score_exact)
    python -m adr_data.pareto_exact --method dv_exact
    python -m adr_data.pareto_exact --outputs-only       # rebuild summaries / figures

Methods (all: MIP gap 0, integrality 0, lexicographic 2-stage, P5 re-check):
  score_exact  stage 1  min sum dV  s.t. sum v' >= eps            (v' = round(K score))
               stage 2  max sum v'  s.t. sum dV <= dV*_k + 1e-6   -> point (s_k, d_k)
               eps_0 = 0, eps_{k+1} = s_k + 1; stop when stage 1 is infeasible
               (sum dV <= dV_max is always imposed, so dV* > dV_max == infeasible)
  dv_exact     stage 1  max sum v   s.t. sum dV <= eps             (arcs dV > eps pruned)
               stage 2  min sum dV  s.t. sum v >= f* - tol
               eps_0 = dV_max, eps_{k+1} = d_k - delta; stop when d_k = 0
               (v raw, or v' if pareto.score_scale is set)

Every finished point is appended to <out>/checkpoint.json at once and written as
<out>/solutions/S###.json (score_exact: numbering = discovery order = dV
ascending); a restart continues from the stored next epsilon.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time

import numpy as np

from td_route.eecm import TimeExpanded, solve_single
from td_route.eecm.solve import last_solve_tolerance_failure

from .adapter import ADRNetwork, dv_max_mps
from .load import load_config, load_dataset, load_mean_elements
from .maneuver import combined_impulses
from .run_pareto import raw_dv_of, setup_solver


class ExactFront:
    def __init__(self, data, cfg, method=None, out_dir=None, day0=0, elems=None,
                 time_limit=None, log=print):
        self.data, self.cfg = data, cfg
        p = cfg["pareto"]
        self.method = method or p.get("mode", "score_exact")
        self.K = p.get("score_scale")
        self.delta = float(p.get("delta_mps", 1e-3))
        self.dvmax = dv_max_mps(cfg)
        self.tl = time_limit if time_limit is not None else p.get("time_limit")
        self.out = out_dir
        self.day0 = day0                       # window offset (sub-problems)
        self.elems = elems
        self.log = log
        self.calls = []                        # per MILP call: (stage, seconds, optimal)
        self._te = {}
        self.epoch = _dt.date.fromisoformat(
            str(cfg["dataset"].get("epoch_utc", "2026-06-05"))[:10])

    # ------------------------------------------------------------ model
    def model(self, cap):
        key = round(float(cap), 9)
        if key not in self._te:
            if len(self._te) > 2:
                self._te.clear()
            net = ADRNetwork(self.data, self.cfg, dv_cap=cap)
            te = TimeExpanded(net, net.S, net.T, t0=0, waiting=True, origin_wait=True,
                              budget=net.H)
            self._te[key] = (net, te)
        return self._te[key]

    def _solve(self, stage, cap, sense, *, eps=None, cost_cap=None):
        net, te = self.model(cap)
        t0 = time.time()
        s = solve_single(te, sense, eps=eps, cost_cap=cost_cap, time_limit=self.tl)
        dt = time.time() - t0
        self.calls.append({"stage": stage, "s": dt,
                           "optimal": None if s is None else bool(s.meta.get("optimal")),
                           "mip_gap": None if s is None else s.meta.get("mip_gap")})
        return net, s, dt

    def _capped(self, stage, cap_arcs, sense, cap, *, eps=None, tries=4):
        """P5: re-solve with a tighter cap if the raw-table dV exceeds cap."""
        margin, tot = 0.0, 0.0
        for _ in range(tries):
            net, s, dt = self._solve(stage, cap_arcs, sense, eps=eps, cost_cap=cap - margin)
            tot += dt
            if s is None:
                if last_solve_tolerance_failure():   # MIP feasible, fixed LP rejected
                    self.log(f"   P5: tolerance failure at cap {cap - margin:.6f} -> tighter")
                    margin = max(2 * margin, 1e-6 * max(1.0, cap))
                    continue
                return net, None, tot
            d = raw_dv_of(net, s)
            if d <= cap + 1e-9:
                return net, s, tot
            self.log(f"   P5: raw dV {d:.6f} > cap {cap:.6f} -> re-solve tighter")
            margin = max(2 * margin, 1e-6 * max(1.0, cap), d - cap + 1e-9)
        return net, s, tot

    def _score_check(self, net, s, eps):
        """P5 for the score axis: raw sum v' >= eps (integers -> exact)."""
        return eps is None or s.f1 >= eps - 1e-9

    # ------------------------------------------------------------ methods
    def _step_score(self, eps):
        net1, s1, t1 = self._capped("min_dV", self.dvmax, "min_f2", self.dvmax, eps=eps)
        if s1 is None:
            return None
        if not self._score_check(net1, s1, eps):
            self.log(f"   P5: stage-1 score {s1.f1} < eps {eps}")
        d_star = raw_dv_of(net1, s1)
        cap = d_star + 1e-6
        net2, s2, t2 = self._capped("max_v", min(cap, self.dvmax), "max_f1", cap)
        s = s2 if (s2 is not None and s2.f1 >= s1.f1 - 1e-9) else s1
        net = net2 if s is s2 else net1
        opt = bool(s1.meta.get("optimal")) and (s2 is None or bool(s2.meta.get("optimal")))
        gaps = [g for g in (s1.meta.get("mip_gap"), s2.meta.get("mip_gap") if s2 else None)
                if g is not None]
        st_opt = [bool(s1.meta.get("optimal")),
                  None if s2 is None else bool(s2.meta.get("optimal"))]
        st_gap = [s1.meta.get("mip_gap"), None if s2 is None else s2.meta.get("mip_gap")]
        return dict(net=net, sol=s, eps=eps, optimal=opt,
                    mip_gap=max(gaps) if gaps else None, solve_s=t1 + t2,
                    stage_s=[t1, t2], stage_optimal=st_opt, stage_gap=st_gap,
                    next_eps=int(round(s.f1)) + 1)

    def _step_dv(self, eps):
        net1, s1, t1 = self._capped("max_v", min(eps, self.dvmax), "max_f1", eps)
        if s1 is None:
            return None
        tol = 0.5 if self.K else 1e-6
        s, t2, s2 = s1, 0.0, None
        if raw_dv_of(net1, s1) > 1e-9:
            net2, s2, t2 = self._capped("min_dV", min(eps, self.dvmax), "min_f2", eps,
                                        eps=s1.f1 - tol)
            if s2 is not None and raw_dv_of(net2, s2) <= raw_dv_of(net1, s1) + 1e-9:
                s = s2
        opt = bool(s1.meta.get("optimal")) and (s2 is None or bool(s2.meta.get("optimal")))
        gaps = [g for g in (s1.meta.get("mip_gap"), s2.meta.get("mip_gap") if s2 else None)
                if g is not None]
        d = raw_dv_of(net1, s)
        return dict(net=net1, sol=s, eps=eps, optimal=opt,
                    mip_gap=max(gaps) if gaps else None, solve_s=t1 + t2,
                    stage_s=[t1, t2], next_eps=(d - self.delta) if d > 1e-9 else None)

    # ------------------------------------------------------------ record
    def describe(self, r) -> dict:
        net, s = r["net"], r["sol"]
        d = net.describe_solution(s)
        ids, norad = self.data.ids, self.data.norad
        idx = {k: n for n, k in enumerate(ids)}
        date = lambda day: (self.epoch + _dt.timedelta(days=float(day) + self.day0)).isoformat()
        visits = []
        for m, v in enumerate(d["visits"], 1):
            e = {"seq": m, "debris": v["id"], "norad": int(norad[idx[v["id"]]])}
            if m > 1:
                dep = v["depart_prev_day"]
                e.update(arc_from=v["from"], arc_depart_day=dep + self.day0,
                         arc_depart_date=date(dep), tof_days=v["tof_days"],
                         arc_dv_mps=round(float(self.data.dv[idx[v["from"]], idx[v["id"]],
                                                             int(round(dep))]), 4))
                if self.elems is not None:
                    Ei = self.elems[v["from"]][int(round(dep + self.day0))]
                    Ej = self.elems[v["id"]][int(round(dep + self.day0))]
                    imp = combined_impulses(Ei[0], Ei[2], Ei[3], Ej[0], Ej[2], Ej[3])
                    e.update(theta_deg=round(imp["theta_deg"], 4),
                             plane_change_at=imp["plane_change_at"],
                             dv1_mps=round(imp["dv1_mps"], 3), dv2_mps=round(imp["dv2_mps"], 3))
            e.update(arrive_day=v["arrival_day"] + self.day0, arrive_date=date(v["arrival_day"]))
            if v["departure_day"] is not None:
                e.update(depart_day=v["departure_day"] + self.day0,
                         depart_date=date(v["departure_day"]), wait_days=v["wait_days"])
            else:
                e.update(depart_day=None, depart_date=None,
                         wait_days=None, stays_until_day=net.day(net.H) + self.day0)
            e.update(score=round(v["score"], 4), score_scaled=v.get("score_scaled"))
            visits.append(e)
        last = visits[-1]
        return {
            "method": self.method, "epsilon": r["eps"],
            "total_score_scaled": (int(round(s.f1)) if self.K else None),
            "total_score": round(d["total_score"], 4),
            "total_dv_mps": round(raw_dv_of(net, s), 6),
            "optimal": bool(r["optimal"]), "mip_gap": r["mip_gap"],
            "solve_time_s": round(r["solve_s"], 2),
            "stage_time_s": [round(x, 2) for x in r["stage_s"]],
            "stage_optimal": r.get("stage_optimal"), "stage_mip_gap": r.get("stage_gap"),
            # A7: stage 1 (min dV s.t. v' >= eps) not proven -> a lower-dV Pareto point
            # with v' in [eps, s_k] may have been missed
            "unproven_interval": ([r["eps"], int(round(s.f1))]
                                  if self.method == "score_exact" and r.get("stage_optimal")
                                  and not r["stage_optimal"][0] else None),
            "launch_day": d["launch_day"] + self.day0, "launch_date": date(d["launch_day"]),
            "mission_end_day": last["arrive_day"],
            "order": [v["debris"] for v in visits],
            "order_norad": [v["norad"] for v in visits],
            "visits": visits}

    # ------------------------------------------------------------ driver
    def _ckpt_path(self):
        return os.path.join(self.out, "checkpoint.json") if self.out else None

    def load(self):
        p = self._ckpt_path()
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
            if st.get("method") == self.method:
                return st
        return None

    def save(self, st, new_point=None):
        if not self.out:
            return
        os.makedirs(os.path.join(self.out, "solutions"), exist_ok=True)
        if new_point is not None and self.method == "score_exact":
            with open(os.path.join(self.out, "solutions", new_point["id"] + ".json"),
                      "w", encoding="utf-8") as f:
                json.dump(new_point, f, indent=1, ensure_ascii=False)
        p = self._ckpt_path()
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1, ensure_ascii=False)
        os.replace(p + ".tmp", p)

    def run(self, max_points=None, eps_start=None, eps_stop=None):
        """eps_start / eps_stop (score_exact, A8 range split): start at eps_start and
        stop once the next epsilon reaches eps_stop (the next range's start)."""
        st = self.load() or {"method": self.method, "K": self.K, "points": [],
                             "calls": [], "done": False, "t_total": 0.0,
                             "eps_start": eps_start, "eps_stop": eps_stop,
                             "next_eps": (eps_start if eps_start is not None else
                                          0 if self.method == "score_exact" else self.dvmax)}
        if eps_stop is not None:
            st["eps_stop"] = eps_stop
        if st["points"]:
            self.log(f"resume: {len(st['points'])} points, next eps = {st['next_eps']}")
        step = self._step_score if self.method == "score_exact" else self._step_dv
        n_new = 0
        while not st["done"] and st["next_eps"] is not None:
            if max_points is not None and n_new >= max_points:
                break
            if st.get("eps_stop") is not None and st["next_eps"] >= st["eps_stop"]:
                st["done"] = True
                self.save(st)
                self.log(f"next eps {st['next_eps']} >= range stop {st['eps_stop']} -> done")
                break
            eps = st["next_eps"]
            t0 = time.time()
            n0 = len(self.calls)
            r = step(eps)
            st["calls"] += self.calls[n0:]
            st["t_total"] += time.time() - t0
            if r is None:
                st["done"] = True
                self.save(st)
                self.log(f"eps={eps}: infeasible -> front complete")
                break
            pt = self.describe(r)
            pt["id"] = f"S{len(st['points']) + 1:03d}"
            pt = {"id": pt.pop("id"), **pt}
            st["points"].append(pt)
            st["next_eps"] = r["next_eps"]
            if st["next_eps"] is None:
                st["done"] = True
            self.save(st, pt)
            n_new += 1
            self.log(f"{pt['id']} eps={eps:.6g}  dV={pt['total_dv_mps']:9.3f}  "
                     f"score={pt['total_score']:8.3f} ({pt['total_score_scaled']})  "
                     f"n={len(pt['order'])}  {'-'.join(o[1:] for o in pt['order'])}  "
                     f"opt={pt['optimal']} gap={pt['mip_gap']}  "
                     f"{pt['solve_time_s']:.1f}s [{', '.join(f'{x:.1f}' for x in pt['stage_time_s'])}]")
        return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default=None, choices=["score_exact", "dv_exact"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--outputs-only", action="store_true")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--eps-start", type=float, default=None, help="A8 range: first eps")
    ap.add_argument("--eps-stop", type=float, default=None, help="A8 range: stop at this eps")
    ap.add_argument("--threads", type=int, default=None, help="CPLEX threads (A8)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.threads:
        cfg["pareto"]["threads"] = args.threads
    out = args.out or cfg["output"]["exact_dir"]
    if not args.outputs_only:
        data = load_dataset(cfg)
        solver = setup_solver(cfg)
        elems = load_mean_elements(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["out_dir"]))
        ef = ExactFront(data, cfg, method=args.method, out_dir=out, elems=elems,
                        log=lambda *a: print(*a, flush=True))
        print(f"solver={solver} method={ef.method} K={ef.K} dV_max={ef.dvmax:.3f} "
              f"gap={cfg['pareto'].get('mip_gap')} time_limit={ef.tl} -> {out}", flush=True)
        ef.run(max_points=args.max_points, eps_start=args.eps_start, eps_stop=args.eps_stop)
    if args.eps_start is None and args.eps_stop is None:   # range runs: merge later
        from .pareto_outputs import write_all
        write_all(out, cfg)


if __name__ == "__main__":
    main()

"""
M3: NSGA-II + random-key encoding (Choi et al. 2024 style), pymoo 0.6.1.3.

Genome (real-coded, integers rounded in the decoder):
  keys[0:n]      in [0,1]   key < 0.5 -> debris excluded; included debris visited in
                            ascending key order (random-key order + selection)
  launch         in [0,180] launch day = arrival day at the first debris
  wait[0:n-1]    in [0,Wmax] days waited at the k-th visited debris before the next
                            transfer, Wmax = horizon - launch window end (= 915 d)
Decoder: transfer 1 day (as the TEG), dV = dV_ij(departure day), v' = round(K score_j(arrival)).
Constraints (pymoo feasibility-first / constraint-domination, not a penalty):
  g1 = sum dV - dV_max <= 0,  g2 = days past day 1095 (+ NaN arc = re-entered) <= 0.
Objectives: min sum dV, min -sum v'/K.
Reported front = non-dominated set of ALL feasible evaluated solutions (external archive,
the most favourable reading for M3); the final population's front is also kept.
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.max_time import TimeBasedTermination

from bench.common import K, PeakMem, nondominated, pt


class RKProblem(Problem):
    def __init__(self, vs, dv, dvmax, launch_hi=180):
        self.vs, self.dv, self.dvmax = vs, dv, dvmax
        self.nd, self.H = vs.shape[0], vs.shape[1] - 1
        self.wmax = self.H - launch_hi
        n = self.nd
        xl = np.r_[np.zeros(n), 0, np.zeros(n - 1)]
        xu = np.r_[np.ones(n), launch_hi, np.full(n - 1, self.wmax)]
        super().__init__(n_var=2 * n, n_obj=2, n_ieq_constr=2, xl=xl, xu=xu)
        self.archive = []
        self.n_feasible = 0

    def decode(self, x):
        n = self.nd
        keys = x[:n]
        sel = [i for i in np.argsort(keys, kind="stable") if keys[i] >= 0.5]
        L = int(round(x[n]))
        w = np.rint(x[n + 1:]).astype(int)
        if not sel:
            return 0.0, 0, 0.0, 0.0
        t = L
        s = int(self.vs[sel[0], t])
        d, over = 0.0, 0.0
        for k, j in enumerate(sel[1:]):
            dep = t + w[k]
            arr = dep + 1
            if arr > self.H:
                over += arr - self.H
                break
            c = self.dv[sel[k], j, dep]
            if not np.isfinite(c):
                over += 1000.0
                break
            d += float(c)
            s += int(self.vs[j, arr])
            t = arr
        return d, s, d - self.dvmax, over

    def _evaluate(self, X, out, *args, **kw):
        F = np.empty((len(X), 2))
        G = np.empty((len(X), 2))
        for r, x in enumerate(X):
            d, s, g1, g2 = self.decode(x)
            F[r] = (d, -s / K)
            G[r] = (g1, g2)
            if g1 <= 1e-9 and g2 <= 0:
                self.n_feasible += 1
                self.archive.append((d, s))
        if len(self.archive) > 5000:
            self.archive = [(p["dv"], p["vs"]) for p in
                            nondominated([pt(a, b) for a, b in self.archive])]
        out["F"], out["G"] = F, G


def m3(sd, dvmax, seed, n_gen=250, pop=200, time_s=None):
    vs = np.round(K * sd.score).astype(np.int64)
    prob = RKProblem(vs, sd.dv, dvmax)
    alg = NSGA2(pop_size=pop)
    term = TimeBasedTermination(max_time=time_s) if time_s else get_termination("n_gen", n_gen)
    with PeakMem() as pm:
        t0 = time.time()
        res = minimize(prob, alg, term, seed=seed, verbose=False)
        wall = time.time() - t0
    arch = nondominated([pt(a, b) for a, b in prob.archive])
    final = []
    if res.F is not None:
        F = np.atleast_2d(res.F)
        G = np.atleast_2d(res.G) if res.G is not None else np.zeros((len(F), 2))
        final = nondominated([pt(f[0], int(round(-f[1] * K))) for f, g in zip(F, G)
                              if (g <= 1e-9).all()])
    return {"points": arch, "final_pop_points": final, "wall_s": wall,
            "n_evals": int(res.algorithm.evaluator.n_eval), "n_feasible": prob.n_feasible,
            "peak_mb": pm.peak, "timeout": False, "seed": seed,
            "setting": "time" if time_s else f"gen{n_gen}", "time_budget_s": time_s}

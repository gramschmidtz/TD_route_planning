"""
M4: exact label-setting DP over (node, visited mask, day).

Same algorithm and conventions as verify/common.dp_front (launch arc dV 0 with the
launch-day score, transfer 1 day, v at arrival, waiting free, end anywhere,
sum dV <= dV_max), with two changes for speed only:
  * labels are kept per day layer (all moves go t -> t+1), instead of scanning every
    stored key each day (verify version is O(days x keys));
  * scores are the integer v' = round(K score) used by every method in the benchmark.
A wall-clock limit aborts the run (reported as timeout).
"""
from __future__ import annotations

import time

import numpy as np

from bench.common import K, PeakMem, nondominated, pt


def _add(L, s, d):
    for (s2, d2) in L:
        if s2 >= s and d2 <= d + 1e-12:
            return False
    L[:] = [(s2, d2) for (s2, d2) in L if not (s >= s2 and d <= d2 + 1e-12)]
    L.append((s, d))
    return True


def m4(sd, dvmax, launch_hi=180, time_limit=7200.0):
    vs = np.round(K * sd.score).astype(np.int64)
    dv = sd.dv
    n, H = vs.shape[0], vs.shape[1] - 1
    labels_made = 0
    archive = []                       # global non-dominated (s, d)
    timeout = False
    with PeakMem() as pm:
        t0 = time.time()
        layer = {}
        for tau in range(0, H + 1):
            if time.time() - t0 > time_limit:
                timeout = True
                break
            if tau <= launch_hi:
                for j in range(n):
                    if vs[j, tau] > 0:
                        labels_made += _add(layer.setdefault((j, 1 << j), []), int(vs[j, tau]), 0.0)
            for L in layer.values():
                for (s, d) in L:
                    _add(archive, s, d)
            if tau == H:
                break
            nxt = {}
            for (j, mask), L in layer.items():
                tgt = nxt.setdefault((j, mask), [])           # wait one day
                for (s, d) in L:
                    labels_made += _add(tgt, s, d)
                for k in range(n):
                    if mask >> k & 1:
                        continue
                    c = dv[j, k, tau]
                    if not np.isfinite(c) or c > dvmax:
                        continue
                    v = int(vs[k, tau + 1])
                    tgt = nxt.setdefault((k, mask | 1 << k), [])
                    for (s, d) in L:
                        if d + c <= dvmax + 1e-9:
                            labels_made += _add(tgt, s + v, d + float(c))
            layer = nxt
        wall = time.time() - t0
    pts = [pt(d, s) for (s, d) in archive]
    return {"points": nondominated(pts), "wall_s": wall, "n_labels": labels_made,
            "peak_mb": pm.peak, "timeout": timeout, "time_limit_s": time_limit,
            "days_done": tau}

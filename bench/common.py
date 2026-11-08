"""
Shared pieces of the method-comparison benchmark (bench/).

Every method optimises the SAME pair of objectives as M0:
    max  sum v'   (v' = round(K * score_j(arrival day)), K = 10)   -> reported as score = sum v' / K
    min  sum dV   [m/s]
so fronts can be compared point by point (equal v' and |dV diff| <= 1e-3 m/s).
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time

import numpy as np
import psutil

from adr_data.adapter import dv_max_mps
from adr_data.load import ADRData, load_config, load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")
FRONTS = os.path.join(OUT, "fronts")
# engine checkpoints are rewritten after every point: keep them off OneDrive (file locks)
WORK = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "td_route_bench_work")
K = 10
TOL_DV, TOL_SC = 1e-3, 1e-3
LONG = ["D01", "D02", "D03", "D04", "D05", "D06"]
REENTRY = ["D07", "D08", "D09", "D10"]


def load():
    cfg = load_config()
    return cfg, load_dataset(cfg)


def sub_instance(data: ADRData, cfg: dict, ids, n_days=None, threads=1, time_limit=600):
    """Debris subset, same days/settings as the main run (horizon from config)."""
    n_days = n_days or int(cfg["mission"]["horizon_days"])
    idx = [data.ids.index(i) for i in ids]
    sd = ADRData(ids=list(ids), norad=[data.norad[k] for k in idx],
                 score=data.score[idx][:, :n_days].copy(),
                 dv=data.dv[np.ix_(idx, idx)][:, :, :n_days].copy(),
                 dv_table=data.dv_table,
                 meta={"debris": [data.meta["debris"][k] for k in idx]}, root=data.root)
    sc = copy.deepcopy(cfg)
    sc["mission"]["horizon_days"] = n_days
    sc["pareto"].update(threads=threads, time_limit=time_limit, mip_gap=0,
                        score_scale=K)
    return sd, sc


def nondominated(pts):
    """pts: list of dict(dv, vs). Keep non-dominated (min dv, max vs), unique."""
    out = []
    for p in sorted(pts, key=lambda q: (q["dv"], -q["vs"])):
        if out and out[-1]["vs"] >= p["vs"]:
            continue
        if out and abs(out[-1]["dv"] - p["dv"]) <= 1e-9:
            out[-1] = p
            continue
        out.append(p)
    return out


def pt(dv, vs, **kw):
    d = {"dv": float(dv), "vs": int(vs), "score": int(vs) / K}
    d.update(kw)
    return d


class PeakMem:
    """Peak RSS [MB] of this process + children (CPLEX), sampled every 0.1 s."""

    def __enter__(self):
        self.p = psutil.Process(os.getpid())
        self.peak = 0.0
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _sample(self):
        try:
            r = self.p.memory_info().rss
            for c in self.p.children(recursive=True):
                try:
                    r += c.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, r / 2 ** 20)
        except psutil.Error:
            pass

    def _run(self):
        while not self._stop:
            self._sample()
            time.sleep(0.1)

    def __exit__(self, *a):
        self._stop = True
        self.t.join()
        self._sample()


def save_front(inst, method, res):
    os.makedirs(FRONTS, exist_ok=True)
    path = os.path.join(FRONTS, f"{inst}__{method}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k in range(20):                      # OneDrive may hold a lock briefly
        try:
            os.replace(path + ".tmp", path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    raise


def load_front(inst, method):
    path = os.path.join(FRONTS, f"{inst}__{method}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def instances():
    with open(os.path.join(ROOT, "bench", "instances.json"), encoding="utf-8") as f:
        return json.load(f)

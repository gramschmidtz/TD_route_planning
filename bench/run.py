"""
Run the benchmark methods on instances (resumable: a finished front file is skipped).

    python -W ignore -m bench.run --inst n4_s0 n4_s1 --methods M0 M4 M1 M2 M5 M3 --threads 1

Order inside one instance: M0 first (its wall time is M3's time budget).
Method keys: M0, M1_11, M1_101, M2_100, M2_50, M2_10, M3_gen_s<seed>, M3_time_s<seed>, M4, M5.
"""
from __future__ import annotations

import argparse
import os
import platform
import time

from adr_data.adapter import dv_max_mps

from bench import methods as M
from bench.common import instances, load, load_front, save_front, sub_instance
from bench.dp import m4
from bench.nsga import m3

SEEDS = range(10)


def machine():
    import psutil
    cpu = platform.processor()
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True,
                             timeout=10).stdout.splitlines()
        cpu = next((l.strip() for l in out[1:] if l.strip()), cpu)
    except Exception:
        pass
    return {"host": platform.node(), "cpu": cpu, "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(), "os": platform.platform(),
            "python": platform.python_version()}


MACHINE = None


def run_one(inst, key, fn):
    global MACHINE
    if load_front(inst, key) is not None:
        return load_front(inst, key)
    MACHINE = MACHINE or machine()
    t = time.time()
    res = fn()
    res.update(instance=inst, method=key, machine=MACHINE,
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_front(inst, key, res)
    print(f"{inst:7s} {key:12s} pts={len(res['points']):4d} wall={res['wall_s']:8.1f}s "
          f"timeout={res.get('timeout')}  ({time.time() - t:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", nargs="*", default=[])
    ap.add_argument("--methods", nargs="+", default=["M0", "M4", "M5", "M3gen", "M1", "M2"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--dp-limit", type=float, default=7200)
    ap.add_argument("--calib", action="store_true",
                    help="machine calibration: re-run M0 and M4 of n4_s0 as CALIB_<host>_*")
    a = ap.parse_args()
    cfg, data = load()
    dvmax = dv_max_mps(cfg)
    # CPLEX: env CPLEX_PATH overrides the config; never fall back to CBC silently
    cp = os.environ.get("CPLEX_PATH") or cfg["pareto"].get("cplex_path")
    if not (cp and os.path.exists(cp)):
        raise SystemExit(f"CPLEX not found ({cp}). Set CPLEX_PATH to cplex(.exe).")
    cfg["pareto"]["cplex_path"] = os.path.abspath(cp)
    cfg["pareto"]["solver"] = "cplex"
    # CPLEX writes cplex.log into the cwd: give every bench process its own cwd
    from bench.common import WORK
    cwd = os.path.join(WORK, f"cwd_{os.getpid()}")
    os.makedirs(cwd, exist_ok=True)
    os.chdir(cwd)
    by = {i["name"]: i for i in instances()}
    if a.calib:
        tag = "CALIB_" + platform.node().replace("_", "-")
        sd, sc = sub_instance(data, cfg, by["n4_s0"]["ids"], threads=a.threads,
                              time_limit=a.time_limit)
        run_one("n4_s0", f"{tag}_M0", lambda: M._exact_run(M.ExactFront, sd, sc, "n4_s0", tag))
        run_one("n4_s0", f"{tag}_M4", lambda: m4(sd, dvmax, time_limit=a.dp_limit))
    for name in a.inst or []:
        ins = by[name]
        sd, sc = sub_instance(data, cfg, ins["ids"], threads=a.threads, time_limit=a.time_limit)
        r0 = run_one(name, "M0", lambda: M.m0(sd, sc, name))
        for m in a.methods:
            if m == "M4":
                run_one(name, "M4", lambda: m4(sd, dvmax, time_limit=a.dp_limit))
            elif m == "M1":
                for nl in (11, 101):
                    run_one(name, f"M1_{nl}", lambda: M.m1(sd, sc, nl))
            elif m == "M2":
                for d in (100, 50, 10):
                    run_one(name, f"M2_{d}", lambda: M.m2(sd, sc, d))
            elif m == "M5":
                run_one(name, "M5", lambda: M.m5(sd, sc, name))
            elif m in ("M3", "M3gen"):
                for s in SEEDS:
                    run_one(name, f"M3_gen_s{s}", lambda: m3(sd, dvmax, s))
            if m in ("M3", "M3time"):
                for s in SEEDS:
                    run_one(name, f"M3_time_s{s}", lambda: m3(sd, dvmax, s, time_s=r0["wall_s"]))


if __name__ == "__main__":
    main()

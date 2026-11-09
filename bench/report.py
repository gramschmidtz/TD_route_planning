"""
Metrics, figures and summary of the method comparison.

    python -W ignore -m bench.report
      -> bench/out/results.csv, summary_table.md, crosscheck.json,
         fig_front_example.png/.svg, fig_scaling.png/.svg, fig_quality.png/.svg

P* = M0 front of the instance. Normalisation: dV / dV_max, score / max score of P*.
HV in minimisation form (dV_n, 1 - score_n) with reference point (1, 1) = (dV_max, 0).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adr_data.adapter import dv_max_mps
from adr_data.load import load_config

from bench.common import FRONTS, OUT, TOL_DV, TOL_SC, WORK, instances


def m0_unproven(name):
    """A9 addendum 1: M0 points whose stage 1 or 2 hit the 600 s limit (with gap)."""
    f = os.path.join(WORK, name, "M0", "checkpoint.json")
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        st = json.load(fh)
    bad = [{"epsilon": p["epsilon"], "v": p["total_score_scaled"], "dv": p["total_dv_mps"],
            "stage_optimal": p.get("stage_optimal"), "stage_mip_gap": p.get("stage_mip_gap"),
            "unproven_interval": p.get("unproven_interval")}
           for p in st["points"] if not p["optimal"]]
    last = st["calls"][-1] if st.get("calls") else None
    return {"n_points": len(st["points"]), "not_optimal_points": bad,
            "end_call_optimal": None if last is None else last.get("optimal"),
            "end_call_s": None if last is None else last.get("s")}

NAVY, RED, GREY = "#19264B", "#C01F2F", "#9AA3B2"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 14, "legend.fontsize": 10,
                     "svg.fonttype": "none"})
DVMAX = dv_max_mps(load_config())


# ------------------------------------------------------------------ metrics
def supported_mask(P):
    """Supported = on the upper concave envelope of (dV, score) (collinear counts)."""
    xs = [p["dv"] for p in P]
    ys = [p["score"] for p in P]
    hull = []
    for i in range(len(P)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (xs[b] - xs[a]) * (ys[i] - ys[a]) - (ys[b] - ys[a]) * (xs[i] - xs[a])
            if cross >= 0:            # b is below segment a-i -> not on upper hull
                hull.pop()
            else:
                break
        hull.append(i)
    sup = [False] * len(P)
    for a, b in zip(hull, hull[1:]):
        for i in range(a, b + 1):
            if xs[b] == xs[a]:
                continue
            yline = ys[a] + (ys[b] - ys[a]) * (xs[i] - xs[a]) / (xs[b] - xs[a])
            if ys[i] >= yline - 1e-9:
                sup[i] = True
    for i in hull:
        sup[i] = True
    return sup


def match(P, Q):
    return [any(abs(p["dv"] - q["dv"]) <= TOL_DV and abs(p["score"] - q["score"]) <= TOL_SC
                for q in Q) for p in P]


def norm(points, smax):
    return [(p["dv"] / DVMAX, 1.0 - p["score"] / smax) for p in points]


def hv(points, smax):
    pts = sorted([z for z in norm(points, smax) if z[0] <= 1 and z[1] <= 1])
    h, best = 0.0, 1.0
    for (x, y) in pts:
        if y < best:
            h += (1 - x) * (best - y)
            best = y
    return h


def igd_plus(P, Q, smax):
    if not Q:
        return float("nan")
    Z, A = norm(P, smax), norm(Q, smax)
    return float(np.mean([min(np.hypot(max(a[0] - z[0], 0), max(a[1] - z[1], 0)) for a in A)
                          for z in Z]))


def best_at(points, B):
    v = [p["score"] for p in points if p["dv"] <= B + 1e-9]
    return max(v) if v else 0.0


def losses(P, Q):
    Bs = sorted({p["dv"] for p in P})
    d = [best_at(P, B) - best_at(Q, B + TOL_DV) for B in Bs]   # same-point tolerance
    return max(d), float(np.mean(d))


def metrics(P, sup, res):
    Q = res["points"]
    smax = max(p["score"] for p in P)
    m = match(P, Q)
    ns = [i for i, s in enumerate(sup) if not s]
    wl, ml = losses(P, Q)
    return {"n_pstar": len(P), "n_found": len(Q), "recovered": sum(m),
            "recovery": sum(m) / len(P), "n_nonsupported": len(ns),
            "nonsupported_found": sum(m[i] for i in ns),
            "hv_ratio": hv(Q, smax) / hv(P, smax), "igd_plus": igd_plus(P, Q, smax),
            "worst_loss": wl, "mean_loss": ml, "wall_s": res["wall_s"],
            "work": res.get("n_solves", res.get("n_evals", res.get("n_labels"))),
            "work_kind": ("milp_solves" if "n_solves" in res else
                          "evals" if "n_evals" in res else "labels"),
            "peak_mb": res.get("peak_mb"), "timeout": bool(res.get("timeout")),
            # fronts written before machine tagging were all computed on the main PC
            "machine": (res.get("machine") or {}).get("host", "Taeuk_Desktop")}


# ------------------------------------------------------------------ collect
def load_all():
    data = {}
    for f in glob.glob(os.path.join(FRONTS, "*.json")):
        inst, meth = os.path.basename(f)[:-5].split("__")
        with open(f, encoding="utf-8") as fh:
            data.setdefault(inst, {})[meth] = json.load(fh)
    return data


def family(m):
    if m.startswith("M3_gen"):
        return "M3_gen"
    if m.startswith("M3_time"):
        return "M3_time"
    return m


# M4 (DP) is excluded from the comparison (user decision 2026-11-09); its fronts, where they
# exist, are only used in crosscheck.json as a check of P*.
# M3_time (NSGA-II with M0's wall time) is excluded as well (user decision 2026-11-09).
ORDER = ["M0", "M1_11", "M1_101", "M2_100", "M2_50", "M2_10", "M3_gen", "M5"]
EXCLUDED = ("M4", "M3_time")
COL = {"M0": NAVY, "M1_11": "#E0A030", "M1_101": "#B07818", "M2_100": "#8FB3D9",
       "M2_50": "#4F7FB8", "M2_10": "#2D5591", "M3_gen": RED, "M3_time": "#E07A84",
       "M4": GREY, "M5": "#5E9E6E"}


def main():
    os.makedirs(OUT, exist_ok=True)
    inst = {i["name"]: i for i in instances()}
    D = load_all()
    rows, cross = [], {}
    for name in sorted(D, key=lambda s: (inst[s]["n"], inst[s]["seed"])):
        if "M0" not in D[name]:
            continue
        P = D[name]["M0"]["points"]
        sup = supported_mask(P)
        cross.setdefault(name, {})["m0_unproven"] = m0_unproven(name)
        cross[name]["n_supported"] = sum(sup)
        cross[name]["n_nonsupported"] = len(sup) - sum(sup)
        if "M4" in D[name] and not D[name]["M4"].get("skipped"):   # skip placeholder files
            r4 = D[name]["M4"]
            a, b = match(P, r4["points"]), match(r4["points"], P)
            cross[name].update({"m0_points": len(P), "m4_points": len(r4["points"]),
                           "m4_timeout": r4["timeout"],
                           "identical": (not r4["timeout"]) and all(a) and all(b),
                           "m0_missing_vs_m4": sum(not x for x in b) if not r4["timeout"] else None})
        for meth, res in D[name].items():
            if meth.startswith("CALIB_") or family(meth) in EXCLUDED:
                continue
            m = metrics(P, sup, res)
            rows.append({"instance": name, "n": inst[name]["n"], "set": inst[name]["set"],
                         "method": family(meth), "variant": meth, "seed": res.get("seed"),
                         **m})
    keys = list(rows[0].keys()) if rows else []
    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "crosscheck.json"), "w") as f:
        json.dump(cross, f, indent=1)
    summary(rows)
    fig_example(D, inst)
    fig_scaling(rows)
    fig_quality(rows)
    print(json.dumps(cross, indent=1))


def _agg(rows, meth, n, key, fn=stats.mean):
    v = [r[key] for r in rows if r["method"] == meth and r["n"] == n and r[key] is not None
         and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return fn(v) if v else None


def summary(rows):
    ns = sorted({r["n"] for r in rows})
    L = ["| 방법 | 회수율 (n 평균) | 비지지점 회수 | HV 비율 | IGD+ | 최악 손실 [점] | 시간 [s] " +
         " ".join(f"| n={n}" for n in ns) + " |",
         "|---|---|---|---|---|---|---" + "|---" * len(ns) + "|"]
    for m in ORDER:
        R = [r for r in rows if r["method"] == m]
        if not R:
            continue
        rec = stats.mean(r["recovery"] for r in R)
        nsf = sum(r["nonsupported_found"] for r in R)
        nst = sum(r["n_nonsupported"] for r in R)
        hvr = stats.mean(r["hv_ratio"] for r in R)
        igd = stats.mean(r["igd_plus"] for r in R if not np.isnan(r["igd_plus"]))
        wl = stats.mean(r["worst_loss"] for r in R)
        t = []
        for n in ns:
            v = _agg(rows, m, n, "wall_s")
            to = any(r["timeout"] for r in R if r["n"] == n)
            t.append("—" if v is None else f"{v:,.1f}{'*' if to else ''}")
        L.append(f"| {m} | {100 * rec:.1f} % | {nsf}/{nst} | {hvr:.4f} | {igd:.4f} | {wl:.2f} | "
                 + " | ".join(t) + " |")
    L += ["", "회수율·HV·IGD+·최악 손실은 인스턴스(와 M3 시드) 평균. 시간 열은 n별 인스턴스 평균 벽시계 시간, "
          "* = 시간 제한에 걸린 풀이/실행 포함. 비지지점 회수 = 찾은 비지지점 / P* 의 비지지점 합계."]
    with open(os.path.join(OUT, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def fig_example(D, inst, name="n6_s0"):
    if name not in D or "M0" not in D[name]:
        cands = [k for k in D if "M0" in D[k]]
        if not cands:
            return
        name = max(cands, key=lambda k: inst[k]["n"])
    d = D[name]
    P = d["M0"]["points"]
    sup = supported_mask(P)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.step([p["dv"] for p in P], [p["score"] for p in P], where="post", color=NAVY, lw=1.0, alpha=0.6)
    ax.scatter([p["dv"] for p, s in zip(P, sup) if s], [p["score"] for p, s in zip(P, sup) if s],
               s=26, color=NAVY, label="P* supported (M0)", zorder=3)
    ax.scatter([p["dv"] for p, s in zip(P, sup) if not s], [p["score"] for p, s in zip(P, sup) if not s],
               s=26, facecolor="white", edgecolor=NAVY, lw=1.1, label="P* non-supported (M0)", zorder=3)
    over = [("M1_11", "M1 weighted sum (11 λ)", "D", 34), ("M2_50", "M2 ΔV-grid ε (δ = 50)", "s", 30),
            ("M3_gen_s0", "M3 NSGA-II (seed 0)", "x", 30), ("M5", "M5 no waiting", "^", 26)]
    for key, lab, mk, sz in over:
        if key in d:
            Q = d[key]["points"]
            c = COL[family(key)]
            kw = dict(facecolor="none", edgecolor=c) if mk not in ("x",) else dict(color=c)
            ax.scatter([q["dv"] for q in Q], [q["score"] for q in Q], marker=mk, s=sz, lw=1.2,
                       label=lab, zorder=4, **kw)
    ax.set_xlabel("Total ΔV [m/s]")
    ax.set_ylabel("Total removal score")
    _style(ax)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.01, 0.98, f"{name}: {', '.join(inst[name]['ids'])}", transform=ax.transAxes,
            va="top", fontsize=9, color=GREY)
    _save(fig, "fig_front_example")


def fig_scaling(rows):
    ns = sorted({r["n"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m, lab in [("M0", "M0 proposed"), ("M2_10", "M2 ΔV-grid ε (δ = 10)"),
                   ("M3_gen", "M3 NSGA-II (pop 200 × 250 gen)"), ("M5", "M5 no waiting")]:
        xs, ys, to = [], [], []
        for n in ns:
            v = _agg(rows, m, n, "wall_s")
            if v is None:
                continue
            xs.append(n)
            ys.append(v)
            to.append(any(r["timeout"] for r in rows if r["method"] == m and r["n"] == n))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=COL[m], lw=2, label=lab)
        for x, y, t in zip(xs, ys, to):
            if t:
                ax.scatter([x], [y], s=120, facecolor="none", edgecolor=RED, lw=1.6, zorder=5)
    ax.scatter([], [], s=120, facecolor="none", edgecolor=RED, lw=1.6, label="hit time limit")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("Number of debris n")
    ax.set_ylabel("Wall-clock time [s]")
    _style(ax)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig_scaling")


def fig_quality(rows):
    ns = sorted({r["n"] for r in rows})
    meths = [m for m in ORDER if m != "M0" and any(r["method"] == m for r in rows)]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.8), sharex=True)
    w = 0.8 / max(1, len(meths))
    for ax, key, lab in [(axs[0], "recovery", "Exact-point recovery"), (axs[1], "hv_ratio", "HV / HV(P*)")]:
        for k, m in enumerate(meths):
            v = [_agg(rows, m, n, key) for n in ns]
            ax.bar([i + (k - (len(meths) - 1) / 2) * w for i in range(len(ns))],
                   [0 if x is None else x for x in v], width=w, color=COL[m], label=m)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"n={n}" for n in ns])
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.05)
        _style(ax)
    axs[1].set_ylim(min(0.9, min((r["hv_ratio"] for r in rows), default=0.9) - 0.02), 1.005)
    axs[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _save(fig, "fig_quality")


if __name__ == "__main__":
    main()

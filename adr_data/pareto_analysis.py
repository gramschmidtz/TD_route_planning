"""
A9 analyses for PARETO_REPORT.md (read-only on the merged front):

  1. knee check  - last 7-removal point vs first 8-removal point, side by side
  2. per-debris  - inclusion share, first-visit distribution, debris never in
                   the max-removal solutions and why (re-entry / score / arcs)
  3. ToF pairs   - inside one removal count: ToF ~600 d vs ~255 d solutions,
                   dV saved by waiting vs score lost

    python -m adr_data.pareto_analysis [--out results/pareto_fy1c_20260605_K10]
        -> <out>/analysis.json, <out>/analysis.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from .load import load_config, load_dataset
from .pareto_outputs import load_points

SHORT_TOF_MAX = 400.0          # ToF < 400 d = "short" (the ~125/255 d windows)


def tof(p):
    return p["visits"][-1]["arrive_day"] - p["launch_day"]


def nrem(p):
    return len(p["order"])


# --------------------------------------------------------------------- 1. knee
def knee(pts):
    front = sorted([p for p in pts if p["pareto"]], key=lambda p: p["total_dv_mps"])
    top = max(nrem(p) for p in front)
    a = max((p for p in front if nrem(p) == top - 1), key=lambda p: p["total_dv_mps"])
    b = min((p for p in front if nrem(p) == top), key=lambda p: p["total_dv_mps"])

    def arcs(p):
        out = {}
        for v in p["visits"][1:]:
            out[(v["arc_from"], v["debris"])] = v
        return out

    aa, bb = arcs(a), arcs(b)
    common = sorted(set(aa) & set(bb), key=lambda k: [v["debris"] for v in b["visits"]].index(k[1]))
    rows = []
    for k in common:
        rows.append({"arc": f"{k[0]}->{k[1]}",
                     "A_depart": aa[k]["arc_depart_day"], "B_depart": bb[k]["arc_depart_day"],
                     "A_dv": aa[k]["arc_dv_mps"], "B_dv": bb[k]["arc_dv_mps"]})
    only_a = [{"arc": f"{k[0]}->{k[1]}", "depart": aa[k]["arc_depart_day"], "dv": aa[k]["arc_dv_mps"]}
              for k in aa if k not in bb]
    only_b = [{"arc": f"{k[0]}->{k[1]}", "depart": bb[k]["arc_depart_day"], "dv": bb[k]["arc_dv_mps"]}
              for k in bb if k not in aa]
    added = [d for d in b["order"] if d not in a["order"]]
    return {"A": a, "B": b, "common_arcs": rows, "arcs_only_A": only_a, "arcs_only_B": only_b,
            "added_debris": added,
            "d_dv": b["total_dv_mps"] - a["total_dv_mps"],
            "d_score": b["total_score"] - a["total_score"]}


# --------------------------------------------------------------- 2. per debris
def per_debris(pts, data, launch_window):
    front = [p for p in pts if p["pareto"]]
    n = len(front)
    ids = data.ids
    incl = {d: sum(d in p["order"] for p in front) / n for d in ids}
    first = collections.Counter(p["order"][0] for p in front)
    top = max(nrem(p) for p in front)
    tops = [p for p in front if nrem(p) == top]
    always_out = [d for d in ids if not any(d in p["order"] for p in tops)]
    H = data.score.shape[1]
    reasons = {}
    for d in ids:
        j = ids.index(d)
        re = data.reentry_day(j)
        last = int(min(H, re)) if re is not None else H
        sc = data.score[j, :last]
        dv_in = np.delete(data.dv[:, j, :last], j, axis=0)    # others -> d
        dv_out = np.delete(data.dv[j, :, :last], j, axis=0)   # d -> others
        # cheapest arc into / out of d in the first 2 years (where 5+ removals end)
        lim = min(last, 730)
        reasons[d] = {
            "reentry_day": re,
            "score_mean": float(np.nanmean(sc)) if len(sc) else None,
            "score_day0": float(data.score[j, 0]),
            "score_rank_day0": int(1 + sum(data.score[:, 0] > data.score[j, 0])),
            "min_dv_in_mps": float(np.nanmin(dv_in[:, :lim])),
            "min_dv_out_mps": float(np.nanmin(dv_out[:, :lim])),
            "median_min_dv_in_mps": float(np.nanmedian(np.nanmin(dv_in[:, :lim], axis=0))),
            "days_in_below_150": int(np.sum(np.nanmin(dv_in[:, :lim], axis=0) < 150)),
        }
    return {"n_front": n, "inclusion": incl, "first_visit": dict(first.most_common()),
            "max_removal": top, "n_max_removal_points": len(tops),
            "always_out_of_max": always_out, "debris_facts": reasons}


# ----------------------------------------------------------------- 3. ToF pairs
def tof_pairs(pts):
    front = [p for p in pts if p["pareto"]]
    out = []
    for k in sorted({nrem(p) for p in front}):
        g = [p for p in front if nrem(p) == k]
        short = [p for p in g if tof(p) < SHORT_TOF_MAX]
        long_ = [p for p in g if tof(p) >= SHORT_TOF_MAX]
        if not short or not long_:
            continue
        # pairs: each short-ToF point s with the long-ToF point l just below it in dV
        # (on the front, lower dV => lower score: l saves dV by waiting, loses score)
        pairs = []
        for s in short:
            below = [l for l in long_ if l["total_dv_mps"] < s["total_dv_mps"]]
            if not below:
                continue
            same = [l for l in below if l["order"] == s["order"]]
            l = max(same or below, key=lambda p: p["total_dv_mps"])
            d_dv = s["total_dv_mps"] - l["total_dv_mps"]
            d_sc = s["total_score"] - l["total_score"]
            pairs.append({"long": l, "short": s, "same_order": bool(same),
                          "dv_saved": d_dv, "score_lost": d_sc, "days_longer": tof(l) - tof(s),
                          "mps_per_point": d_dv / d_sc if d_sc > 0 else None})
        if not pairs:
            continue
        rep = min(pairs, key=lambda q: (not q["same_order"], q["dv_saved"]))
        r = [q["mps_per_point"] for q in pairs if q["mps_per_point"] is not None]
        out.append({"n_removed": k, "n_short": len(short), "n_long": len(long_),
                    "rep": rep, "n_pairs": len(pairs),
                    "mps_per_point_median": float(np.median(r)) if r else None,
                    "mps_per_point_range": [float(min(r)), float(max(r))] if r else None,
                    "pairs": pairs})
    return out


# ---------------------------------------------------------------------- report
def _visit_line(p):
    return " → ".join(f"{v['debris']}({v['arrive_date']})" for v in p["visits"])


def to_markdown(kn, pdb, tp):
    L = []
    a, b = kn["A"], kn["B"]
    L += ["### 무릎점 확인: 7개 제거 마지막 점 vs 8개 제거 첫 점", "",
          "| | 해 | ΔV [m/s] | 점수 | ToF [일] | 발사 | 순서 |", "|---|---|---|---|---|---|---|",
          f"| A | {a['id']} | {a['total_dv_mps']:.1f} | {a['total_score']:.2f} | {tof(a):.0f} | {a['launch_date']} | {'-'.join(a['order'])} |",
          f"| B | {b['id']} | {b['total_dv_mps']:.1f} | {b['total_score']:.2f} | {tof(b):.0f} | {b['launch_date']} | {'-'.join(b['order'])} |",
          "", f"ΔV 차이 {kn['d_dv']:+.2f} m/s, 점수 차이 {kn['d_score']:+.2f}, 추가 파편 {', '.join(kn['added_debris'])}.", "",
          "방문(도착일):", "", f"- A: {_visit_line(a)}", f"- B: {_visit_line(b)}", "",
          "| 아크 | A 출발일 | B 출발일 | A ΔV | B ΔV |", "|---|---|---|---|---|"]
    for r in kn["common_arcs"]:
        L.append(f"| {r['arc']} | {r['A_depart']:.0f} | {r['B_depart']:.0f} | {r['A_dv']:.2f} | {r['B_dv']:.2f} |")
    for r in kn["arcs_only_A"]:
        L.append(f"| {r['arc']} (A만) | {r['depart']:.0f} | — | {r['dv']:.2f} | — |")
    for r in kn["arcs_only_B"]:
        L.append(f"| {r['arc']} (B만) | — | {r['depart']:.0f} | — | {r['dv']:.2f} |")
    L += ["", "### 파편별 통계", "",
          f"프론트 해 {pdb['n_front']}개 기준. 최대 제거 수 {pdb['max_removal']} ({pdb['n_max_removal_points']}개 해).", "",
          "| 파편 | 포함 비율 | 첫 방문 횟수 | 재진입일 | 0일 점수 (순위) | 평균 점수 | 들어오는 아크 최소 ΔV (2년) | 일별 최소의 중앙값 | <150 m/s 로 들어올 수 있는 날 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d, f in pdb["debris_facts"].items():
        re = "—" if f["reentry_day"] is None else "%.0f" % f["reentry_day"]
        L.append(f"| {d} | {100 * pdb['inclusion'][d]:.0f} % | {pdb['first_visit'].get(d, 0)} | "
                 f"{re} | "
                 f"{f['score_day0']:.1f} ({f['score_rank_day0']}) | {f['score_mean']:.1f} | "
                 f"{f['min_dv_in_mps']:.0f} | {f['median_min_dv_in_mps']:.0f} | {f['days_in_below_150']} |")
    L += ["", f"최대 제거 해에서 늘 빠지는 파편: {', '.join(pdb['always_out_of_max'])}", "",
          "### 같은 제거 수 안의 ToF 장/단 대표쌍", "",
          f"짧은 ToF = < {SHORT_TOF_MAX:.0f}일. 짧은 ToF 해마다 ΔV 가 바로 아래인 긴 ToF 해(같은 순서가 있으면 그것)를 짝지음. "
          "긴 해는 대기로 ΔV 를 아끼는 대신 점수(시간에 따라 감소)를 잃는다.", "",
          "| 제거 수 | 긴 ToF 해 (ΔV, 점수, ToF, 순서) | 짧은 ToF 해 (ΔV, 점수, ToF, 순서) | 대기로 아낀 ΔV | 잃은 점수 | 더 걸린 일수 | 교환비 [m/s / 점] | 같은 제거 수 전체 짝의 교환비 중앙값 [범위] (짝 수) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in tp:
        q = r["rep"]
        l, s = q["long"], q["short"]
        def d(p):
            return f"{p['id']} ({p['total_dv_mps']:.1f}, {p['total_score']:.2f}, {tof(p):.0f}, {'-'.join(x[1:] for x in p['order'])})"
        ratio = "—" if q["mps_per_point"] is None else f"{q['mps_per_point']:.2f}"
        med = "—" if r["mps_per_point_median"] is None else (
            f"{r['mps_per_point_median']:.2f} [{r['mps_per_point_range'][0]:.2f}, {r['mps_per_point_range'][1]:.2f}]")
        L.append(f"| {r['n_removed']} | {d(l)} | {d(s)}{' (같은 순서)' if q['same_order'] else ''} | "
                 f"{q['dv_saved']:.1f} | {q['score_lost']:.2f} | {q['days_longer']:.0f} | {ratio} | {med} ({r['n_pairs']}) |")
    return "\n".join(L) + "\n"


def _slim(o):
    if isinstance(o, dict):
        if "visits" in o and "order" in o:
            return {k: o[k] for k in ("id", "total_dv_mps", "total_score", "order", "launch_day")} | {"tof": tof(o)}
        return {str(k): _slim(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_slim(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out or cfg["output"]["exact_dir"]
    data = load_dataset(cfg)
    st, pts = load_points(out)
    kn = knee(pts)
    pdb = per_debris(pts, data, cfg.get("launch_window"))
    tp = tof_pairs(pts)
    with open(os.path.join(out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(_slim({"knee": kn, "per_debris": pdb, "tof_pairs": tp}), f, indent=1, ensure_ascii=False)
    md = to_markdown(kn, pdb, tp)
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"-> {out}/analysis.md")


if __name__ == "__main__":
    main()

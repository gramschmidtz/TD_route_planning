"""
Deliverables from <out>/checkpoint.json (adr_data.pareto_exact):

  solutions/S###.json        one file per solution (dV ascending, S001 = lowest dV)
  pareto_summary.csv         one row per solution
  pareto_solutions.txt       human-readable itinerary per solution
  order_groups.csv           solutions sharing the same visiting order
  pareto_front.png (+ pareto_front_zoom_k.png)   labelled static front
  pareto_front.html          interactive plotly front (hover = full itinerary)
  timelines_representative.png   lowest dV / knee / highest score

    python -m adr_data.pareto_outputs [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os

import numpy as np

EPOCH = _dt.date(2026, 6, 5)


def _checkpoints(out):
    """main checkpoint + A8 range checkpoints (<out>/ranges/*/checkpoint.json)."""
    paths = [os.path.join(out, "checkpoint.json")]
    rdir = os.path.join(out, "ranges")
    if os.path.isdir(rdir):
        paths += [os.path.join(rdir, d, "checkpoint.json") for d in sorted(os.listdir(rdir))]
    return [p for p in paths if os.path.exists(p)]


def load_points(out):
    """Merge all checkpoints, drop duplicates, flag dominated points (A7: kept,
    pareto = False) and collect unproven score intervals."""
    sts = []
    for p in _checkpoints(out):
        with open(p, encoding="utf-8") as f:
            sts.append(json.load(f))
    st = dict(sts[0])
    st["done"] = all(s["done"] for s in sts)
    st["t_total"] = sum(s.get("t_total", 0.0) for s in sts)
    st["n_calls"] = sum(len(s.get("calls", [])) for s in sts)
    st["n_checkpoints"] = len(sts)
    key = lambda q: (q["total_score_scaled"] if q["total_score_scaled"] is not None
                     else round(q["total_score"], 6), round(q["total_dv_mps"], 6))
    uniq = {}
    for s in sts:
        for q in s["points"]:
            k = key(q)
            if k not in uniq or (q["optimal"] and not uniq[k]["optimal"]):
                uniq[k] = q
    pts = sorted(uniq.values(), key=lambda q: (q["total_dv_mps"], -q["total_score"]))
    for q in pts:
        sq, dq = key(q)
        q["pareto"] = not any((key(r)[0] >= sq and key(r)[1] <= dq) and key(r) != (sq, dq)
                              for r in pts)
    for k, q in enumerate(pts, 1):            # dV-ascending ids (stable for score_exact)
        q["id"] = f"S{k:03d}"
    st["unproven_intervals"] = [(q["id"], q["unproven_interval"]) for q in pts
                                if q.get("unproven_interval")]
    return st, pts


def short(order):
    return "-".join(str(int(o[1:])) for o in order)


def date(day):
    return (EPOCH + _dt.timedelta(days=float(day))).isoformat()


# ------------------------------------------------------------------ text
def write_tables(out, st, pts):
    sol = os.path.join(out, "solutions")
    os.makedirs(sol, exist_ok=True)
    keep = {p["id"] + ".json" for p in pts}
    for fn in os.listdir(sol):                # drop stale numbering (dv_exact resume)
        if fn.endswith(".json") and fn not in keep:
            os.remove(os.path.join(sol, fn))
    for p in pts:
        with open(os.path.join(sol, p["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(p, f, indent=1, ensure_ascii=False)

    with open(os.path.join(out, "pareto_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "total_dv_mps", "total_score", "total_score_scaled", "n_removed",
                    "order", "launch_day", "mission_end_day", "optimal", "mip_gap",
                    "solve_time_s", "epsilon", "pareto", "unproven_interval",
                    "mission_tof_days"])
        for p in pts:
            w.writerow([p["id"], f"{p['total_dv_mps']:.4f}", f"{p['total_score']:.4f}",
                        p["total_score_scaled"], len(p["order"]), ">".join(p["order"]),
                        f"{p['launch_day']:g}", f"{p['mission_end_day']:g}", p["optimal"],
                        "" if p["mip_gap"] is None else f"{p['mip_gap']:.3g}",
                        f"{p['solve_time_s']:.1f}", p["epsilon"], p.get("pareto", True),
                        "" if not p.get("unproven_interval") else
                        "[{}, {}]".format(*p["unproven_interval"]),
                        f"{p['mission_end_day'] - p['launch_day']:g}"])

    with open(os.path.join(out, "pareto_solutions.txt"), "w", encoding="utf-8") as f:
        f.write(f"# FY-1C 2026-06-05 Pareto solutions  method={st['method']}  K={st.get('K')}"
                f"  points={len(pts)}  complete={st['done']}\n\n")
        for p in pts:
            tag = "optimal" if p["optimal"] else f"NOT proven optimal, gap {p['mip_gap']}"
            if not p.get("pareto", True):
                tag += "; DOMINATED by a later point (pareto=false)"
            if p.get("unproven_interval"):
                tag += f"; unproven score interval {p['unproven_interval']}"
            f.write(f"{p['id']}  ΔV {p['total_dv_mps']:.1f} m/s  score {p['total_score']:.2f}"
                    f"  ({tag})\n")
            v0 = p["visits"][0]
            f.write(f"  launch {p['launch_date']} (day {p['launch_day']:g}) → {v0['debris']}"
                    f"  score {v0['score']:.2f}\n")
            for v in p["visits"][1:]:
                th = f", θ {v['theta_deg']:.2f}°" if "theta_deg" in v else ""
                pc = (f", plane change at {v['plane_change_at']} ({v['dv1_mps']:.1f} + "
                      f"{v['dv2_mps']:.1f})") if "plane_change_at" in v else ""
                f.write(f"  day {v['arc_depart_day']:g} → {v['debris']} (ΔV {v['arc_dv_mps']:.1f}"
                        f"{th}{pc}, arrive day {v['arrive_day']:g} {v['arrive_date']},"
                        f" score {v['score']:.2f})\n")
            ws = [f"{v['debris']} {v['wait_days']:g} d" for v in p["visits"]
                  if v.get("wait_days") is not None]
            f.write(f"  waits: {', '.join(ws) if ws else '-'};  stays at "
                    f"{p['visits'][-1]['debris']} until day "
                    f"{p['visits'][-1].get('stays_until_day', 0):g}\n\n")

    groups = {}
    for p in pts:
        groups.setdefault(">".join(p["order"]), []).append(p)
    rows = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    with open(os.path.join(out, "order_groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order", "n_solutions", "ids", "dv_min", "dv_max", "score_min", "score_max",
                    "launch_days"])
        for o, L in rows:
            w.writerow([o, len(L), " ".join(q["id"] for q in L),
                        f"{min(q['total_dv_mps'] for q in L):.2f}",
                        f"{max(q['total_dv_mps'] for q in L):.2f}",
                        f"{min(q['total_score'] for q in L):.3f}",
                        f"{max(q['total_score'] for q in L):.3f}",
                        " ".join(sorted({f"{q['launch_day']:g}" for q in L}, key=float))])
    return rows


# ------------------------------------------------------------------ figures
def knee_index(pts):
    x = np.array([p["total_dv_mps"] for p in pts])
    y = np.array([p["total_score"] for p in pts])
    if len(pts) < 3:
        return len(pts) - 1
    xn = (x - x.min()) / (np.ptp(x) or 1)
    yn = (y - y.min()) / (np.ptp(y) or 1)
    a = np.array([xn[0], yn[0]]); b = np.array([xn[-1], yn[-1]])
    d = np.abs(np.cross(b - a, np.stack([xn, yn], 1) - a)) / np.linalg.norm(b - a)
    return int(np.argmax(d))


def _scatter_labels(ax, pts, fontsize):
    from adjustText import adjust_text
    texts = []
    for k, p in enumerate(pts):
        x, y = p["total_dv_mps"], p["total_score"]
        texts.append(ax.text(x, y + (1 if k % 2 == 0 else -1) * 0.0, f"{p['id']}: {short(p['order'])}",
                             fontsize=fontsize))
    adjust_text(texts, ax=ax, expand=(1.15, 1.4),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5))


def _plot(ax, pts, labels=True, fontsize=6):
    dom = [p for p in pts if not p.get("pareto", True)]
    pts_nd = [p for p in pts if p.get("pareto", True)]
    opt = [p for p in pts_nd if p["optimal"]]
    non = [p for p in pts_nd if not p["optimal"]]
    if dom:
        ax.scatter([p["total_dv_mps"] for p in dom], [p["total_score"] for p in dom], s=26,
                   marker="x", color="0.55", zorder=3, label=f"dominated, pareto=false ({len(dom)})")
    ax.plot([p["total_dv_mps"] for p in pts_nd], [p["total_score"] for p in pts_nd], "-",
            color="0.8", lw=0.8, zorder=1, drawstyle="steps-post")
    ax.scatter([p["total_dv_mps"] for p in opt], [p["total_score"] for p in opt], s=16,
               color="#1f5fa8", zorder=3, label=f"proven optimal ({len(opt)})")
    if non:
        ax.scatter([p["total_dv_mps"] for p in non], [p["total_score"] for p in non], s=22,
                   facecolors="white", edgecolors="#c0392b", zorder=3,
                   label=f"time limit, not proven ({len(non)})")
    ax.set_xlabel("total ΔV [m/s]")
    ax.set_ylabel("total removal score")
    ax.grid(alpha=0.3)
    if labels:
        _scatter_labels(ax, pts, fontsize)


def plot_front(out, st, pts, per_zoom=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(pts)
    fig, ax = plt.subplots(figsize=(16, 10))
    _plot(ax, pts, labels=n <= 60, fontsize=6 if n <= 40 else 5)
    ax.set_title(f"FY-1C 2026-06-05 Pareto front — {st['method']}, K={st.get('K')}, "
                 f"{n} solutions{'' if st['done'] else ' (incomplete)'}"
                 + ("" if n <= 60 else "  — labels in zoom figures"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto_front.png"), dpi=200)
    plt.close(fig)
    for fn in os.listdir(out):
        if fn.startswith("pareto_front_zoom_"):
            os.remove(os.path.join(out, fn))
    zooms = []
    if n > 30:
        for z, k0 in enumerate(range(0, n, per_zoom), 1):
            sub = pts[k0:k0 + per_zoom]
            fig, ax = plt.subplots(figsize=(14, 9))
            _plot(ax, sub, labels=True, fontsize=7)
            x0, x1 = sub[0]["total_dv_mps"], sub[-1]["total_dv_mps"]
            ax.set_title(f"zoom {z}: {sub[0]['id']}–{sub[-1]['id']}  "
                         f"ΔV {x0:.0f}–{x1:.0f} m/s")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fn = f"pareto_front_zoom_{z}.png"
            fig.savefig(os.path.join(out, fn), dpi=200)
            plt.close(fig)
            zooms.append(fn)
    return zooms


def plot_front_tof(out, st, pts):
    """Front coloured by mission ToF = last arrival - launch [days]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = [p for p in pts if p.get("pareto", True)]
    x = [p["total_dv_mps"] for p in nd]
    y = [p["total_score"] for p in nd]
    t = [p["mission_end_day"] - p["launch_day"] for p in nd]
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.plot(x, y, "-", color="0.85", lw=0.8, zorder=1, drawstyle="steps-post")
    sc = ax.scatter(x, y, c=t, cmap="plasma", vmin=0, vmax=max(t) if t else 1, s=22,
                    edgecolors=["0.3" if p["optimal"] else "red" for p in nd],
                    linewidths=[0.3 if p["optimal"] else 1.2 for p in nd], zorder=3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("mission ToF = last arrival − launch [days]")
    ax.set_xlabel("total ΔV [m/s]")
    ax.set_ylabel("total removal score")
    ax.grid(alpha=0.3)
    ax.set_title(f"Pareto front coloured by mission ToF — {len(nd)} non-dominated solutions"
                 + ("" if st["done"] else " (incomplete)") + "; red edge = not proven optimal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto_front_tof.png"), dpi=200)
    plt.close(fig)


def plot_html(out, st, pts):
    import plotly.graph_objects as go
    groups = {}
    for p in pts:
        groups.setdefault(short(p["order"]), []).append(p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["total_dv_mps"] for p in pts],
                             y=[p["total_score"] for p in pts], mode="lines",
                             line=dict(color="lightgray", shape="hv"), name="front",
                             hoverinfo="skip"))

    def hover(p):
        lines = [f"<b>{p['id']}</b>  ΔV {p['total_dv_mps']:.2f} m/s  score "
                 f"{p['total_score']:.3f}" + ("" if p["optimal"] else
                                               f"  <i>not proven (gap {p['mip_gap']})</i>")
                 + ("" if p.get("pareto", True) else "  <i>dominated (pareto=false)</i>"),
                 f"order {' > '.join(p['order'])}",
                 f"mission ToF {p['mission_end_day'] - p['launch_day']:g} d (launch → last arrival)",
                 f"launch {p['launch_date']} (day {p['launch_day']:g})"]
        for v in p["visits"]:
            arc = (f" ← {v['arc_from']} ΔV {v['arc_dv_mps']:.1f}"
                   + (f" θ {v['theta_deg']:.2f}°" if "theta_deg" in v else "")) \
                if v["seq"] > 1 else " (launch)"
            dep = (f"dep d{v['depart_day']:g}, wait {v['wait_days']:g} d"
                   if v.get("depart_day") is not None else "stays to end")
            lines.append(f"{v['seq']}. {v['debris']}{arc} | arr d{v['arrive_day']:g} "
                         f"{v['arrive_date']} | {dep} | score {v['score']:.2f}")
        return "<br>".join(lines)

    for o, L in sorted(groups.items(), key=lambda kv: kv[1][0]["total_dv_mps"]):
        fig.add_trace(go.Scatter(
            x=[p["total_dv_mps"] for p in L], y=[p["total_score"] for p in L],
            mode="markers", name=f"{o} ({len(L)})",
            marker=dict(size=9, symbol=["x" if not p.get("pareto", True) else
                                   "circle" if p["optimal"] else "circle-open" for p in L],
                        line=dict(width=1)),
            text=[hover(p) for p in L], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=f"FY-1C 2026-06-05 Pareto front ({st['method']}, K={st.get('K')}, "
                            f"{len(pts)} solutions; open marker = not proven optimal, x = dominated)",
                      xaxis_title="total ΔV [m/s]", yaxis_title="total removal score",
                      legend_title="visit order (click to toggle)", template="plotly_white",
                      hoverlabel=dict(font_size=11), height=800)
    fig.write_html(os.path.join(out, "pareto_front.html"), include_plotlyjs=True, full_html=True)


def plot_timelines(out, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    if not pts:
        return []
    kn = knee_index(pts)
    reps = [("lowest ΔV", pts[0]), ("knee", pts[kn]), ("highest score", pts[-1])]
    ids = sorted({v["debris"] for _, p in reps for v in p["visits"]})
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    end_day = max(v.get("stays_until_day") or v["arrive_day"] for _, p in reps
                  for v in p["visits"])
    for ax, (lab, p) in zip(axes, reps):
        row = {d: k for k, d in enumerate(ids)}
        for v in p["visits"]:
            a = v["arrive_day"]
            b = v["depart_day"] if v.get("depart_day") is not None else v.get("stays_until_day", a)
            ax.barh(row[v["debris"]], max(b - a, 0.6), left=mdates.date2num(
                _dt.datetime.combine(EPOCH, _dt.time()) + _dt.timedelta(days=a)),
                height=0.5, color="#1f5fa8" if v.get("depart_day") is not None else "#9bb7d8")
            ax.text(mdates.date2num(_dt.datetime.combine(EPOCH, _dt.time()) +
                                    _dt.timedelta(days=a)), row[v["debris"]] + 0.32,
                    f"{v['score']:.1f}", fontsize=7)
        for v0, v1 in zip(p["visits"], p["visits"][1:]):
            x0 = mdates.date2num(_dt.datetime.combine(EPOCH, _dt.time()) +
                                 _dt.timedelta(days=v1["arc_depart_day"]))
            x1 = mdates.date2num(_dt.datetime.combine(EPOCH, _dt.time()) +
                                 _dt.timedelta(days=v1["arrive_day"]))
            ax.annotate("", xy=(x1, row[v1["debris"]]), xytext=(x0, row[v0["debris"]]),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
            ax.text(x0 + 4, (row[v0["debris"]] + row[v1["debris"]]) / 2,
                    f"{v1['arc_dv_mps']:.0f} m/s", fontsize=7, color="#c0392b")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{lab}: {p['id']}  ΔV {p['total_dv_mps']:.1f} m/s, score "
                     f"{p['total_score']:.2f}, order {short(p['order'])}"
                     + ("" if p["optimal"] else "  (not proven optimal)"), fontsize=10)
        ax.set_xlim(mdates.date2num(_dt.datetime.combine(EPOCH, _dt.time())) - 5,
                    mdates.date2num(_dt.datetime.combine(EPOCH, _dt.time()) +
                                    _dt.timedelta(days=end_day)) + 5)
    loc = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axes[-1].set_xlabel("date (dark bar = wait at debris, light bar = last debris until mission end,"
                        " arrow = transfer with ΔV, number = score)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timelines_representative.png"), dpi=200)
    plt.close(fig)
    return [(lab, p["id"]) for lab, p in reps]


def write_all(out, cfg=None):
    st, pts = load_points(out)
    groups = write_tables(out, st, pts)
    zooms = plot_front(out, st, pts)
    plot_front_tof(out, st, pts)
    plot_html(out, st, pts)
    reps = plot_timelines(out, pts)
    info = {"n_points": len(pts), "done": st["done"], "zooms": zooms, "representatives": reps,
            "n_groups": len(groups)}
    print(f"outputs -> {out}: {info}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pareto_fy1c_20260605_K10")
    write_all(ap.parse_args().out)

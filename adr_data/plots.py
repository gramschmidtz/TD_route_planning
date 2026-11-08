"""
Figures from the saved sweep (results/pareto_adr_fy1c_20260605.json):
  * Pareto front  total dV vs total score (all epsilon rows + non-dominated set)
  * timelines of 3 representative solutions: low-dV / knee / max-score
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .load import load_dataset  # noqa: E402

C_LINE = "#2a6fdb"
C_PICK = ["#2a9d8f", "#e76f51", "#6a4c93"]


def representatives(rows):
    """low-dV = cheapest Pareto point with >= 2 visits; high = max score;
    knee = Pareto point farthest from the chord between the two extremes
    (both axes normalised to [0, 1])."""
    P = sorted([r for r in rows if r["pareto"]], key=lambda r: r["total_dv"])
    multi = [r for r in P if r["n_visits"] >= 2] or P
    low = multi[0]
    high = max(P, key=lambda r: (r["total_score"], -r["total_dv"]))
    x = np.array([r["total_dv"] for r in P])
    y = np.array([r["total_score"] for r in P])
    xn = (x - x.min()) / max(np.ptp(x), 1e-9)
    yn = (y - y.min()) / max(np.ptp(y), 1e-9)
    a, b = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = np.abs(np.cross(b - a, np.stack([xn, yn], 1) - a)) / max(np.linalg.norm(b - a), 1e-9)
    knee = P[int(np.argmax(d))]
    return [("low dV", low), ("knee", knee), ("max score", high)]


def plot_front(rows, reps, path, dvmax):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter([r["total_dv"] for r in rows], [r["total_score"] for r in rows],
               s=14, color="0.75", zorder=2, label="ε-sweep solutions")
    P = sorted([r for r in rows if r["pareto"]], key=lambda r: r["total_dv"])
    xs = [r["total_dv"] for r in P]
    ys = [r["total_score"] for r in P]
    ax.step(xs + [dvmax], ys + [ys[-1]], where="post", color=C_LINE, lw=1.6, zorder=3)
    ax.scatter(xs, ys, s=26, color=C_LINE, zorder=4, label="non-dominated")
    for r in P:
        ax.annotate(str(r["n_visits"]), (r["total_dv"], r["total_score"]),
                    xytext=(3, -11), textcoords="offset points", fontsize=7, color="0.35")
    for (lab, r), c in zip(reps, C_PICK):
        ax.scatter([r["total_dv"]], [r["total_score"]], s=110, facecolor="none",
                   edgecolor=c, lw=2, zorder=5)
        ax.annotate(f"{lab}\n{'-'.join(v['id'] for v in r['visits'])}",
                    (r["total_dv"], r["total_score"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, color=c)
    ax.axvline(dvmax, color="0.4", ls=":", lw=1)
    ax.text(dvmax, ax.get_ylim()[0], " ΔV_max", rotation=90, va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("total ΔV  [m/s]   (sum of transfer arcs, combined 2-impulse)")
    ax.set_ylabel("total removal score  Σ score_j(t_arr)")
    ax.set_title("ε-constraint Pareto front — FY-1C ADR, epoch 2026-06-05 "
                 "(numbers = debris removed)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_timelines(rows, reps, data, path, horizon_days):
    n = data.n
    fig, axes = plt.subplots(len(reps), 1, figsize=(11, 3.1 * len(reps)), sharex=True)
    days = np.arange(horizon_days)
    for ax, (lab, r), c in zip(np.atleast_1d(axes), reps, C_PICK):
        for j in range(n):                        # faint score_j(t) inside each row
            ax.plot(days, j + 0.42 * data.score[j, :horizon_days] / 100 - 0.21,
                    color="0.82", lw=0.8, zorder=1)
        ax.axvspan(0, 180, color="0.95", zorder=0)
        prev = None
        for v in r["visits"]:
            y = data.ids.index(v["id"])
            ax.scatter([v["arrival_day"]], [y], s=40, color=c, zorder=4)
            end = v["departure_day"] if v["departure_day"] is not None else None
            if end is not None and end > v["arrival_day"]:
                ax.plot([v["arrival_day"], end], [y, y], color=c, lw=3, alpha=0.35, zorder=2)
            if prev is not None:
                x0 = v["depart_prev_day"]
                ax.annotate("", xy=(v["arrival_day"], y), xytext=(x0, prev),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1), zorder=3)
                ax.text(x0 + 6, (y + prev) / 2, f"{v['dv']:.0f}", fontsize=7, color=c,
                        va="center")
            ax.text(v["arrival_day"], y + 0.28, f"{v['score']:.1f}", fontsize=7,
                    ha="center", color="0.25")
            prev = y
        ax.set_yticks(range(n))
        ax.set_yticklabels(data.ids, fontsize=8)
        ax.set_ylim(n - 0.5, -0.6)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{lab}: ΣΔV = {r['total_dv']:.0f} m/s, Σscore = {r['total_score']:.1f}, "
                     f"{r['n_visits']} removed, launch day {r['launch_day']:g}   "
                     f"(arrow label = arc ΔV m/s, number = score at removal)",
                     fontsize=9, color=c, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("day since 2026-06-05  (shaded: launch window 0–180 d; "
                                       "bar = waiting at debris; grey = score_j(t))")
    np.atleast_1d(axes)[-1].set_xlim(-10, horizon_days)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_all(cfg):
    from .adapter import dv_max_mps
    out = cfg["output"]
    with open(out["json"], "r", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    reps = representatives(rows)
    plot_front(rows, reps, out["fig_front"], dv_max_mps(cfg))
    plot_timelines(rows, reps, load_dataset(cfg), out["fig_timeline"],
                   int(cfg["mission"]["horizon_days"]))

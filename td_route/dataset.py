"""
Fixed 120-network benchmark: 12 sizes (|V| = 10..200) x 10 instances, numbered
001..120 in size order, each fully reproducible from its seed.

Each network is serialised to a self-contained JSON (node positions + every
arc's length, speed category, bp, beta and preference schedule). Because the
time-dependent travel time c_ijt and preference u_ijt are deterministic
functions of those stored attributes plus the speed matrix / horizons (also
saved), reloading a JSON reproduces the exact TD data -- no need to store the
per-minute curves. A PNG of each network structure and a manifest.csv are
written alongside.
"""
from __future__ import annotations

import csv
import json
import os

from .network import TDNetwork
from .instance import generate_random_network


# -------------------------------------------------------------------------- #
# Serialisation
# -------------------------------------------------------------------------- #
def serialize_network(net: TDNetwork, meta: dict | None = None) -> dict:
    grid = net.region_grid or {}
    return {
        "meta": {
            "n_nodes": net.n_nodes,
            "n_arcs": net.n_arcs,
            "pref_density": net.pref_density(),
            "free_flow_speed": net.free_flow_speed,
            "t_start": net.t_start,
            "t_end": net.t_end,
            "horizons": [list(h) for h in net.horizons],
            "speed_categories": net.speed_categories,
            "n_region_cols": grid.get("n_cols"),
            "n_region_rows": grid.get("n_rows"),
            **(meta or {}),
        },
        "nodes": [{"id": v, "x": d["x"], "y": d["y"]}
                  for v, d in net.G.nodes(data=True)],
        "arcs": [{"i": i, "j": j,
                  "length": d["length"], "speed_cat": d["speed_cat"],
                  "bp": d["bp"], "beta": d.get("beta", 0.0),
                  "pref_schedule": [list(s) for s in d.get("pref_schedule", [])]}
                 for i, j, d in net.G.edges(data=True)],
    }


def save_network_json(net: TDNetwork, path: str, meta: dict | None = None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_network(net, meta), f)


def load_network_json(path: str) -> TDNetwork:
    """Rebuild a TDNetwork from its JSON (fully reconstructs the TD data)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data["meta"]
    net = TDNetwork(free_flow_speed=m["free_flow_speed"],
                    t_start=m["t_start"], t_end=m["t_end"],
                    horizons=[tuple(h) for h in m["horizons"]],
                    speed_categories=m["speed_categories"])
    for nd in data["nodes"]:
        net.G.add_node(nd["id"], x=nd["x"], y=nd["y"])
    for a in data["arcs"]:
        attrs = {"length": a["length"], "speed_cat": a["speed_cat"],
                 "bp": a["bp"], "beta": a.get("beta", 0.0)}
        if a.get("pref_schedule"):
            attrs["pref_schedule"] = [tuple(s) for s in a["pref_schedule"]]
        net.G.add_edge(a["i"], a["j"], **attrs)
    net.partition_regions(m["n_region_cols"], m["n_region_rows"])
    # expose the stored origin/destination (farthest pair) on the graph
    if "origin" in m:
        net.G.graph["origin"] = m["origin"]
    if "destination" in m:
        net.G.graph["destination"] = m["destination"]
    return net


# -------------------------------------------------------------------------- #
# PNG of the network structure
# -------------------------------------------------------------------------- #
def network_png(net: TDNetwork, path: str, title: str = "", dpi: int = 110,
                O=None, D=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = {v: (d["x"], d["y"]) for v, d in net.G.nodes(data=True)}
    for i, j, d in net.G.edges(data=True):
        x = [pos[i][0], pos[j][0]]; y = [pos[i][1], pos[j][1]]
        if d["bp"] > 0:
            ax.plot(x, y, "-", lw=0.4 + d["bp"] / 5,
                    color=plt.cm.autumn(1 - d["bp"] / 10), alpha=0.8, zorder=2)
        else:
            ax.plot(x, y, "-", lw=0.35, color="0.8", alpha=0.6, zorder=1)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=10, color="0.3", zorder=3)
    # label each node with its id next to the dot
    fs = max(4, min(8, int(360 / max(1, net.n_nodes) ** 0.5)))
    for v, (px, py) in pos.items():
        ax.annotate(str(v), (px, py), textcoords="offset points",
                    xytext=(3, 3), fontsize=fs, color="0.15", zorder=7)

    g = net.region_grid
    if g:
        for c in range(g["n_cols"] + 1):
            ax.axvline(g["xmin"] + c * g["cw"], color="0.7", lw=0.5, ls="--")
        for r in range(g["n_rows"] + 1):
            ax.axhline(g["ymin"] + r * g["ch"], color="0.7", lw=0.5, ls="--")

    if O is not None:
        ax.scatter(*pos[O], s=150, marker="*", color="green", zorder=6,
                   edgecolor="k", linewidths=0.5, label=f"O={O}")
    if D is not None:
        ax.scatter(*pos[D], s=150, marker="*", color="red", zorder=6,
                   edgecolor="k", linewidths=0.5, label=f"D={D}")
    if O is not None or D is not None:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_title(title or f"|V|={net.n_nodes}  |A|={net.n_arcs}  "
                 f"ρ={net.pref_density():.0%}")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# -------------------------------------------------------------------------- #
# Build the whole benchmark
# -------------------------------------------------------------------------- #
def build_dataset(cfg: dict, verbose: bool = True) -> list:
    """Generate and save the 120-network benchmark described by cfg['dataset'].
    Returns the manifest as a list of dict rows."""
    from .config import network_kwargs
    from .instance import pick_od
    ds = cfg.get("dataset", {})
    out_dir = ds.get("out_dir", "dataset")
    sizes = ds.get("sizes", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200])
    per = ds.get("instances_per_size", 10)
    base_seed = ds.get("base_seed", 0)
    save_png = ds.get("save_png", True)
    dpi = ds.get("png_dpi", 110)

    net_dir = os.path.join(out_dir, "networks")
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(net_dir, exist_ok=True)
    if save_png:
        os.makedirs(png_dir, exist_ok=True)

    gen_kwargs = network_kwargs(cfg)
    manifest = []
    idx = 0
    for size in sizes:
        for _ in range(per):
            idx += 1
            num = f"{idx:03d}"
            seed = base_seed + idx
            net = generate_random_network(size, seed=seed, **gen_kwargs)
            # origin/destination = the two farthest-apart nodes (deterministic)
            O, D = pick_od(net, strategy="farthest")

            json_path = os.path.join(net_dir, f"net_{num}.json")
            save_network_json(net, json_path,
                              meta={"number": num, "size": size, "seed": seed,
                                    "origin": O, "destination": D})
            png_rel = ""
            if save_png:
                png_path = os.path.join(png_dir, f"net_{num}.png")
                network_png(net, png_path,
                            title=f"net {num}   |V|={size}   |A|={net.n_arcs}   "
                                  f"ρ={net.pref_density():.0%}   seed={seed}",
                            dpi=dpi, O=O, D=D)
                png_rel = os.path.relpath(png_path, out_dir)

            row = {"number": num, "size": size, "seed": seed,
                   "n_arcs": net.n_arcs, "pref_density": round(net.pref_density(), 3),
                   "origin": O, "destination": D,
                   "json": os.path.relpath(json_path, out_dir), "png": png_rel}
            manifest.append(row)
            if verbose:
                print(f"  net_{num}  |V|={size:<3} |A|={net.n_arcs:<4} "
                      f"ρ={net.pref_density():.0%}  O={O} D={D}  seed={seed}")

    man_path = os.path.join(out_dir, "manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    if verbose:
        print(f"\n{len(manifest)} networks -> {out_dir}/  (manifest.csv written)")
    return manifest

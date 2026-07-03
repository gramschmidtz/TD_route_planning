"""
YAML configuration loader and object builders.

`config.yaml` at the project root is the single source of truth for every
tunable value. The scripts in examples/ read it and pass the values into the
library, so behaviour is changed by editing the YAML -- not by CLI numbers.

The library modules keep their own sensible defaults, so they remain usable
without a config file; this module just wires YAML values into them.
"""
from __future__ import annotations

import os

import yaml

from .instance import generate_random_network, pick_od
from .nsga2 import NSGA2Config

DEFAULT_CONFIG = os.path.join(os.getcwd(), "config.yaml")


def load_config(path: str | None = None) -> dict:
    """Load the YAML config. Falls back to ./config.yaml."""
    path = path or DEFAULT_CONFIG
    if not os.path.isfile(path):
        raise SystemExit(f"config file not found: {path}\n"
                         f"Create one (see the project's config.yaml) or pass "
                         f"--config <path>.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------- #
# Builders
# -------------------------------------------------------------------------- #
def network_kwargs(cfg: dict) -> dict:
    """All generate_random_network() keyword args EXCEPT n_nodes and seed,
    taken from the `network` + `instance` sections. Shared by build_network
    and the dataset builder (which vary n_nodes / seed themselves)."""
    net_c = cfg.get("network", {})
    ins_c = cfg.get("instance", {})
    horizons = [tuple(h) for h in net_c["horizons"]] if "horizons" in net_c \
        else None
    return dict(
        area=net_c.get("area", 100.0),
        free_flow_speed=net_c.get("free_flow_speed", 2.0),
        pref_density=ins_c.get("pref_density", 0.5),
        bp_range=tuple(ins_c.get("bp_range", (1, 10))),
        beta_range=tuple(ins_c.get("beta_range", (0.2, 0.8))),
        pref_state_dur=tuple(ins_c.get("pref_state_dur", (60, 240))),
        bidirectional=ins_c.get("bidirectional", True),
        n_region_cols=ins_c.get("n_region_cols", 4),
        n_region_rows=ins_c.get("n_region_rows", 3),
        t_start=net_c.get("t_start", 420),
        t_end=net_c.get("t_end", 1260),
        horizons=horizons,
        speed_categories=net_c.get("speed_categories"),
    )


def build_network(cfg: dict):
    """Create a TDNetwork from the `network` + `instance` sections."""
    ins_c = cfg.get("instance", {})
    return generate_random_network(ins_c["n_nodes"], seed=ins_c.get("seed"),
                                   **network_kwargs(cfg))


def build_od(cfg: dict, net):
    """Pick (O, D) using the `query` section."""
    q = cfg.get("query", {})
    return pick_od(net, seed=q.get("od_seed", 0),
                   strategy=q.get("od_strategy", "farthest"))


def build_t0(cfg: dict, net):
    q = cfg.get("query", {})
    t0 = q.get("t0")
    return net.t_start if t0 is None else int(t0)


def load_dataset_network(cfg: dict):
    """Load the fixed dataset network selected by `network_number`.
    Returns (net, number_str). O/D are on net.G.graph['origin'/'destination']."""
    from .dataset import load_network_json
    num = str(cfg.get("network_number", "001")).zfill(3)
    out_dir = cfg.get("dataset", {}).get("out_dir", "dataset")
    path = os.path.join(out_dir, "networks", f"net_{num}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"network file not found: {path}\n"
                         f"Run  python -m examples.build_dataset  first.")
    return load_network_json(path), num


def build_nsga2_config(cfg: dict) -> NSGA2Config:
    n = cfg.get("nsga2", {})
    return NSGA2Config(
        pop_size=n.get("pop_size", 50),
        crossover_prob=n.get("crossover_prob", 0.85),
        mutation_prob=n.get("mutation_prob", 0.15),
        gmax=n.get("gmax", 5),
        max_decode=n.get("max_decode", 6),
        seed=n.get("seed"),
    )

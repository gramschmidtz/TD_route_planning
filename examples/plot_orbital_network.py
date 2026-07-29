"""
Visualise the orbital transfer network (space twin of the road network_png).

Produces up to three files (paths from config.yaml `orbital.viz`):
  * a static 3D picture of Earth + the 10 circular orbits            (.png)
  * the J2 RAAN-drift-over-time chart (windows = curve crossings)    (.png)
  * an interactive 3D HTML with a departure-time slider              (.html,
    needs Plotly:  pip install plotly)

    python -m examples.plot_orbital_network
    python -m examples.plot_orbital_network --config my.yaml

Run from the project root (the folder containing td_route/ and config.yaml).
"""
from __future__ import annotations

import argparse

from td_route.config import load_config
from td_route.orbital_viz import orbits_3d_png, raan_drift_png, orbits_3d_html
from examples.plot_transfer import build_from_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    net = build_from_cfg(cfg)
    viz = cfg.get("orbital", {}).get("viz", {})
    orbits_3d_png(net, viz.get("png_3d", "orbital_network_3d.png"))
    raan_drift_png(net, viz.get("png_drift", "orbital_raan_drift.png"))
    orbits_3d_html(net, viz.get("html_3d", "orbital_network_3d.html"))


if __name__ == "__main__":
    main()

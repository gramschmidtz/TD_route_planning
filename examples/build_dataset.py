"""
Build the fixed 120-network benchmark (12 sizes x 10 instances, numbered
001..120) into the folder given by `dataset.out_dir` in config.yaml.

All settings come from config.yaml (sections `network`, `instance`, `dataset`);
the script takes no numeric arguments.

    python -m examples.build_dataset
    python -m examples.build_dataset --config my.yaml

Outputs under <out_dir>/:
    networks/net_001.json ... net_120.json   (fully reloadable TD networks)
    png/net_001.png       ... net_120.png     (network-structure images)
    manifest.csv                              (number, size, seed, |A|, rho, paths)
"""
from __future__ import annotations

import argparse
import time

from td_route.config import load_config
from td_route.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="path to the YAML config (default ./config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    t = time.time()
    build_dataset(cfg, verbose=True)
    print(f"done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()

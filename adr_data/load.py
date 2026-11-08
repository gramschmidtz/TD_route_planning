"""
Loaders for the precomputed FY-1C ADR tables (external/adr_time_varying_benefit/adr/out_0605).

Only the *_0605 folders are read (epoch 2026-06-05 00:00 UTC = day 0).  No
network access, no SGP4: the tables are used as shipped.

    ADRData.score[j, d]      score_j(day d)                 (10, 1827)
    ADRData.dv[i, j, d]      dV for departing i -> j on day d [m/s]   (10, 10, 1828)
                             NaN = an end-point has re-entered (and on i == j)
    ADRData.ids              ["D01", ..., "D10"]
    ADRData.norad            [31837, ...]
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

DEFAULT_CONFIG = os.path.join("config", "adr_fy1c_20260605.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ADRData:
    ids: list
    norad: list
    score: np.ndarray                 # (n, n_days)
    dv: np.ndarray                    # (n, n, n_days_dv)  [m/s]
    dv_table: str
    meta: dict = field(default_factory=dict)
    root: str = ""

    @property
    def n(self) -> int:
        return len(self.ids)

    def reentry_day(self, j: int):
        """Model re-entry day (debris_dataset.json), None if it survives 5 yr."""
        r = self.meta.get("debris", [{}] * self.n)[j].get("reentry_days")
        return None if r is None else float(r)


def _paths(cfg: dict):
    ds = cfg["dataset"]
    root = ds["root"]
    return root, os.path.join(root, ds["out_dir"]), os.path.join(root, ds["data_dir"])


def load_score_csv(path: str):
    """score_daily.csv -> (ids, norad, score[j, day])."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    cols = head[2:]                                   # D01_31837 ...
    ids = [c.split("_")[0] for c in cols]
    norad = [int(c.split("_")[1]) for c in cols]
    body = rows[1:]
    days = [int(r[0]) for r in body]
    assert days == list(range(len(days))), "score_daily.csv days must be 0..N-1"
    score = np.array([[float(v) for v in r[2:]] for r in body]).T
    return ids, norad, score


def load_dv(out_dir: str, table: str = "combined") -> np.ndarray:
    return np.load(os.path.join(out_dir, f"edge_dv_{table}.npy"))


def load_mean_elements(out_dir: str):
    """debris_mean_elements_daily.csv -> dict id -> array (n_days, 5)
    columns a_km, e, i_deg, raan_deg, argp_deg (NaN after re-entry)."""
    with open(os.path.join(out_dir, "debris_mean_elements_daily.csv"),
              "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_days = max(int(r["day"]) for r in rows) + 1
    out: dict = {}
    for r in rows:
        arr = out.setdefault(r["id"], np.full((n_days, 5), np.nan))
        arr[int(r["day"])] = [float(r[k]) for k in
                              ("a_km", "e", "i_deg", "raan_deg", "argp_deg")]
    return out


def load_dataset(cfg: dict | None = None) -> ADRData:
    cfg = cfg or load_config()
    root, out_dir, _ = _paths(cfg)
    ids, norad, score = load_score_csv(
        os.path.join(out_dir, cfg["dataset"].get("score_file", "score_daily.csv")))
    table = cfg["dataset"].get("dv_table", "combined")
    dv = load_dv(out_dir, table)
    with open(os.path.join(out_dir, "debris_dataset.json"), "r", encoding="utf-8",
              errors="replace") as f:
        meta = json.load(f)
    if dv.shape[0] != len(ids) or dv.shape[1] != len(ids):
        raise ValueError(f"dV table shape {dv.shape} does not match {len(ids)} debris")
    meta_ids = [d["id"] for d in meta["debris"]]
    if meta_ids != ids:
        raise ValueError(f"debris order mismatch: csv {ids} vs json {meta_ids}")
    return ADRData(ids=ids, norad=norad, score=score, dv=dv, dv_table=table,
                   meta=meta, root=root)


def read_tles(cfg: dict | None = None):
    """All 16 TLE pairs of the dataset: 10 debris (debris_dataset.json) +
    6 protected satellites (data_0605/targets_tle.txt). -> [(name, l1, l2)]."""
    cfg = cfg or load_config()
    _, out_dir, data_dir = _paths(cfg)
    with open(os.path.join(out_dir, "debris_dataset.json"), "r", encoding="utf-8",
              errors="replace") as f:
        meta = json.load(f)
    tles = [(d["id"], d["tle"][0], d["tle"][1]) for d in meta["debris"]]
    with open(os.path.join(data_dir, "targets_tle.txt"), "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    for k in range(0, len(lines), 3):
        tles.append((lines[k].strip(), lines[k + 1], lines[k + 2]))
    return tles


def tle_checksum_ok(line: str) -> bool:
    """Standard TLE modulo-10 checksum: digits count, '-' counts 1."""
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return s % 10 == int(line[68])

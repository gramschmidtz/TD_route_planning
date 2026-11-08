"""
Prompt §7 checks for the FY-1C ADR dataset + adapter + epsilon-constraint results.

    python -m pytest tests/test_adr_data.py -q
Run from the repository root.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from adr_data.adapter import ADRNetwork, dv_max_mps
from adr_data.load import (load_config, load_dataset, load_dv, load_mean_elements,
                           read_tles, tle_checksum_ok)
from adr_data.maneuver import dv_table_from_elements, transfer_dv

CFG = load_config()
DATA = load_dataset(CFG)
OUT_DIR = os.path.join(CFG["dataset"]["root"], CFG["dataset"]["out_dir"])
IDX = {d: k for k, d in enumerate(DATA.ids)}

SCORE0 = dict(D01=100.00, D02=95.29, D03=89.85, D04=85.41, D05=79.69,
              D06=73.45, D07=57.58, D08=53.17, D09=52.78, D10=48.05)
SCORE150 = dict(D01=93.00, D02=81.02, D03=83.19, D04=77.41, D05=75.76,
                D06=64.59, D07=49.28, D08=47.74, D09=50.48, D10=45.31)
# (from, to, day, theta_deg, split, combined)
MANEUVER = [("D01", "D02", 0, 22.189, 2915.2, 2910.3),
            ("D03", "D08", 0, 2.996, 402.2, 398.7),
            ("D05", "D03", 129, 1.083, 179.5, 161.9),
            ("D03", "D10", 130, 0.028, 22.3, 19.3),
            ("D01", "D07", 307, 1.398, 213.5, 199.4),
            ("D08", "D10", 6, 0.130, 46.7, 37.4)]


# ---------------------------------------------------------------- 7.1 data
def test_tle_checksums_all_16():
    tles = read_tles(CFG)
    assert len(tles) == 16
    bad = [(n, k) for n, l1, l2 in tles for k, ln in ((1, l1), (2, l2))
           if not tle_checksum_ok(ln)]
    assert not bad, bad


@pytest.mark.parametrize("day,expected", [(0, SCORE0), (150, SCORE150)])
def test_score_values(day, expected):
    for d, v in expected.items():
        assert DATA.score[IDX[d], day] == pytest.approx(v, abs=0.05), d


def test_score_monotone_and_zero_after_reentry():
    S = DATA.score
    assert np.all(np.diff(S, axis=1) <= 1e-9)
    assert np.all(S >= 0)
    for j, deb in enumerate(DATA.meta["debris"]):
        r = deb.get("reentry_days")
        if r is not None:
            assert np.all(S[j, int(np.ceil(r)) + 1:] == 0), deb["id"]


# ------------------------------------------------------------ 7.2 maneuver
@pytest.fixture(scope="module")
def recomputed():
    elems = load_mean_elements(OUT_DIR)
    comb, th = dv_table_from_elements(elems, DATA.ids, "combined")
    split, _ = dv_table_from_elements(elems, DATA.ids, "split")
    return comb, split, th, elems


@pytest.mark.parametrize("a,b,day,theta,dv_split,dv_comb", MANEUVER)
def test_maneuver_reference_values(recomputed, a, b, day, theta, dv_split, dv_comb):
    comb, split, th, _ = recomputed
    i, j = IDX[a], IDX[b]
    npy_c = load_dv(OUT_DIR, "combined")
    npy_s = load_dv(OUT_DIR, "split")
    assert th[i, j, day] == pytest.approx(theta, abs=2e-3)
    assert npy_c[i, j, day] == pytest.approx(dv_comb, abs=0.5)
    assert npy_s[i, j, day] == pytest.approx(dv_split, abs=0.5)
    assert comb[i, j, day] == pytest.approx(dv_comb, abs=0.5)
    assert split[i, j, day] == pytest.approx(dv_split, abs=0.5)


def test_recomputed_tables_match_npy(recomputed):
    comb, split, _, _ = recomputed
    for M, name in ((comb, "combined"), (split, "split")):
        npy = load_dv(OUT_DIR, name)
        assert np.array_equal(np.isnan(M), np.isnan(npy))
        assert np.nanmax(np.abs(M - npy)) < 0.5


def test_combined_le_split_and_symmetric():
    c = load_dv(OUT_DIR, "combined")
    s = load_dv(OUT_DIR, "split")
    ok = ~np.isnan(c)
    assert np.all(c[ok] <= s[ok] + 1e-9)
    assert np.allclose(c, np.transpose(c, (1, 0, 2)), atol=1e-6, equal_nan=True)


def test_transfer_dv_scalar_matches():
    # same-orbit transfer costs nothing
    d, th = transfer_dv(7000.0, 98.0, 200.0, 7000.0, 98.0, 200.0)
    assert d == pytest.approx(0.0, abs=1e-6) and th == pytest.approx(0.0, abs=1e-4)


# ------------------------------------------------------------- adapter
def test_adapter_interface():
    net = ADRNetwork(DATA, CFG)
    assert dv_max_mps(CFG) == pytest.approx(2161.6, abs=0.1)
    S, T = net.S, net.T
    assert net.arc_feasible(S, 1, 0) and net.arc_feasible(S, 1, 180)
    assert not net.arc_feasible(S, 1, 181)
    K = CFG["pareto"].get("score_scale") or 1
    assert net.preference(S, IDX["D01"] + 1, 0) == pytest.approx(round(K * 100.0))
    assert net.arc_cost(S, 3, 10) == 0.0
    assert not net.arc_feasible(1, T, net.H - 1) and net.arc_feasible(1, T, net.H)
    # D03 -> D10 on day 130 (19.3 m/s): cost at departure, score at GRAPH arrival (P1)
    i, j = IDX["D03"] + 1, IDX["D10"] + 1
    tof = net.travel_time(i, j, 130)
    assert tof == 1                                   # A2: max(1, ceil((0.07 + 0)/1))
    assert net.arc_cost(i, j, 130) == pytest.approx(19.3, abs=0.5)
    assert net.preference(i, j, 130) == pytest.approx(round(K * DATA.score[j - 1, 130 + tof]))
    # arcs above dV_max are never created
    assert not net.arc_feasible(IDX["D01"] + 1, IDX["D02"] + 1, 0)   # 2910 m/s


# --------------------------------------------------------- 7.3 Pareto
def _solve(eps):
    # CBC (bundled with PuLP): small eps solve in seconds, needs no CPLEX, and
    # writes its scratch files to the temp dir (CPLEX_CMD writes cplex.log to
    # the cwd, which collides with a concurrently running sweep).
    from td_route.eecm.solve import configure_solver
    from adr_data.run_pareto import solve_eps
    configure_solver({"solver": "cbc"})
    return solve_eps(DATA, CFG, eps)


def test_eps_zero_single_visit_score_100():
    r = _solve(0.0)
    assert r["n_visits"] == 1
    assert r["total_dv"] == 0.0
    v = r["visits"][0]
    assert v["id"] == "D01"
    # optimum = max_j score_j(t_L) over the launch window, in objective units:
    # with K = score_scale, ties within 1/K (e.g. 99.98 and 100.00 -> 1000) are allowed
    K = CFG["pareto"].get("score_scale")
    best = DATA.score[:, :181].max()
    if K:
        assert round(K * r["total_score"]) == round(K * best)
    else:
        assert r["total_score"] == pytest.approx(best, abs=1e-9)


def test_small_eps_monotone():
    vals = [_solve(e)["total_score"] for e in (0.0, 100.0, 200.0)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_tof_formula_A2():
    import copy
    for dt, tau, k in ((1, 0, 1), (5, 0, 1), (1, 1, 2), (1, 2.5, 3), (5, 5, 2)):
        c = copy.deepcopy(CFG)
        c["mission"]["dt_days"], c["maneuver"]["tau_s_days"] = dt, tau
        assert ADRNetwork(DATA, c).k_transfer == k, (dt, tau)


def test_reentry_hook_P6():
    import copy
    from td_route.eecm import TimeExpanded
    d2 = copy.deepcopy(DATA)
    d2.meta["debris"][8]["reentry_days"] = 500.0          # synthetic: D09 on day 500
    net = ADRNetwork(d2, CFG)
    te = TimeExpanded(net, net.S, net.T, t0=0, waiting=True, origin_wait=True, budget=net.H)
    assert not any(w[0] == 9 and w[1] + 1 >= 500 for w in te.waits)
    assert not any(a[0] == 9 and a[1] == net.T for a in te.arcs)
    assert not any((a[1] == 9 and a[2] + te.arcs[a][0] >= 500) or (a[0] == 9 and a[2] >= 500)
                   for a in te.arcs)


def test_exact_front_checkpoint():
    """score_exact front (adr_data.pareto_exact): strictly increasing, dV <= dV_max,
    first point = best single visit."""
    path = os.path.join(CFG["output"]["exact_dir"], "checkpoint.json")
    if not os.path.exists(path):
        pytest.skip("run `python -m adr_data.pareto_exact` first")
    import json
    with open(path, encoding="utf-8") as f:
        pts = sorted(json.load(f)["points"], key=lambda p: p["total_dv_mps"])
    sc = [p["total_score_scaled"] for p in pts]
    dv = [p["total_dv_mps"] for p in pts]
    assert all(b > a for a, b in zip(sc, sc[1:]))
    assert all(b > a for a, b in zip(dv, dv[1:]))
    assert all(d <= dv_max_mps(CFG) + 1e-9 for d in dv)
    assert dv[0] == 0.0 and len(pts[0]["order"]) == 1
    assert pts[0]["total_score"] == pytest.approx(DATA.score[:, :181].max(), abs=1e-9)

# Bi-Objective Time-Dependent Preference-Driven Route Planning

Reference implementation of:

> L. Gao, F. Chu, C. Chen, *"Bi-Objective Optimization for Time-Dependent
> Preference-Driven Route Planning"*, IEEE Transactions on Emerging Topics in
> Computational Intelligence, vol. 10, no. 1, 2026.

Two objectives over a time-dependent road network `G=(V,A)`:

- **f1 — total preference score** `Σ u_ijt · x_ijt` → **maximised**
- **f2 — total travel time** `Σ c_ijt · x_ijt` → **minimised**

Both the travel time `c_ijt` and the preference `u_ijt` of an arc depend on the
departure time `t`. The output is a Pareto front of routes from origin `O` to
destination `D`, plus a fuzzy-logic recommended route.

## Install

```bash
pip install -r requirements.txt   # networkx, pulp (bundles CBC), matplotlib, numpy
```

## Quick start

All settings live in **`config.yaml`** — edit values there; the scripts take no
numeric arguments (only an optional `--config <path>`).

```bash
python -m examples.run_planner      # solve + fuzzy pick + plot (per config.yaml)
python -m examples.plot_edge        # plot one arc's c_ijt & u_ijt over the day
python -m examples.build_dataset    # build the fixed 120-network benchmark
python -m examples.simulate_agent   # animate an agent along a route (window)
python -m examples.plot_route_plane # place a route as a point in the (f1,f2) plane
python -m examples.eecm_endpoints   # exact ε-constraint: endpoints + P2(ε) front, saved to json/csv
python -m td_route.demo             # network sanity check + plot

python -m examples.run_planner --config experiments/big.yaml   # alternative file
```

To change the problem you edit the YAML, e.g.:

```yaml
network_number: "001"                                   # which dataset network
instance: { n_nodes: 80, seed: 7, pref_density: 0.5 }   # |V|, rho, ...
run:      { solver: nsga2, w1: 0.8, plot: true }        # or solver: exact | both
query:    { edge: [2, 17] }                             # arc for plot_edge
simulation: { route: [9, 8, 3, 2, 10], t0: 420 }        # agent route + departure
```

`network_number` selects the fixed dataset network (`dataset/networks/net_NNN.json`)
that `simulate_agent` operates on — bump it to run on another network.

`config.yaml` groups every tunable value: `network` (speed matrix, horizons),
`instance` (|V|, ρ, bp/β ranges, state durations, regions), `query` (O/D, t0,
edge), `nsga2`, `exact`, `run`, `plot_edge`.

## Fixed 120-network benchmark

`python -m examples.build_dataset` writes the paper's benchmark — 12 sizes
`|V| ∈ {10,20,…,100,150,200}` × 10 instances = **120 networks numbered
001–120** (size order; network `NNN` uses `seed = base_seed + NNN`) — under
`dataset/` (configurable via the `dataset` section of `config.yaml`):

```
dataset/
  networks/net_001.json … net_120.json   # self-contained, fully reloadable
  png/net_001.png       … net_120.png     # network-structure images (O/D starred)
  manifest.csv                            # number, size, seed, |A|, ρ, O, D, paths
```

Every network is **strongly connected** (from any node you can reach every
other — no isolated nodes/subnetworks), and each stores its **origin/destination
= the two farthest-apart nodes** (green/red stars in the PNG). Each JSON stores
node positions and every arc's `length, speed_cat, bp, beta, pref_schedule`
plus the speed matrix/horizons — enough to reproduce the exact `c_ijt`/`u_ijt`.
Reload with `td_route.dataset.load_network_json(path)` (verified identical to
regenerating from the seed; O/D exposed on `net.G.graph`).

## Layout

| Module | Paper section | Role |
|---|---|---|
| `td_route/network.py`        | III.A, V.A | Time-dependent graph; `c_ijt`, `u_ijt`; region partition + adjacency |
| `td_route/instance.py`       | V.A        | Random road-network generator + 120-network benchmark |
| `td_route/decision.py`       | IV.A, IV.D | Dominance, non-dominated sort, crowding distance, **fuzzy selection** (eqs 15-17) |
| `td_route/metrics.py`        | V.A.2      | Cardinality, hypervolume ratio (HR), e-dominance (eqs 18-19) |
| `td_route/eecm/`             | IV.B       | **Exact ε-constraint** package: `time_expanded`, `solve`, `endpoints`, `core` |
| `td_route/nsga2/encoding.py` | IV.C.2,4   | Region chromosome encoding + feasibility condition |
| `td_route/nsga2/imoop.py`    | App. A,C   | Improved arc table, time budget (longest route), **IMOOP initial population** |
| `td_route/nsga2/operators.py`| IV.C.3-5   | 1-point crossover, mutation, repair, **region-based decoding** |
| `td_route/nsga2/core.py`     | IV.C       | **Problem-specific NSGA-II** main loop |
| `examples/run_planner.py`    |            | CLI driver (solve, compare, fuzzy pick, plot) |

## Key implementation decisions

- **Exact method (small instances).** The decision variable `x_ijt` is built
  only over **reachable time-states** of a time-expanded graph, pruned by a
  *safe* travel-time budget (sum of the `|V|-1` slowest arcs) so no
  Pareto-optimal route is lost. The front is traced with `P2(εk)` (minimise
  travel time s.t. preference ≥ εk), raising εk by δ=1 until the program is
  **infeasible** — this avoids ever solving the NP-hard longest-path direction
  `P(f1^I)`, which is exactly why the paper prefers `P2` over `P1` (Fig. 10).
  - ⚠️ With CBC, set **no** `--exact-time-limit` for a *truly exact* front; a
    time limit makes CBC return an incumbent (approximate). Practical for
    `|V| ≲ 15` on CBC; the paper used CPLEX for `|V|<60`.
- **NSGA-II (large instances).** Individuals are routes carrying a **region
  chromosome**; one chromosome **decodes into a set** of feasible routes
  confined to its regions. Crossover/mutation preserve the `O`/`D` endpoint
  regions; infeasible children are repaired by inserting the region-shortest
  path across a non-adjacent gap. Path *search* uses static weights; every
  candidate is then scored with the **true** time-dependent objectives.

## Validation

On `|V|=10` (seed 3) the NSGA-II front matches the true exact ε-constraint
front point-for-point (e.g. both find `(f1=23, f2=150)`), while running in
~0.14 s vs the exact method's tens of seconds. NSGA-II scales to `|V|=80` in
~0.2 s.

## Notes on the time-dependent models

- **Speed model — exact [11].** The 4 unequal time periods (7-9, 9-17, 17-19,
  19-21) and the 5 arc categories use the **exact speed matrix of Table 1** in
  Verbeeck et al. (2014, EJOR), and `travel_time` implements their **FIFO
  piecewise Algorithm 1**: the speed changes when a period boundary is crossed
  mid-arc, then `c_ijt = ⌈t_a − t_d⌉`.
- **Preference model.** A fraction ρ (`pref_density`) of arcs are preference
  arcs with a basic score `bp ~ U{1..10}`; the rest have score 0. Each
  preference arc gets its OWN piecewise multiplier `m_ij(t) ∈ {1, 1-β, 1+β}`
  with `β ~ U[0.2, 0.8]` per arc, and states each lasting `U[60, 240]` minutes.
  The score is `u_ijt = ⌈bp · m_ij(t)⌉`. See `build_pref_schedule` in
  `td_route/network.py` and `_arc_attrs` in `td_route/instance.py`.
- Daytime only (7am-9pm), as in [11]; times outside clamp to the nearest period.

Use `examples/plot_edge.py` to visualise any arc's `c_ijt` and `u_ijt` curves.

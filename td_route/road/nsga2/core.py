"""
Problem-specific NSGA-II main loop (Section IV.C, Fig. 2).

Population individuals are decision.Solution objects (a route + its true
time-dependent objectives) carrying their region chromosome in
meta['chromosome']. Genetic operators act on the chromosomes; the region-based
decoder turns each child chromosome into a SET of feasible solutions
(Section IV.C.5), so one crossover can spawn several offspring.

Selection uses Deb's non-dominated sorting + crowding distance with the
crowded-comparison operator. An elitist archive keeps every non-dominated
solution discovered, and the run stops at gmax generations OR when a generation
adds no new non-dominated solution.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ...decision import (Solution, non_dominated, dedup_front,
                        fast_non_dominated_sort, crowding_distance)
from .imoop import initial_population
from .operators import (crossover_one_point, mutate, repair, decode_chromosome,
                       _region_graph)


@dataclass
class NSGA2Config:
    pop_size: int = 50          # paper: 50
    crossover_prob: float = 0.85
    mutation_prob: float = 0.15
    gmax: int = 5               # paper: max 5 iterations
    max_decode: int = 6         # solutions decoded per chromosome
    seed: int | None = None


class NSGA2:
    def __init__(self, net, O, D, t0=None, config: NSGA2Config | None = None):
        self.net = net
        self.O, self.D = O, D
        self.t0 = net.t_start if t0 is None else t0
        self.cfg = config or NSGA2Config()
        self.rng = random.Random(self.cfg.seed)
        self.adjacency = net.region_adjacency()
        self.region_graph = _region_graph(self.adjacency)
        self.all_regions = set(self.adjacency.keys())
        self.history = []        # cardinality of the archive per generation

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(s: Solution):
        return (round(s.f1, 6), round(s.f2, 6), tuple(s.route))

    def _dedup(self, sols):
        seen, out = set(), []
        for s in sols:
            k = self._key(s)
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    def _ranks_and_crowding(self, pop):
        """Return {id: rank} and {id: crowding} for the population."""
        fronts = fast_non_dominated_sort(pop)
        rank, crowd = {}, {}
        for r, front in enumerate(fronts):
            cd = crowding_distance(front)
            for s in front:
                rank[id(s)] = r
                crowd[id(s)] = cd[id(s)]
        return rank, crowd, fronts

    def _tournament(self, pop, rank, crowd):
        a, b = self.rng.choice(pop), self.rng.choice(pop)
        if rank[id(a)] != rank[id(b)]:
            return a if rank[id(a)] < rank[id(b)] else b
        return a if crowd[id(a)] >= crowd[id(b)] else b

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self, verbose=False):
        cfg = self.cfg
        pop = initial_population(self.net, self.O, self.D, self.t0,
                                 cfg.pop_size, rng=self.rng)
        if not pop:
            return []
        for s in pop:
            s.meta.setdefault("chromosome",
                              None)   # initial_population already set these
        archive = dedup_front(non_dominated(pop))
        prev_front_keys = {(round(s.f1, 6), round(s.f2, 6)) for s in archive}
        self.history = [len(archive)]

        for g in range(cfg.gmax):
            rank, crowd, _fronts = self._ranks_and_crowding(pop)

            # --- create offspring chromosomes ---------------------------
            child_chromos = []
            while len(child_chromos) < cfg.pop_size:
                p1 = self._tournament(pop, rank, crowd)
                p2 = self._tournament(pop, rank, crowd)
                c1 = list(p1.meta["chromosome"])
                c2 = list(p2.meta["chromosome"])
                if self.rng.random() < cfg.crossover_prob:
                    c1, c2 = crossover_one_point(c1, c2, self.rng)
                for c in (c1, c2):
                    if self.rng.random() < cfg.mutation_prob:
                        c = mutate(c, self.all_regions, self.rng)
                    c = repair(c, self.adjacency, self.rng, self.region_graph)
                    child_chromos.append(c)

            # --- decode children into feasible solutions ----------------
            offspring = []
            for c in child_chromos:
                offspring.extend(
                    decode_chromosome(self.net, c, self.O, self.D, self.t0,
                                      self.rng, max_solutions=cfg.max_decode))

            # --- population renewal (R_t = P_t U Q_t) -------------------
            R = self._dedup(pop + offspring)
            fronts = fast_non_dominated_sort(R)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= cfg.pop_size:
                    new_pop.extend(front)
                else:
                    cd = crowding_distance(front)
                    front_sorted = sorted(front, key=lambda s: -cd[id(s)])
                    new_pop.extend(front_sorted[:cfg.pop_size - len(new_pop)])
                    break
            pop = new_pop

            # --- elitist archive + stopping check -----------------------
            archive = dedup_front(non_dominated(archive + fronts[0]))
            front_keys = {(round(s.f1, 6), round(s.f2, 6)) for s in archive}
            self.history.append(len(archive))
            if verbose:
                print(f"  gen {g+1}: archive |F|={len(archive)}  "
                      f"pop={len(pop)}")
            new_points = front_keys - prev_front_keys
            if not new_points and g > 0:
                if verbose:
                    print("  no new non-dominated solutions -> stop")
                break
            prev_front_keys = front_keys

        # final Pareto front
        front = dedup_front(non_dominated(archive))
        front.sort(key=lambda s: (s.f2, -s.f1))
        return front


"""
Reduced instances: n in {4,5,6,7,8}, three subsets per n (seed 0, 1, 2), each with at
least one long-lived (D01-D06) and one re-entering (D07-D10) debris.

    python -m bench.instances   -> bench/instances.json
"""
import json
import os

import numpy as np

from bench.common import LONG, REENTRY, ROOT

ALL = LONG + REENTRY


def draw(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        s = sorted(rng.choice(ALL, size=n, replace=False).tolist())
        if any(d in LONG for d in s) and any(d in REENTRY for d in s):
            return s


def main():
    inst = []
    for n in (4, 5, 6, 7, 8):
        seen = set()
        for seed in (0, 1, 2):
            ids = draw(n, seed)
            k = 0
            while tuple(ids) in seen:          # keep the three subsets distinct
                k += 1
                ids = draw(n, 1000 * k + seed)
            seen.add(tuple(ids))
            inst.append({"name": f"n{n}_s{seed}", "n": n, "seed": seed, "ids": ids,
                         "set": "full_1095"})
    with open(os.path.join(ROOT, "bench", "instances.json"), "w", encoding="utf-8") as f:
        json.dump(inst, f, indent=1)
    for i in inst:
        print(i["name"], i["ids"])


if __name__ == "__main__":
    main()

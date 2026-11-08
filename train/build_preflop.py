"""
train/build_preflop.py - preflop equity (vs a uniformly random hand, full
5-card runout) for every one of the C(36,5) = 376,992 starting hands.

Hands are grouped into suit-isomorphism classes (24 suit permutations); each
class is simulated once with R runouts, then the value is copied to every
member. Output: plo5bot/data/preflop_eq.npy, uint16 = round(equity * 65535),
indexed by the colex index of the sorted hand (see plo5bot.abstraction).

    python train/build_preflop.py --runouts 40000 --procs 12
"""

import argparse
import os
import sys
import time
from itertools import combinations, permutations
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plo5bot.abstraction import colex_index_np  # noqa: E402


def canonical_classes():
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    idx = colex_index_np(combos)
    best = None
    for perm in permutations(range(4)):
        pm = np.array(perm)
        mapped = (combos >> 2) * 4 + pm[combos & 3]
        mapped.sort(axis=1)
        ci = colex_index_np(mapped)
        best = ci if best is None else np.minimum(best, ci)
    return combos, idx, best


def _work(args):
    from plo5bot import core
    holes, R, seed = args
    return core.preflop_equity(holes, R, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runouts", type=int, default=40000)
    p.add_argument("--procs", type=int, default=12)
    a = p.parse_args()
    t0 = time.time()
    combos, idx, canon = canonical_classes()
    reps, inverse = np.unique(canon, return_inverse=True)
    # representative hand for each class: first member
    first = np.full(len(reps), -1)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(reps)))
    first = order[starts]
    rep_hands = combos[first].astype(np.int32)
    print(f"{len(reps)} suit-isomorphism classes ({time.time() - t0:.1f}s)")
    chunks = np.array_split(np.arange(len(rep_hands)), a.procs * 8)
    with Pool(a.procs) as pool:
        parts = pool.map(_work, [(rep_hands[c], a.runouts, 1000 + i) for i, c in enumerate(chunks)])
    eq_rep = np.concatenate(parts)
    eq = np.empty(len(combos))
    eq[idx] = eq_rep[inverse]
    out = np.round(eq * 65535).astype(np.uint16)
    os.makedirs(os.path.join(ROOT, "plo5bot", "data"), exist_ok=True)
    np.save(os.path.join(ROOT, "plo5bot", "data", "preflop_eq.npy"), out)
    print(f"done in {time.time() - t0:.1f}s; equity range {eq.min():.3f}..{eq.max():.3f}, mean {eq.mean():.4f}")


if __name__ == "__main__":
    main()

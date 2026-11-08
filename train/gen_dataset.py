"""
train/gen_dataset.py - sample random deals, compute both players' features
on every street (same sample sizes as play), fit the bucket edges, and write

    plo5bot/data/abstraction.npz   bucket edges (committed, used at play time)
    train/data/deals.npz           buckets[N,2,4] uint8, result[N] int8 (not committed)

result[d] = +1 if hole0 (cards 0-4) beats hole1 (cards 5-9) at showdown.

    python train/gen_dataset.py --deals 4000000 --procs 12
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plo5bot import abstraction as A  # noqa: E402


def _work(args):
    from plo5bot import core
    n, seed = args
    return core.gen_deals(n, seed, A.FLOP_R, A.FLOP_K, A.TURN_R, A.TURN_K, A.RIVER_K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=4_000_000)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=os.path.join(ROOT, "train", "data", "deals.npz"))
    a = p.parse_args()
    t0 = time.time()
    chunk = 20000
    tasks = [(min(chunk, a.deals - i), a.seed * 1_000_003 + i) for i in range(0, a.deals, chunk)]
    with Pool(a.procs) as pool:
        parts = pool.map(_work, tasks)
    cards = np.concatenate([p_[0] for p_ in parts])
    feats = np.concatenate([p_[1] for p_ in parts]).astype(np.float64)
    res = np.concatenate([p_[2] for p_ in parts])
    print(f"generated {len(cards)} deals in {time.time() - t0:.0f}s")

    pre = A.preflop_table()
    holes = np.sort(cards[:, :10].reshape(-1, 2, 5).astype(np.int64), axis=2)
    pre_eq = pre[A.colex_index_np(holes.reshape(-1, 5))].reshape(-1, 2).astype(np.float64)

    abs_ = {"pre_edges": A.quantile_edges(pre.astype(np.float64), A.NB_PRE)}
    for key, off in (("flop", 0), ("turn", 3)):
        e = feats[:, :, off].ravel()
        var = (feats[:, :, off + 1] - feats[:, :, off] ** 2).ravel()
        eq_edges = A.quantile_edges(e, A.NB_EQ)
        eb = np.searchsorted(eq_edges, e, side="right")
        pot = np.zeros((A.NB_EQ, A.NB_POT - 1))
        for b in range(A.NB_EQ):
            pot[b] = A.quantile_edges(var[eb == b], A.NB_POT)
        abs_[f"{key}_eq_edges"] = eq_edges
        abs_[f"{key}_pot_edges"] = pot
    abs_["riv_edges"] = A.quantile_edges(feats[:, :, 6].ravel(), A.NB_RIV)
    abs_["meta"] = np.array([A.FLOP_R, A.FLOP_K, A.TURN_R, A.TURN_K, A.RIVER_K,
                             A.NB_PRE, A.NB_EQ, A.NB_POT, A.NB_RIV])
    np.savez(os.path.join(ROOT, "plo5bot", "data", "abstraction.npz"), **abs_)

    buckets = np.zeros((len(cards), 2, 4), dtype=np.uint8)
    for pl in range(2):
        buckets[:, pl, 0] = A.buckets_from_features_np(0, pre_eq[:, pl], pre_eq[:, pl] ** 2, abs_)
        buckets[:, pl, 1] = A.buckets_from_features_np(1, feats[:, pl, 0], feats[:, pl, 1], abs_)
        buckets[:, pl, 2] = A.buckets_from_features_np(2, feats[:, pl, 3], feats[:, pl, 4], abs_)
        buckets[:, pl, 3] = A.buckets_from_features_np(3, feats[:, pl, 6], feats[:, pl, 6] ** 2, abs_)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, buckets=buckets, result=res, cards=cards,
             feats=feats.astype(np.float16), pre_eq=pre_eq.astype(np.float16))
    for s in range(4):
        cnt = np.bincount(buckets[:, :, s].ravel(), minlength=A.NB[s])
        print(f"street {s}: buckets used {np.count_nonzero(cnt)}/{A.NB[s]}, min/max mass "
              f"{cnt.min() / cnt.sum():.4f}/{cnt.max() / cnt.sum():.4f}")
    print(f"showdown results: p0 win {np.mean(res == 1):.4f} tie {np.mean(res == 0):.4f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

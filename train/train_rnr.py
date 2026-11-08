"""
train/train_rnr.py - Restricted Nash Response counter-strategies.

For every opponent archetype below (a tilt of the blueprint, see mccfr.c),
train a counter-strategy with MCCFR in RNR mode, warm-started from the
blueprint, then measure in the abstract game (exact reach-weighted
traversal over held-out-ish deals):
    * the archetype's action frequencies per situation class (used by
      plo5bot/exploit.py to recognise the opponent at play time)
    * EV of the blueprint and of the counter-strategy against the archetype
Writes plo5bot/data/models.npz.

    python train/train_rnr.py --iters 40000000 --threads 12
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plo5bot import abstraction as A  # noqa: E402
from plo5bot.exploit import N_CLASSES, situation_class  # noqa: E402
from plo5bot.tree import get_tree  # noqa: E402
from train.train_blueprint import DATA, load_raw, quantize, to_strategy  # noqa: E402

# name, p_fixed, preflop (mult[4], add[4]), postflop (mult[4], add[4]); actions F, C, H, P
ARCHETYPES = [
    ("blueprint", None, None, None),
    ("station", 0.9, ([0, 0, 0, 0], [0, 1, 0, 0]), ([0, 0, 0, 0], [0, 1, 0, 0])),
    ("loose_passive", 0.75, ([0.35, 1, 0.5, 0.5], [0, 0.02, 0, 0]), ([0.35, 1, 0.5, 0.5], [0, 0.02, 0, 0])),
    ("overfolder", 0.75, ([2.5, 1, 0.7, 0.7], [0.03, 0, 0, 0]), ([2.5, 1, 0.7, 0.7], [0.03, 0, 0, 0])),
    ("maniac", 0.75, ([0.4, 1, 2.5, 2.5], [0, 0, 0.05, 0.05]), ([0.4, 1, 2.5, 2.5], [0, 0, 0.05, 0.05])),
    ("always_pot", 0.9, ([0, 0, 0, 0], [0, 0.001, 0, 1]), ([0, 0, 0, 0], [0, 0.001, 0, 1])),
    ("always_half", 0.9, ([0, 0, 0, 0], [0, 0.001, 1, 0]), ([0, 0, 0, 0], [0, 0.001, 1, 0])),
    ("random", 0.9, ([0, 0, 0, 0], [1, 1, 1, 1]), ([0, 0, 0, 0], [1, 1, 1, 1])),
]


def model_sigma(bp_rows, legal, params):
    """bp_rows (D,4) blueprint probabilities -> tilted model probabilities."""
    mult, add = np.asarray(params[0], float), np.asarray(params[1], float)
    v = np.maximum(bp_rows * mult + add, 0) * legal
    tot = v.sum(axis=1, keepdims=True)
    uni = legal / legal.sum()
    return np.where(tot > 0, v / np.maximum(tot, 1e-300), uni)


def play(us_strat, bp_strat, arche, u, buckets, res0):
    """Exact (over deals) play of `us` (position u) vs opponent = archetype.
    Returns (EV for us in chips per hand, opponent class counts (9,4))."""
    t = get_tree()
    D = len(res0)
    counts = np.zeros((N_CLASSES, 4))
    sign = 1.0 if u == 0 else -1.0

    def opp_sig(node, s, bk):
        rows = bp_strat[s][t.local[node]][bk]
        if arche[1] is None:
            return rows
        legal = (t.child[node] >= 0).astype(float)
        params = arche[2] if s == 0 else arche[3]
        return model_sigma(rows, legal, params)

    def rec(node, reach):
        if t.kind[node] == 1:
            pay0 = -float(t.contrib[node, 0]) if t.folder[node] == 0 else float(t.contrib[node, 1])
            return np.full(D, sign * pay0)
        if t.kind[node] == 2:
            return sign * res0 * float(t.contrib[node, 0])
        p, s = int(t.player[node]), int(t.street[node])
        bk = buckets[:, p, s]
        if p == u:
            sig = us_strat[s][t.local[node]][bk]
        else:
            sig = opp_sig(node, s, bk)
            counts[situation_class(t, node)] += (reach[:, None] * sig).sum(axis=0)
        v = np.zeros(D)
        for a in range(4):
            if t.child[node, a] < 0:
                continue
            r2 = reach * sig[:, a]
            if r2.any():
                v += sig[:, a] * rec(int(t.child[node, a]), r2)
        return v

    return rec(0, np.ones(D)).mean(), counts


def float_strategy(table_u8):
    q = table_u8.astype(np.float64)
    tot = q.sum(axis=2, keepdims=True)
    return np.where(tot > 0, q / np.maximum(tot, 1), 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=40_000_000)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--eval-deals", type=int, default=12000)
    p.add_argument("--only", default="", help="comma list of archetype names to (re)train")
    p.add_argument("--skip-train", action="store_true")
    a = p.parse_args()
    exe = os.path.join(ROOT, "train", "mccfr")
    bp_raw = os.path.join(DATA, "blueprint_raw.bin")
    bp_iters = json.load(open(bp_raw + ".json"))["iters"]
    bp = np.load(os.path.join(ROOT, "plo5bot", "data", "blueprint.npz"))
    bp_strat = [float_strategy(bp[f"s{s}"]) for s in range(4)]
    tables = {}
    only = set(filter(None, a.only.split(",")))
    for k, arche in enumerate(ARCHETYPES):
        if arche[1] is None:
            continue
        name = arche[0]
        raw = os.path.join(DATA, f"rnr_{name}.bin")
        if not a.skip_train and (not only or name in only):
            pf = os.path.join(DATA, f"rnr_{name}.params")
            with open(pf, "w") as f:
                vals = [arche[1]] + list(arche[2][0]) + list(arche[2][1]) + list(arche[3][0]) + list(arche[3][1])
                f.write(" ".join(str(v) for v in vals))
            t0 = time.time()
            subprocess.check_call([exe, os.path.join(DATA, "tree.bin"), os.path.join(DATA, "deals.bin"), raw,
                                   str(a.iters), str(a.threads), str(100 + k), bp_raw, str(bp_iters), pf])
            print(f"[{name}] trained in {time.time() - t0:.0f}s", flush=True)
        strat = to_strategy(load_raw(raw))
        tables[name] = [quantize(strat[f"s{s}"]) for s in range(4)]

    # ---- evaluation in the abstract game
    z = np.load(os.path.join(DATA, "deals.npz"))
    rng = np.random.default_rng(7)
    idx = rng.choice(len(z["result"]), a.eval_deals, replace=False)
    b = z["buckets"][idx].astype(np.int64)
    r = z["result"][idx].astype(np.float64)
    buckets = np.concatenate([b, b[:, ::-1, :]])
    res0 = np.concatenate([r, -r])
    names, freqs, report = [], [], {}
    out = {}
    for k, arche in enumerate(ARCHETYPES):
        name = arche[0]
        us = bp_strat if arche[1] is None else [float_strategy(x) for x in tables[name]]
        ev_c, ev_b, cnt = 0.0, 0.0, np.zeros((N_CLASSES, 4))
        for u in (0, 1):
            v, c = play(us, bp_strat, arche, u, buckets, res0)
            ev_c += v / 2
            cnt += c
            if arche[1] is not None:
                ev_b += play(bp_strat, bp_strat, arche, u, buckets, res0)[0] / 2
            else:
                ev_b = ev_c
        freq = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1e-12)
        names.append(name)
        freqs.append(freq)
        report[name] = {"ev_counter_bb": round(ev_c / 2, 3), "ev_blueprint_bb": round(ev_b / 2, 3)}
        print(f"{name:14s} EV vs archetype (bb/hand): blueprint {ev_b / 2:+.3f}  counter {ev_c / 2:+.3f}", flush=True)
        if arche[1] is not None:
            for s in range(4):
                out[f"m{k}_s{s}"] = tables[name][s]
    prior = np.array([0.4] + [0.6 / (len(ARCHETYPES) - 1)] * (len(ARCHETYPES) - 1))
    np.savez_compressed(os.path.join(ROOT, "plo5bot", "data", "models.npz"), names=np.array(names),
                        freq=np.array(freqs), prior=prior, **out)
    json.dump(report, open(os.path.join(ROOT, "experiments", "rnr_abstract_eval.json"), "w"), indent=1)
    print("wrote plo5bot/data/models.npz")


if __name__ == "__main__":
    main()

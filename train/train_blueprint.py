"""
train/train_blueprint.py - train the MCCFR blueprint and export it for play.

    python train/train_blueprint.py --iters 400000000 --threads 12
    python train/train_blueprint.py --iters 400000000 --resume   # continue training

Steps: export the exact betting tree and the bucketed deal dataset to flat
binary files, compile train/mccfr.c, run it, and convert the accumulated
average strategy into plo5bot/data/blueprint.npz:
    s{street}: uint8 [n_local[street], NB[street], 4]  (probability * 255)
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
from plo5bot.tree import get_tree  # noqa: E402

DATA = os.path.join(ROOT, "train", "data")

NODE_DT = np.dtype([("player", "i1"), ("street", "i1"), ("kind", "i1"), ("folder", "i1"),
                    ("child", "i4", (4,)), ("c0", "i4"), ("c1", "i4"), ("local", "i4")])


def export_tree(path):
    t = get_tree()
    arr = np.zeros(t.n, dtype=NODE_DT)
    arr["player"] = t.player
    arr["street"] = t.street
    arr["kind"] = t.kind
    arr["folder"] = t.folder
    arr["child"] = t.child
    arr["c0"] = t.contrib[:, 0]
    arr["c1"] = t.contrib[:, 1]
    arr["local"] = t.local
    header = np.array([t.n] + t.n_local + list(A.NB), dtype=np.int32)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())
    assert NODE_DT.itemsize == 32


def export_deals(npz_path, path):
    z = np.load(npz_path)
    b = np.ascontiguousarray(z["buckets"], dtype=np.uint8)
    r = np.ascontiguousarray(z["result"], dtype=np.int8)
    with open(path, "wb") as f:
        f.write(np.array([len(r)], dtype=np.int64).tobytes())
        f.write(b.tobytes())
        f.write(r.tobytes())
    return len(r)


def load_raw(path):
    t = get_tree()
    raw = np.fromfile(path, dtype=np.float64)
    out, off = [], 0
    for s in range(4):
        n = t.n_local[s] * A.NB[s] * 4
        R = raw[off:off + n].reshape(t.n_local[s], A.NB[s], 4)
        off += n
        S = raw[off:off + n].reshape(t.n_local[s], A.NB[s], 4)
        off += n
        out.append((R, S))
    assert off == len(raw)
    return out


def legal_mask(street):
    t = get_tree()
    idx = np.where((t.kind == 0) & (t.street == street))[0]
    idx = idx[np.argsort(t.local[idx])]
    return t.child[idx] >= 0     # (n_local, 4)


def to_strategy(raw):
    """Average strategy per street, normalised over legal actions. Info sets
    never reached fall back to the current regret-matching strategy, then to
    a 'check/call' default."""
    out = {}
    for s, (R, S) in enumerate(raw):
        mask = legal_mask(s)[:, None, :].repeat(A.NB[s], axis=1)
        tot = S.sum(axis=2, keepdims=True)
        avg = np.where(tot > 0, S / np.maximum(tot, 1e-300), 0.0)
        pos = np.where(mask, np.maximum(R, 0), 0)
        ptot = pos.sum(axis=2, keepdims=True)
        cur = np.where(ptot > 0, pos / np.maximum(ptot, 1e-300), 0.0)
        default = np.zeros_like(avg)
        default[..., 1] = 1.0
        strat = np.where(tot > 0, avg, np.where(ptot > 0, cur, default))
        strat = np.where(mask, strat, 0.0)
        strat /= strat.sum(axis=2, keepdims=True)
        out[f"s{s}"] = strat
        out[f"visits{s}"] = tot[..., 0]
    return out


def quantize(strat):
    q = np.round(strat * 255).astype(np.int32)
    # keep every legal action that had probability > 0.2% at >= 1/255 is NOT
    # wanted (adds noise); instead just renormalise at play time.
    return np.clip(q, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200_000_000)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deals", default=os.path.join(DATA, "deals.npz"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--raw", default=os.path.join(DATA, "blueprint_raw.bin"))
    p.add_argument("--out", default=os.path.join(ROOT, "plo5bot", "data", "blueprint.npz"))
    a = p.parse_args()
    os.makedirs(DATA, exist_ok=True)
    exe = os.path.join(ROOT, "train", "mccfr")
    src = os.path.join(ROOT, "train", "mccfr.c")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-o", exe, src, "-lm"])
    tree_bin = os.path.join(DATA, "tree.bin")
    deals_bin = os.path.join(DATA, "deals.bin")
    export_tree(tree_bin)
    n = export_deals(a.deals, deals_bin)
    meta_path = a.raw + ".json"
    t_off = 0
    init = "-"
    if a.resume and os.path.exists(a.raw):
        init = a.raw + ".prev"
        os.replace(a.raw, init)
        t_off = json.load(open(meta_path))["iters"]
    t0 = time.time()
    subprocess.check_call([exe, tree_bin, deals_bin, a.raw, str(a.iters), str(a.threads), str(a.seed),
                           init, str(t_off)])
    json.dump({"iters": t_off + a.iters, "deals": n, "seconds": time.time() - t0}, open(meta_path, "w"))
    strat = to_strategy(load_raw(a.raw))
    np.savez_compressed(a.out, **{f"s{s}": quantize(strat[f"s{s}"]) for s in range(4)},
                        nb=np.array(A.NB), iters=np.array([t_off + a.iters]))
    for s in range(4):
        v = strat[f"visits{s}"]
        print(f"street {s}: info sets {v.size}, never-averaged {np.mean(v == 0):.3f}")
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB), total iters {t_off + a.iters}")


if __name__ == "__main__":
    main()

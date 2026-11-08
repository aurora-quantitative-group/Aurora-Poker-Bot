"""
train/eval_abstract.py - exploitability of a blueprint INSIDE the abstract game.

For each position i, computes a best response (BR) of player i against the
blueprint of the other player, over D sampled deals (both seatings), with
the BR's information sets = (betting node, own bucket on that street) - the
same information the blueprint has. Reports, in big blinds per hand:
    value(BR_btn vs bp) + value(BR_bb vs bp)  / 2  = exploitability
which is 0 at an abstract-game equilibrium. This is a convergence /
sanity metric for MCCFR, not the exploitability in the real game.

    python train/eval_abstract.py plo5bot/data/blueprint.npz --deals 40000
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plo5bot.tree import get_tree  # noqa: E402


def load_strategy(path):
    z = np.load(path)
    out = []
    for s in range(4):
        q = z[f"s{s}"].astype(np.float64)
        tot = q.sum(axis=2, keepdims=True)
        out.append(np.where(tot > 0, q / np.maximum(tot, 1), 0.0))
    return out


class Evaluator:
    def __init__(self, strat, buckets, result):
        self.t = get_tree()
        self.strat = strat
        self.b = buckets        # (D, 2, 4) bucket of player (position) p on street s
        self.res0 = result.astype(np.float64)   # +1 if position 0 wins showdown

    def payoff(self, node):
        t = self.t
        if t.kind[node] == 1:
            pay0 = -float(t.contrib[node, 0]) if t.folder[node] == 0 else float(t.contrib[node, 1])
            return np.full(len(self.res0), pay0)
        return self.res0 * float(t.contrib[node, 0])

    def br_value(self, br, fixed=None):
        """Per-deal value for position br best-responding to the blueprint.
        If `fixed` (a dict node -> best action per bucket, -1 = unknown) is
        given, play that policy instead of maximising (held-out evaluation);
        otherwise the fitted policy is stored in self.policy."""
        sign = 1.0 if br == 0 else -1.0
        t = self.t
        self.policy = {}

        def rec(node, reach):
            if t.kind[node] != 0:
                return sign * self.payoff(node)
            p, s = int(t.player[node]), int(t.street[node])
            legal = [a for a in range(4) if t.child[node, a] >= 0]
            bk = self.b[:, p, s]
            if p == br and fixed is not None:
                choice = fixed.get(node)
                sig = self.strat[s][t.local[node]][bk].copy()        # unknown -> follow blueprint
                if choice is not None:
                    c = choice[bk]
                    known = c >= 0
                    sig[known] = 0
                    sig[known, c[known]] = 1
                v = np.zeros(len(reach))
                for a in legal:
                    r2 = reach * sig[:, a]
                    if r2.any():
                        v += sig[:, a] * rec(int(t.child[node, a]), r2)
                return v
            if p != br:
                sig = self.strat[s][t.local[node]][bk]            # (D, 4)
                v = np.zeros(len(reach))
                for a in legal:
                    w = sig[:, a]
                    r2 = reach * w
                    if not r2.any():
                        continue
                    v += w * rec(int(t.child[node, a]), r2)
                return v
            vals = [rec(int(t.child[node, a]), reach) for a in legal]
            nb = 128
            score = np.stack([np.bincount(bk, weights=reach * v, minlength=nb) for v in vals], axis=1)
            best = np.argmax(score, axis=1)
            seen = np.bincount(bk, weights=reach, minlength=nb) > 0
            self.policy[node] = np.where(seen, np.array(legal)[best], -1)
            V = np.stack(vals, axis=1)
            return V[np.arange(len(reach)), best[bk]]

        return rec(0, np.ones(len(self.res0)))

    def self_play_value(self):
        """Value for position 0 when both play the blueprint."""
        t = self.t

        def rec(node):
            if t.kind[node] != 0:
                return self.payoff(node)
            p, s = int(t.player[node]), int(t.street[node])
            sig = self.strat[s][t.local[node]][self.b[:, p, s]]
            v = np.zeros(len(self.res0))
            for a in range(4):
                if t.child[node, a] >= 0 and sig[:, a].any():
                    v += sig[:, a] * rec(int(t.child[node, a]))
            return v

        return rec(0)


def _split(z, idx):
    b = z["buckets"][idx].astype(np.int64)
    r = z["result"][idx]
    # both seatings: dataset hand 0 as button, and hand 1 as button
    return np.concatenate([b, b[:, ::-1, :]]), np.concatenate([r, -r])


def evaluate(path, deals_path, n, seed=0):
    """In-sample BR (upper-biased: it also exploits sampling noise) and a
    held-out BR (policy fitted on set A, evaluated on disjoint set B: an
    unbiased estimate of a lower bound on the true BR value)."""
    z = np.load(deals_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z["result"]), 2 * n, replace=False)
    strat = load_strategy(path)
    fit = Evaluator(strat, *_split(z, idx[:n]))
    test = Evaluator(strat, *_split(z, idx[n:]))
    out = {}
    for br, name in ((0, "button"), (1, "bigblind")):
        out[f"br_{name}_insample_bb"] = fit.br_value(br).mean() / 2
        pol = fit.policy
        out[f"br_{name}_heldout_bb"] = test.br_value(br, fixed=pol).mean() / 2
    out["exploitability_insample_bb"] = (out["br_button_insample_bb"] + out["br_bigblind_insample_bb"]) / 2
    out["exploitability_heldout_bb"] = (out["br_button_heldout_bb"] + out["br_bigblind_heldout_bb"]) / 2
    out["selfplay_button_bb"] = test.self_play_value().mean() / 2
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("blueprint")
    p.add_argument("--deals", type=int, default=40000)
    p.add_argument("--data", default=os.path.join(ROOT, "train", "data", "deals.npz"))
    a = p.parse_args()
    r = evaluate(a.blueprint, a.data, a.deals)
    import json
    print(json.dumps({k: round(float(v), 4) for k, v in r.items()}))


if __name__ == "__main__":
    main()

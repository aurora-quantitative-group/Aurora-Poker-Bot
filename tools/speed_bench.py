"""
tools/speed_bench.py - speed benchmark that also pins the bot's decisions.

Plays a fixed set of hands (fixed deal seed, scripted opponent with a fixed
RNG) through plo5bot.player.Player exactly as agent.py would, and records
every decision plus every hand feature/bucket. Timing covers choose_action
and on_hand_end (what the judge charges, minus the HTTP plumbing).

    python tools/speed_bench.py --save experiments/speed_baseline_c.json
    python tools/speed_bench.py --compare experiments/speed_baseline_c.json
    PLO5_FORCE_NUMPY=1 python tools/speed_bench.py --save experiments/speed_baseline_np.json

PLO5_DETERMINISTIC=1 is forced so the time manager cannot change sample
sizes; any difference in decisions is then caused by code changes only.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

os.environ["PLO5_DETERMINISTIC"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# --pkg DIR: benchmark the plo5bot package found in DIR (A/B of code versions)
if "--pkg" in sys.argv:
    sys.path.insert(0, os.path.abspath(sys.argv[sys.argv.index("--pkg") + 1]))

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from plo5bot import abstraction as A  # noqa: E402
from plo5bot import core  # noqa: E402
from plo5bot.player import Player  # noqa: E402

# --- count Monte Carlo work: every features() call evaluates R*(K+1) hands
_EVALS = [0, 0.0]          # evaluations, seconds
_orig_hs = core.hs_features


def _counting_hs(hole, board, R, K, seed=0, dead=()):
    t = time.perf_counter()
    out = _orig_hs(hole, board, R, K, seed, dead)
    _EVALS[1] += time.perf_counter() - t
    _EVALS[0] += (1 if len(board) == 5 else R) * (K + 1)
    return out


core.hs_features = _counting_hs


def opponent(obs, rng):
    """Scripted opponent: mixes folds, calls and both raise sizes so every
    part of the tree and the opponent model is exercised."""
    legal = obs.legal_actions
    r = rng.random()
    if Action.FOLD in legal and r < 0.15:
        return Action.FOLD
    raises = [a for a in legal if a >= 2]
    if raises and r > 0.7:
        return rng.choice(raises)
    return Action.CHECK_CALL


def run(hands, seed):
    player = Player()
    mem = {}
    player.new_half(AgentConfig(num_hands=hands), mem)
    rng = random.Random(seed)
    decisions, feats, times = [], [], []
    _EVALS[0], _EVALS[1] = 0, 0.0
    for h in range(hands):
        g = Game(make_deal(f"speed-{seed}", h), h % 2, h)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            if g.to_act == 0:
                t = time.perf_counter()
                a = player.choose_action(obs, mem)
                times.append(time.perf_counter() - t)
                decisions.append([h, int(obs.street), int(a)])
                c = mem["cache"].get((h, int(obs.street)))
                feats.append([round(c[0], 12), round(c[1], 12), c[2]] if c else None)
            else:
                a = opponent(obs, rng)
            g.apply(a)
        t = time.perf_counter()
        player.on_hand_end(g.hand_result(0), mem)
        times.append(time.perf_counter() - t)
    total = sum(times)
    return {
        "backend": core.BACKEND,
        "decisions": decisions,
        "features": feats,
        "n_decisions": len(decisions),
        "total_s": total,
        "per_decision_ms": 1000 * total / len(decisions),
        "p50_ms": 1000 * statistics.median(times),
        "p99_ms": 1000 * sorted(times)[int(0.99 * len(times))],
        "mc_evals": _EVALS[0],
        "mc_evals_per_s": _EVALS[0] / _EVALS[1] if _EVALS[1] else 0.0,
        "mc_share": _EVALS[1] / total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--repeat", type=int, default=3, help="timing runs (decisions must agree)")
    p.add_argument("--save")
    p.add_argument("--compare")
    p.add_argument("--pkg", help="directory containing the plo5bot package to benchmark")
    a = p.parse_args()
    runs = [run(a.hands, a.seed) for _ in range(a.repeat)]
    for r in runs[1:]:
        assert r["decisions"] == runs[0]["decisions"], "nondeterministic decisions between repeats"
    best = min(runs, key=lambda r: r["total_s"])
    print(f"backend {best['backend']}: {best['n_decisions']} decisions, "
          f"{best['per_decision_ms']:.4f} ms/decision (best of {a.repeat}; "
          f"median {statistics.median(r['per_decision_ms'] for r in runs):.4f}), "
          f"p50 {best['p50_ms']:.4f} ms, p99 {best['p99_ms']:.3f} ms, "
          f"MC {best['mc_evals_per_s'] / 1e6:.2f} M evals/s ({100 * best['mc_share']:.0f}% of time)")
    if a.save:
        json.dump(best, open(a.save, "w"))
        print(f"saved baseline -> {a.save}")
    if a.compare:
        base = json.load(open(a.compare))
        same_d = base["decisions"] == best["decisions"]
        same_f = base["features"] == best["features"]
        speedup = base["per_decision_ms"] / best["per_decision_ms"]
        print(f"vs baseline: decisions identical={same_d}, features identical={same_f}, "
              f"speedup {speedup:.3f}x ({base['per_decision_ms']:.4f} -> {best['per_decision_ms']:.4f} ms)")
        if not (same_d and same_f):
            for i, (x, y) in enumerate(zip(base["decisions"], best["decisions"])):
                if x != y:
                    print(f"first decision difference at #{i}: baseline {x} now {y}")
                    break
            sys.exit(1)


if __name__ == "__main__":
    main()

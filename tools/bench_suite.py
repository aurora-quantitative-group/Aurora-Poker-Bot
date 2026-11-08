"""
tools/bench_suite.py - run one agent against the whole opponent pool and
print / save a markdown table (paired duplicate statistics, see arena.py).

    python tools/bench_suite.py agent.py --seeds 1-4 --deals 1000 --jobs 12 --tag v3
    python tools/bench_suite.py agent.py --opponents bots/v2_blueprint/agent.py,bots/opponents/heuristic_tag.py
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.arena import parse_seeds, run  # noqa: E402

DEFAULT_POOL = [
    "baseline-bots/baseline_random.py",
    "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_pot.py",
    "baseline-bots/baseline_always_half.py",
    "bots/opponents/check_fold.py",
    "bots/opponents/heuristic_tag.py",
    "bots/opponents/heuristic_lag.py",
    "bots/opponents/heuristic_nit.py",
    "bots/opponents/pot_odds_caller.py",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent")
    p.add_argument("--opponents", default=",".join(DEFAULT_POOL))
    p.add_argument("--seeds", default="1-4")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="")
    a = p.parse_args()
    rows = []
    t0 = time.time()
    for opp in filter(None, a.opponents.split(",")):
        s = run(a.agent, opp, parse_seeds(a.seeds), a.deals, a.jobs, a.tag, quiet=True)
        rows.append(s)
        print(f"{s['names'][1]:22s} {s['mean_bb_per_deal']:+7.3f} bb/deal  CI [{s['ci95'][0]:+.3f}, "
              f"{s['ci95'][1]:+.3f}]  seeds {dict(s['seed_verdicts'])}  think/half "
              f"{s['agent1_think_s_per_half']['mean']:.2f}s  errors {s['errors'][0]}", flush=True)
    lines = [f"### {a.agent} {a.tag} ({len(parse_seeds(a.seeds))} seeds x {a.deals} deals, "
             f"commit {rows[0]['commit']})", "",
             "| opponent | bb/deal | 95% CI | bb/100 hands | match verdicts (W/C/L) | think s/half (mean/max) | max decision s |",
             "|---|---|---|---|---|---|---|"]
    for s in rows:
        v = s["seed_verdicts"]
        lines.append(f"| {s['names'][1]} | {s['mean_bb_per_deal']:+.3f} | [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] | "
                     f"{s['bb_per_100_hands']:+.1f} | {v.get('WIN', 0)}/{v.get('CHOP', 0)}/{v.get('LOSS', 0)} | "
                     f"{s['agent1_think_s_per_half']['mean']:.2f}/{s['agent1_think_s_per_half']['max']:.2f} | "
                     f"{s['agent1_max_decision_s']:.3f} |")
    table = "\n".join(lines)
    print("\n" + table + f"\n\n(total wall {time.time() - t0:.0f}s)")
    out = a.out or os.path.join(ROOT, "experiments", "benchmarks.md")
    with open(out, "a", encoding="utf-8") as f:
        f.write(table + "\n\n")
    with open(os.path.join(ROOT, "experiments", "last_suite.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

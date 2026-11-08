"""
tools/arena.py - fast duplicate-match benchmarking without HTTP.

Plays real duplicate matches with game.py (same dealing, seats, buttons and
pairing as judge_engine.py) but calls the bots' functions directly. Each bot
runs in its OWN worker process that is restarted for every half, exactly
like the tournament's fresh servers, so bots cannot share globals or remember
half 1 in half 2.

    python tools/arena.py agent.py baseline-bots/baseline_random.py --seeds 1-4 --deals 1000 --jobs 4

A bot is a path to an agent-style .py file. Its directory is put first on
sys.path, so snapshots in bots/<name>/ load their own plo5bot package.
Environment overrides for a bot: "path.py?KEY=VAL&KEY2=VAL2".

Statistics follow the judge: agent1's per-deal result over BOTH halves
(paired_bb[d]) is the unit; mean +- 1.96 * stderr over all deals of all seeds.
Results are appended to experiments/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import BIG_BLIND, Action, AgentConfig, Game, button_for_deal, default_action, make_deal  # noqa: E402

# ---------------------------------------------------------------------------
# Worker process: hosts one bot for one half
# ---------------------------------------------------------------------------


def _worker(conn, spec: str) -> None:
    path, _, query = spec.partition("?")
    for kv in filter(None, query.split("&")):
        k, _, v = kv.partition("=")
        os.environ[k] = v
    path = os.path.abspath(os.path.join(ROOT, path))
    bot_dir = os.path.dirname(path)
    sys.path[:0] = [bot_dir, ROOT]
    import importlib.util

    from game import HandResult, Observation
    spec_ = importlib.util.spec_from_file_location("arena_bot", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    memory: dict = {}
    conn.send(("ready", getattr(mod, "AGENT_NAME", os.path.basename(path))))
    while True:
        msg = conn.recv()
        kind = msg[0]
        t0 = time.perf_counter()
        err = None
        out = None
        try:
            if kind == "new_half":
                memory.clear()
                mod.new_half(AgentConfig.from_dict(msg[1]), memory)
            elif kind == "act":
                for r in msg[2]:
                    mod.on_hand_end(HandResult.from_dict(r), memory)
                a = mod.choose_action(Observation.from_dict(msg[1]), memory)
                out = a.name if isinstance(a, Action) else a
            elif kind == "hand_end":
                for r in msg[1]:
                    mod.on_hand_end(HandResult.from_dict(r), memory)
            elif kind == "stats":
                out = {k: v for k, v in memory.get("_stats", {}).items()} if isinstance(memory.get("_stats"), dict) else None
            elif kind == "quit":
                conn.send(("ok", None, 0.0, None))
                return
        except Exception as exc:  # report like the judge would
            import traceback
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        conn.send(("ok", out, time.perf_counter() - t0, err))


class BotHandle:
    def __init__(self, spec: str):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, spec), daemon=True)
        self.proc.start()
        _, self.name = self.conn.recv()
        self.pending: list = []
        self.think = 0.0
        self.max_think = 0.0
        self.calls = 0
        self.errors: list[str] = []

    def call(self, *msg):
        self.conn.send(msg)
        _, out, dt, err = self.conn.recv()
        self.think += dt
        self.calls += 1
        self.max_think = max(self.max_think, dt)
        if err:
            self.errors.append(err)
        return out, err

    def act(self, obs) -> Action:
        results, self.pending = self.pending, []
        out, err = self.call("act", obs.to_dict(), results)
        fallback = default_action(obs.to_call)
        if err or out not in Action.__members__:
            if not err:
                self.errors.append(f"bad action {out!r}")
            return fallback
        a = Action[out]
        if a not in obs.legal_actions:
            self.errors.append(f"illegal {a.name}")
            return fallback
        return a

    def close(self):
        if self.pending:
            self.call("hand_end", self.pending)
        stats, _ = self.call("stats")
        self.call("quit")
        self.proc.join(timeout=5)
        return stats


# ---------------------------------------------------------------------------
# One match (two halves) for one seed
# ---------------------------------------------------------------------------


def play_half(spec1: str, spec2: str, seed: str, half: int, deals: int) -> dict:
    a1_seat = 0 if half == 1 else 1
    b1, b2 = BotHandle(spec1), BotHandle(spec2)
    bots = (b1, b2) if a1_seat == 0 else (b2, b1)
    cfg = AgentConfig(num_hands=deals).to_dict()
    for b in bots:
        b.call("new_half", cfg)
    nets1 = []
    detail = []        # per hand: (a1_is_button, street_ended, showdown)
    acts1 = Counter()  # agent1 action counts by street
    for d in range(deals):
        button = button_for_deal(d)
        g = Game(make_deal(seed, d), button, hand_index=d)
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            a = bots[seat].act(obs)
            if seat == a1_seat:
                acts1[(int(obs.street), a.name)] += 1
            g.apply(a)
        res = g.result()
        nets1.append(res.nets[a1_seat])
        last_street = int(res.history[-1].street) if res.history else 0
        detail.append((a1_seat == button, last_street, res.went_to_showdown))
        for seat in (0, 1):
            bots[seat].pending.append(g.hand_result(seat).to_dict())
    s1, s2 = b1.close(), b2.close()
    return {
        "nets1": nets1, "detail": detail, "acts1": {f"{k[0]}:{k[1]}": v for k, v in acts1.items()},
        "names": [b1.name, b2.name],
        "think": [b1.think, b2.think], "max_think": [b1.max_think, b2.max_think],
        "calls": [b1.calls, b2.calls], "errors": [b1.errors[:3], b2.errors[:3]],
        "n_errors": [len(b1.errors), len(b2.errors)], "bot_stats": [s1, s2],
    }


def play_match(args) -> dict:
    spec1, spec2, seed, deals = args
    h1 = play_half(spec1, spec2, seed, 1, deals)
    h2 = play_half(spec1, spec2, seed, 2, deals)
    paired = [(a + b) / BIG_BLIND for a, b in zip(h1["nets1"], h2["nets1"])]
    return {"seed": seed, "h1": h1, "h2": h2, "paired_bb": paired}


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------


def ci(values):
    n = len(values)
    m = sum(values) / n
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return m, sd, se, (m - 1.96 * se, m + 1.96 * se)


def summarize(matches, spec1, spec2, deals) -> dict:
    all_paired = [x for m in matches for x in m["paired_bb"]]
    mean, sd, se, (lo, hi) = ci(all_paired)
    per_seed = []
    for m in matches:
        mm, _, _, (l, h) = ci(m["paired_bb"])
        verdict = "WIN" if l > 0 else "LOSS" if h < 0 else "CHOP"
        per_seed.append({"seed": m["seed"], "mean_bb_per_deal": round(mm, 4), "ci": [round(l, 4), round(h, 4)],
                         "verdict": verdict})
    pos = defaultdict(list)
    street = defaultdict(list)
    acts = Counter()
    think1, think2, maxt1, calls1 = [], [], 0.0, []
    errors = [0, 0]
    stats1 = []
    for m in matches:
        for h in (m["h1"], m["h2"]):
            for net, (btn, st, sd_) in zip(h["nets1"], h["detail"]):
                pos["button" if btn else "big_blind"].append(net / BIG_BLIND)
                street[("showdown" if sd_ else ["preflop", "flop", "turn", "river"][st])].append(net / BIG_BLIND)
            acts.update(h["acts1"])
            think1.append(h["think"][0])
            think2.append(h["think"][1])
            calls1.append(h["calls"][0])
            maxt1 = max(maxt1, h["max_think"][0])
            errors[0] += h["n_errors"][0]
            errors[1] += h["n_errors"][1]
            if h["bot_stats"][0]:
                stats1.append(h["bot_stats"][0])
    return {
        "agent1": spec1, "agent2": spec2, "names": matches[0]["h1"]["names"],
        "seeds": [m["seed"] for m in matches], "deals_per_half": deals, "total_deals": len(all_paired),
        "mean_bb_per_deal": round(mean, 4), "bb_per_100_hands": round(mean * 50, 2),
        "std_bb": round(sd, 3), "stderr": round(se, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": "AGENT1_BETTER" if lo > 0 else "AGENT2_BETTER" if hi < 0 else "NOT_SIGNIFICANT",
        "seed_verdicts": Counter(p["verdict"] for p in per_seed), "per_seed": per_seed,
        "by_position_bb_per_hand": {k: round(sum(v) / len(v), 4) for k, v in pos.items()},
        "by_end_street_bb_per_hand": {k: [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(street.items())},
        "agent1_action_freq": dict(sorted(acts.items())),
        "agent1_think_s_per_half": {"mean": round(sum(think1) / len(think1), 3), "max": round(max(think1), 3)},
        "agent1_calls_per_half": round(sum(calls1) / len(calls1), 1),
        "agent1_max_decision_s": round(maxt1, 4),
        "agent2_think_s_per_half": {"mean": round(sum(think2) / len(think2), 3), "max": round(max(think2), 3)},
        "errors": errors, "agent1_stats_sample": stats1[:1],
    }


def git_commit() -> str:
    for exe in ("git", "git.exe"):
        try:
            out = subprocess.check_output([exe, "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                          stderr=subprocess.DEVNULL).decode().strip()
            dirty = subprocess.check_output([exe, "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                            stderr=subprocess.DEVNULL).decode().strip()
            return out + ("-dirty" if dirty else "")
        except Exception:
            continue
    return "unknown"


def parse_seeds(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        if "-" in part and part.replace("-", "").isdigit():
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        elif part:
            out.append(part)
    return out


def run(spec1, spec2, seeds, deals, jobs, tag="", log=True, quiet=False) -> dict:
    t0 = time.time()
    tasks = [(spec1, spec2, f"arena-{s}", deals) for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            matches = list(ex.map(play_match, tasks))
    else:
        matches = [play_match(t) for t in tasks]
    summary = summarize(matches, spec1, spec2, deals)
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["commit"] = git_commit()
    summary["tag"] = tag
    summary["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if log:
        os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
        with open(os.path.join(ROOT, "experiments", "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    if not quiet:
        print(format_summary(summary))
    return summary


def format_summary(s: dict) -> str:
    lines = [
        f"{s['names'][0]} ({s['agent1']})  vs  {s['names'][1]} ({s['agent2']})",
        f"  deals: {s['total_deals']} ({len(s['seeds'])} seeds x {s['deals_per_half']})   "
        f"mean {s['mean_bb_per_deal']:+.3f} bb/deal = {s['bb_per_100_hands']:+.1f} bb/100 hands   "
        f"95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  -> {s['verdict']}",
        f"  per-seed verdicts: {dict(s['seed_verdicts'])}   std/deal {s['std_bb']}",
        f"  by position (bb/hand): {s['by_position_bb_per_hand']}",
        f"  by end street (bb/hand, n): {s['by_end_street_bb_per_hand']}",
        f"  agent1 think/half: {s['agent1_think_s_per_half']}  calls/half {s['agent1_calls_per_half']}  "
        f"max decision {s['agent1_max_decision_s']}s   errors {s['errors']}   wall {s.get('wall_s')}s",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent1")
    p.add_argument("agent2")
    p.add_argument("--seeds", default="1-4")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--tag", default="")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--json", help="also print the summary as JSON", action="store_true")
    a = p.parse_args()
    s = run(a.agent1, a.agent2, parse_seeds(a.seeds), a.deals, a.jobs, a.tag, not a.no_log)
    if a.json:
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

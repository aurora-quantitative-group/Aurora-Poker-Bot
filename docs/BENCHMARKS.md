# Benchmarks

All numbers are **measured**, on a laptop (Intel Core Ultra 7 155U, WSL2 Ubuntu 24.04,
Python 3.12.3), with `tools/arena.py` / `tools/bench_suite.py` unless stated. Raw records:
`experiments/results.jsonl` (one JSON line per match series, with commit, seeds, timing),
`experiments/benchmarks.md` (every suite table in run order).

## Method

* **Duplicate matches exactly like the judge**: `game.make_deal(seed, d)`, button `d % 2`,
  half 1 / half 2 with swapped seats, each half in a **fresh bot process**.
* Unit of analysis = one deal: agent1's chips over **both** halves of that deal, in bb
  (`paired_bb`, identical to `judge_engine.combine_halves`). Mean ± 1.96·SE over all deals of all
  seeds. "Match verdicts" applies the judge's rule to each 1,000-deal seed separately.
* Every opponent is played on the same seeds by every version, so versions are compared on
  identical cards.
* Tuning was done on seeds 1-4; **reported release numbers use unseen seeds** (11-16 and 21-24).

## Ablation — same opponents, same seeds 21-24 (4 × 1,000 deals = 4,000 paired deals per cell)

bb/deal (95% CI half-width ≈ 0.1-0.8, see `experiments/benchmarks.md`):

| opponent | v1 heuristic | v2 blueprint | **v3b final** |
|---|---|---|---|
| RandomBot (baseline) | +2.13 | +3.75 | **+15.11** [+14.26, +15.97] |
| CallingStation (baseline) | +4.44 | +3.34 | **+15.38** [+14.88, +15.89] |
| AlwaysPot (baseline) | +1.72 | +3.65 | **+7.82** [+7.01, +8.63] |
| AlwaysHalf (baseline) | +1.36 | +5.19 | **+14.30** [+13.70, +14.90] |
| CheckFold (= starter agent) | +0.73 | +1.29 | **+1.29** [+1.27, +1.31] |
| HeuristicTAG | +0.03 | +0.61 | **+0.53** [+0.31, +0.75] |
| HeuristicLAG (bluffy) | −0.18 | +0.47 | **+0.55** [+0.30, +0.80] |
| HeuristicNit | +0.31 | +1.03 | **+0.97** [+0.82, +1.11] |
| PotOddsCaller | +1.70 | +1.12 | **+2.37** [+1.96, +2.78] |

* v2 (blueprint) improves on v1 everywhere except vs the calling station (a near-equilibrium
  strategy bluffs into a station).
* v3b's counter-strategies multiply the win rate vs exploitable bots by 2-4×.
* vs TAG / Nit v3b is 0.06-0.08 bb/deal below v2 (not significant): the cost of probation trials of
  a non-fitting archetype. Kept because probation gains +1.25 bb/deal vs the pot-odds caller.

## Head-to-head between versions

| match | deals | bb/deal | 95% CI |
|---|---|---|---|
| v2 blueprint (300M) vs v1 heuristic | 4,000 | +0.478 | [+0.248, +0.709] |
| blueprint 1.8B vs blueprint 300M | 8,000 | +0.060 | [−0.085, +0.205] (n.s.) |
| v3 vs v1 heuristic (seeds 11-16) | 6,000 | +0.504 | [+0.325, +0.684] |
| v3b vs v1 heuristic (seeds 21-24) | 4,000 | +0.458 | [+0.232, +0.685] |
| v3b vs v2 / v3 (mirror: same blueprint, no deviation triggered) | 4,000 | 0.000 | exact |

## Release validation on unseen seeds 11-16 (v3, 6,000 paired deals per opponent)

Positive and significant against all 10 non-mirror opponents: Random +16.02, CallingStation
+15.10, AlwaysPot +7.96, AlwaysHalf +14.63, CheckFold +1.28, TAG +0.51, LAG +0.38, Nit +0.82,
PotOddsCaller +1.68, v1 +0.50 (bb/deal). 0 errors, 0 exceptions.

## Bugs that the benchmarks caught (kept as regression knowledge)

| version | symptom | cause | fix |
|---|---|---|---|
| v3 (first) | −3.88 bb/deal vs LAG | posterior is relative: LAG labelled "random" with p = 1.0 | absolute G-test fit gate |
| v3 gated | station/pot/half bots exploited only for the first ~150 hands | posterior smoothing (0.02) also used in the fit test penalised perfectly predicted bots | separate light smoothing (0.002) for the fit test |
| v3 gated | +2.45 → +0.92 vs PotOddsCaller | gate discarded a profitable approximate fit | probation (non-bluffy archetypes only) |
| numpy fallback | −2.31 bb/deal vs RandomBot over HTTP | time manager returned "no budget" → check/fold every postflop spot | recalibrated, feedback controller, quality test |

## Real HTTP runs with `judge_engine.py` (1,000 deals per half, idle machine)

| agent / backend | opponent | bb/deal | charged thinking per half | requests |
|---|---|---|---|---|
| v3b, C | CallingStation | +16.23 | 2.90 s / 2.80 s of 8 s | 3,984 / 3,990 |
| v3b, C | RandomBot | +15.01 | 2.03 s / 1.96 s | 3,206 / 3,236 |
| v3b, numpy fallback (forced) | CallingStation | +14.28 | 5.54 s / 5.35 s | 3,984 / 3,990 |
| v3b, numpy fallback (forced) | RandomBot | +14.28 | 3.72 s / 3.70 s | 3,155 / 3,172 |

For scale: the do-nothing calling-station bot itself was charged 1.2-1.9 s per half in the same
runs — most of the charge is per-request server overhead, not our computation.

* Startup (process start → answering `GET /`): 2.3-2.9 s (limit 15 s).
* Resident memory: ~165-170 MB.
* Max single decision in the arena: 3-6 ms (limit 1 s). Own compute ≈ 0.2-0.4 ms per decision.
* Errors / forfeits in all runs: 0.

## What is NOT shown here

* Performance against other participants' bots (unknown field). The opponent pool contains the 4
  supplied baselines and 5 independently written heuristics; strong opponents may be closer to
  equilibrium than anything here, in which case expect chops.
* The Docker build (`tools/check_build.sh`) — Docker is unavailable on the dev machine.
* Contended-CPU timing: with 12 training threads running, every bot (including trivial ones) blew
  through the time bank; the tournament says it limits CPU to ~1 core per bot, which was not
  reproduced here.

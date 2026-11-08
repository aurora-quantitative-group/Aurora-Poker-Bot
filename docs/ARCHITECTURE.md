# PLO5-MCCFR: architecture and algorithms

Heads-up Short Deck 5-card Pot-Limit Omaha agent. One sentence: an **exact-betting-tree
MCCFR blueprint over a card abstraction**, plus **Restricted-Nash-Response counter-strategies**
that are switched in by a **statistically gated opponent model**, all driven by a
**C hand evaluator** and a **feedback time manager**.

```
            offline (train/)                                   online (plo5bot/, ~0.3 ms/decision)
 ┌──────────────────────────────────────┐        ┌───────────────────────────────────────────────┐
 │ build_preflop.py  preflop equity     │        │ obs.action_history ─► exact tree node (tree.py)│
 │   table, 376,992 hands               │        │ cards ─► features (C, cached per street)       │
 │ gen_dataset.py  4M deals, features   │──data─►│       ─► bucket (abstraction.py)               │
 │   both players, all streets, buckets │        │ blueprint[street][node, bucket] (blueprint.py) │
 │ mccfr.c  ES-MCCFR blueprint (1.8B it)│        │   or counter-strategy k  (exploit.py)          │
 │ mccfr.c  RNR mode x 7 archetypes     │        │ seeded sampling ─► legal action  (player.py)   │
 │ eval_abstract.py / train_rnr.py eval │        │ on_hand_end ─► opponent model, k selection     │
 └──────────────────────────────────────┘        └───────────────────────────────────────────────┘
```

## 1. Game facts that shaped the design

* Only four actions (`FOLD`, `CHECK_CALL`, `RAISE_HALF`, `RAISE_POT`), at most four raises per
  street, 40 bb stacks. The complete betting tree has **15,952 decision nodes**
  (62 preflop / 810 flop / 3,678 turn / 11,402 river). So unlike NLHE there is **no action
  abstraction and no action translation**: the blueprint is defined on the real betting tree,
  and every opponent action maps to an exact node. Only cards are abstracted.
* Omaha with exactly 2 hole + 3 board cards, short-deck ranking (flush > full house,
  trips > straight, A-6-7-8-9 is the lowest straight). All ranking logic is taken from
  the referee's `evaluator.py`, never re-implemented by hand.
* Scoring is a 95% CI on bb/deal over 1,000 duplicate deals; the paired std is
  ~7-16 bb/deal, so a match is only *won* with an edge of roughly ≥0.5-1 bb/deal. Being
  unexploitable gives chops; **winning needs exploitation** of weaker opponents.
* 8 s of charged time per half (~2,500-4,000 requests). Measured request overhead alone is
  0.3-0.5 ms, so the agent's own work must stay well under 1 ms per decision on average.

## 2. Components

### 2.1 C core: `plo5bot/csrc/plo5core.c` → `plo5bot/libplo5core.so`
* **Additive rank key**: `key = Σ 5^rank` over the 5 cards. Each rank occurs ≤ 4 times, so the key
  is a unique base-5 code of the rank multiset, and `key(hole pair) + key(board triple)` is the key
  of the 5-card hand — no sorting. Two `uint16` tables (non-flush, flush) of 5^9 entries map a key
  to a score **ordinal**; the tables are generated in Python from `evaluator._score_ranks` (the
  referee's function), so ordering is identical by construction. Only ~1,400 entries are ever
  touched, so the working set stays in L2.
* Omaha evaluation = 10 pairs × 10 triples = 100 lookups: **~60 ns per hand** (16M/s).
* `p5_hs_features`: Monte Carlo E[HS], E[HS²] and P(HS ≥ 0.9) of a hand vs a uniformly random
  opponent hand over R runouts × K opponents. `p5_equity_vs_hands`: equity vs a weighted list of
  opponent hands. `p5_gen_deals`: training-data generator.
* Built for generic x86-64 without fortify wrappers; requires only `GLIBC_2.2.5`/`GLIBC_2.14`
  symbols, so it loads in `python:3.12-slim`. `core.py` self-checks it against `evaluator.py` at
  import and falls back to a **vectorised numpy implementation** if loading fails (same results,
  ~2-10× slower per call at the reduced sample sizes the time manager then uses).

### 2.2 Card abstraction: `plo5bot/abstraction.py`
| street | feature(s) vs random hand | buckets |
|---|---|---|
| preflop | exact equity table (all 376,992 hands, 40k runouts per suit-isomorphism class) | 60 equal-mass |
| flop | E[HS], Var[HS] over 50 runouts × 50 opponents | 16 equity × 6 potential = 96 |
| turn | same, 40 × 50 | 96 |
| river | HS vs 600 opponents | 64 equal-mass |

Potential bins are quantiles *within* each equity bin, separating draws from made hands of equal
equity (a cheap stand-in for RoboPoker's EMD clustering over next-street histograms). Training and
play use identical sample sizes so bucket noise is the same in both.

### 2.3 Blueprint: external-sampling MCCFR (`train/mccfr.c`)
* Info set = (street, exact decision node, own bucket on this street): perfect recall of betting,
  imperfect recall of cards — 1.16M info sets × ≤4 actions.
* External sampling (Lanctot et al. 2009), alternating traversers; chance is sampled from a 4M-deal
  dataset of precomputed buckets and showdown results, with random seat assignment.
* **Linear weighting** of regret and average-strategy increments (w = 1 + t / 2M, as in
  Pluribus/RoboPoker "linear CFR"), **regret-based pruning** after 10% of iterations (95% of
  traversals skip actions with regret < −2·10⁵), hogwild pthreads. 1.8B iterations in ~11 min on
  12 threads (up to 2.5M it/s once pruning is active).
* Exported as uint8-quantised average strategy (1.6 MB).

### 2.4 Restricted Nash Response counter-strategies (`mccfr.c` RNR mode, `train/train_rnr.py`)
Johanson, Zinkevich & Bowling (2008). For each archetype, the opponent plays a **fixed model** with
probability p and a **free CFR learner** otherwise; our strategy (shared info sets) learns a
counter-strategy that exploits the model while staying robust if the model is wrong. Models are
tilts of the blueprint, `σ_m[a] ∝ max(0, σ_bp[a]·mult[a] + add[a])`, separately preflop/postflop:

| archetype | p | idea | abstract-game EV vs archetype (bb/hand): blueprint → counter |
|---|---|---|---|
| station | 0.9 | never folds or raises | +1.41 → **+7.88** |
| always_pot / always_half | 0.9 | raises whenever legal | +1.87 → **+4.53** / +2.40 → **+7.49** |
| random | 0.9 | uniform over legal actions | +1.82 → **+7.88** |
| maniac | 0.75 | folds less, raises 2.5× more | +0.31 → +0.89 |
| loose_passive | 0.75 | folds 0.35×, raises 0.5× | +0.04 → +0.07 |
| overfolder | 0.75 | folds 2.5× | +0.08 → +0.12 |

The same traversal also gives each archetype's predicted **action frequencies per situation
class**, which the online model uses for recognition.

### 2.5 Opponent model and selection (`plo5bot/exploit.py`)
Only legitimate information: visible `action_history`, `HandResult`, and shown-down cards. Nothing
crosses halves (all state lives in `memory`).
* Opponent actions are counted in 9 situation classes (preflop open / unopened / facing;
  flop/turn/river × unopened / facing). The opponent's *final* action, which no observation
  shows (e.g. folding to our bet), is inferred at hand end from the last known node,
  `went_to_showdown` and `my_net`.
* Shown-down hands are scored (HS vs random) separately for river-aggressive and passive lines.
* **Selection**, re-evaluated every 10 hands after 40 observed actions:
  1. tempered Bayesian posterior over archetypes, prior 0.4 on "blueprint/unknown";
  2. the top archetype needs posterior ≥ 0.9, **and**
  3. an **absolute goodness-of-fit G-test**: 2G ≤ df + 4√(2df) + 0.05·N. (Lesson learned: a
     relative posterior picks the least-bad model even when none fits — that misclassified an
     aggressive heuristic as "random" and lost 3.9 bb/deal.)
  4. archetypes that win the posterior but fail the fit get **probation** (non-bluffy archetypes
     only): kept only while our results plausibly reach 1 bb/hand (mean + 1 SE ≥ 1 after 30 hands);
  5. "bluffy" counters (random/always_*) are refused if the opponent's river-aggressive showdowns
     average HS ≥ 0.70;
  6. **performance safety net**: any counter-strategy that is losing significantly
     (mean + 2 SE < 0) is banned for the rest of the half.
  Otherwise the blueprint is played.

### 2.6 Runtime player (`plo5bot/player.py`)
1. Replay `action_history` through the exact tree → node (verified against pot/street/actor).
2. Features → bucket, cached per (hand, street); preflop is a table lookup.
3. Strategy row = blueprint or selected counter-strategy; mask illegal actions; sample with a
   per-half `random.Random` (seeded → reproducible mixed strategies).
4. **Feedback time manager**: estimates the charged time as process CPU since `new_half` minus the
   free 1 ms per request; once per hand projects end-of-half usage and steps the Monte Carlo sample
   scale down/up (C: 1.0/0.5/0.25; numpy: 0.16/0.08/0.04) against a 5 s target (3.5 s numpy). Only
   past 6.5 s does it check/fold — the judge would force that at 8 s anyway.
   `PLO5_DETERMINISTIC=1` pins the scale for tests/experiments.
5. Every path is wrapped: an exception, a tree mismatch or an empty strategy row falls back to a
   legal action (heuristic or check/call) and is counted in `memory["_stats"]`; `on_hand_end`
   never raises.

## 3. What was taken from the references

| idea | source | here |
|---|---|---|
| MCCFR with external sampling, linear weighting, regret-based pruning | Pluribus, RoboPoker `mccfr` | **implemented** (`train/mccfr.c`) |
| offline blueprint + compact abstraction | Libratus, Pluribus, RoboPoker | **implemented** (exact betting tree, card buckets) |
| distribution-aware hand clustering (EMD / k-means over next-street histograms) | RoboPoker `lloyd`/`monge` | **approximated** by 2-D equity × potential quantile buckets |
| action abstraction + translation | Libratus, RoboPoker `pokerkit` | **not needed**: the game has 4 actions, tree is exact |
| nested safe subgame solving, depth-limited search | Libratus, Pluribus, RoboPoker `subgame` | **not implemented** (see §4) |
| mixed strategies | all | **implemented** (seeded sampling from average strategies) |
| robust exploitation | Johanson et al. RNR (not in the three references) | **implemented** |
| multiplayer techniques of Pluribus | Pluribus | not applicable (heads-up) |

## 4. Status: implemented, experimental, not implemented

**Implemented and validated** (tests + benchmarks): C evaluator/equity engine with numpy fallback;
preflop table; card abstraction; exact betting tree; ES-MCCFR blueprint; RNR counter-strategies;
opponent model with G-test gate, probation and safety ban; feedback time manager; duplicate-match
arena, benchmark suite, version snapshots, experiment log.

**Experimental / weakly validated**
* *Abstract-game exploitability.* `train/eval_abstract.py` computes a best response inside the
  abstraction. With ~1.16M info sets and 40k sampled deals the in-sample BR mostly fits sampling
  noise (+0.86 bb/hand) while the held-out BR *loses* (−0.44 bb/hand); the metric therefore only
  brackets exploitability and was **not** used to make decisions.
* *Training length.* 1.8B vs 300M iterations head-to-head: +0.06 bb/deal, CI [−0.09, +0.21] —
  no measurable gain beyond 300M; the longer run is kept as it is at least as converged.
* The tilt archetypes `loose_passive`, `overfolder`, `maniac` have small abstract-game gains and no
  real opponent in the pool that they verifiably fit.

**Not implemented (research ideas, honestly unexplored)**
* Real-time safe subgame re-solving on the turn/river (Libratus/Pluribus). Feasible in budget only
  for a few big pots per half with the C evaluator (sampled opponent ranges of ~300 hands); not
  attempted.
* Neural networks (Deep CFR, value nets, NFSP). The tabular blueprint is 1.6 MB and a lookup costs
  microseconds; no experiment showed a need for function approximation, so none was added.
  PyTorch is not imported.
* Showdown-based range inference (Bayesian updating of the opponent's hand distribution from
  their actions) and AIVAT variance reduction.
* A Docker build test: Docker is not installed on the development machine, so
  `tools/check_build.sh` has **not** been run (see README).

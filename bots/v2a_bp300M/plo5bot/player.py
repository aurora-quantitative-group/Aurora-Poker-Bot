"""
plo5bot.player - the runtime decision engine called from agent.py.

Per decision:
  1. replay obs.action_history through the exact betting tree -> node
     (the game has no off-tree actions, so this always succeeds)
  2. compute this street's hand features / bucket (cached per hand+street)
  3. read the MCCFR blueprint strategy for (node, bucket)
  4. let the opponent model adjust it (plo5bot.exploit) when there is
     enough evidence
  5. sample an action with a per-half seeded RNG (reproducible mixing)
Everything is guarded: any exception or an exhausted time budget falls back
to a cheap legal action, so the bot can never crash or time out.

Per-half state lives in `memory` (wiped by the server every half). The
Player object holds only read-only tables loaded at startup.
"""

from __future__ import annotations

import os
import random
import time

from game import Action

from . import abstraction as A
from . import core
from .blueprint import try_load
from .heuristic import heuristic_action
from .tree import get_tree

TIME_BANK = 8.0
# Our own compute target per half (the remainder is margin for slower
# tournament hardware and for request overhead beyond the 1 ms grace).
OWN_BUDGET = float(os.environ.get("PLO5_BUDGET", "4.0"))
EXPECTED_DECISIONS = 2800


class Player:
    def __init__(self, blueprint_path: str | None = None, use_exploit: bool | None = None):
        self.tree = get_tree()
        A.preflop_table()
        self.abs = A.load_abstraction()
        self.bp = try_load(blueprint_path or os.environ.get("PLO5_BLUEPRINT"))
        if use_exploit is None:
            use_exploit = os.environ.get("PLO5_EXPLOIT", "1") == "1"
        self.use_exploit = use_exploit
        self.mode = os.environ.get("PLO5_MODE", "blueprint" if self.bp is not None else "heuristic")
        # warm caches / code paths so the first decision is not slow
        A.features([0, 5, 10, 15, 20], [1, 6, 11])
        A.features([0, 5, 10, 15, 20], [1, 6, 11, 16, 21])
        if self.use_exploit:
            from . import exploit
            self.exploit = exploit
        else:
            self.exploit = None

    # ------------------------------------------------------------------ API
    def new_half(self, config, memory: dict) -> None:
        memory["rng"] = random.Random(int(os.environ.get("PLO5_SEED", "20261107")))
        memory["cache"] = {}            # (hand_index, street) -> (e, e2, bucket)
        memory["hand"] = -1
        memory["num_hands"] = getattr(config, "num_hands", 1000)
        memory["_stats"] = {"decisions": 0, "time": 0.0, "max_time": 0.0, "feature_calls": 0,
                            "cache_hits": 0, "fallbacks": 0, "exceptions": 0, "backend": core.BACKEND,
                            "mode": self.mode, "exploit_adjusted": 0}
        if self.exploit is not None:
            memory["opp"] = self.exploit.new_model()

    def on_hand_end(self, result, memory: dict) -> None:
        if self.exploit is not None and "opp" in memory:
            try:
                self.exploit.observe_result(memory["opp"], result, memory)
            except Exception:
                memory["_stats"]["exceptions"] += 1

    def choose_action(self, obs, memory: dict) -> Action:
        t0 = time.perf_counter()
        st = memory["_stats"]
        try:
            action = self._decide(obs, memory, st)
            if action not in obs.legal_actions:
                st["fallbacks"] += 1
                action = Action.CHECK_CALL
        except Exception:
            st["exceptions"] += 1
            action = Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else obs.legal_actions[0]
        if self.exploit is not None:
            try:
                self.exploit.observe_decision(memory["opp"], obs, self.tree, action)
            except Exception:
                st["exceptions"] += 1
        dt = time.perf_counter() - t0
        st["decisions"] += 1
        st["time"] += dt
        st["max_time"] = max(st["max_time"], dt)
        return action

    # ------------------------------------------------------------- internals
    def _scale(self, memory, st) -> float:
        """Sample-size scale for this decision from the remaining budget."""
        left = OWN_BUDGET - st["time"]
        hands_left = max(1, memory["num_hands"] - memory.get("hand", 0))
        per_dec = left / (hands_left * EXPECTED_DECISIONS / 1000.0)
        if core.BACKEND == "c":
            return 1.0 if per_dec > 0.0008 else (0.3 if per_dec > 0.0002 else 0.0)
        # numpy fallback: ~0.01 ms per opponent-hand evaluation
        if per_dec > 0.02:
            return 0.25
        if per_dec > 0.006:
            return 0.1
        return 0.0

    def _bucket(self, obs, memory, st):
        street = int(obs.street)
        key = (obs.hand_index, street)
        cache = memory["cache"]
        if key in cache:
            st["cache_hits"] += 1
            return cache[key]
        if obs.hand_index != memory.get("hand"):
            cache.clear()
            memory["hand"] = obs.hand_index
        scale = 1.0 if street == 0 else self._scale(memory, st)
        if scale <= 0.0:
            return None
        e, e2 = A.features(obs.hole_cards, obs.board, scale)
        st["feature_calls"] += 1
        b = A.bucket_from_features(street, e, e2, self.abs)
        cache[key] = (e, e2, b)
        return cache[key]

    def _decide(self, obs, memory, st) -> Action:
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        rng = memory["rng"]
        feat = self._bucket(obs, memory, st)
        if feat is None:                          # out of time budget: cheapest sane action
            st["fallbacks"] += 1
            return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        e, e2, bucket = feat
        if self.mode == "heuristic" or self.bp is None:
            return heuristic_action(obs, rng, hs=e)
        node = self.tree.node_from_history([int(h.action) for h in obs.action_history])
        t = self.tree
        my_pos = 0 if obs.is_button else 1
        if t.player[node] != my_pos or t.pot[node] != obs.pot or t.street[node] != int(obs.street):
            st["fallbacks"] += 1
            return heuristic_action(obs, rng, hs=e)
        street = int(obs.street)
        probs = self.bp.probs(street, int(t.local[node]), bucket)
        if self.exploit is not None:
            probs = self.exploit.adjust(probs, obs, node, e, e2, bucket, memory, t, self)
        probs = [probs[a] if Action(a) in legal else 0.0 for a in range(4)]
        tot = sum(probs)
        if tot <= 0:
            st["fallbacks"] += 1
            return heuristic_action(obs, rng, hs=e)
        u = rng.random() * tot
        acc = 0.0
        for a in range(4):
            acc += probs[a]
            if probs[a] > 0 and u < acc:
                return Action(a)
        return Action.CHECK_CALL

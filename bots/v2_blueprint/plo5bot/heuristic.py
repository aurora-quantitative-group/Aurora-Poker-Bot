"""
plo5bot.heuristic - a compact equity / pot-odds policy.

Used (1) as the safety fallback when the blueprint cannot be used, and
(2) as an independently designed sparring opponent (bots/heuristic/).
It is NOT the main strategy.

strength: preflop equity percentile, postflop E[HS] vs a random hand,
discounted for the opponent's aggression this hand (a range that bets and
raises is stronger than a random hand).
"""

from __future__ import annotations

import random

from game import Action, Street

from . import abstraction as A


def opp_aggression(obs) -> int:
    return sum(1 for h in obs.action_history if not h.is_me and h.action in (Action.RAISE_HALF, Action.RAISE_POT))


def heuristic_action(obs, rng: random.Random, hs: float | None = None, params: dict | None = None) -> Action:
    p = {"tight": 0.0, "bluff": 0.08, "aggr": 1.0}
    if params:
        p.update(params)
    legal = obs.legal_actions
    if hs is None:
        hs = A.features(obs.hole_cards, obs.board, scale=0.5)[0]
    if obs.street == Street.PREFLOP:
        # map equity vs random (0.1 .. 0.69, median 0.5) to a rough percentile
        strength = min(1.0, max(0.0, (hs - 0.38) / 0.24))
    else:
        strength = hs
    k = opp_aggression(obs)
    adj = strength ** (1.0 + 0.6 * k) - p["tight"]
    pot_after = obs.pot + obs.to_call
    can_raise = Action.RAISE_POT in legal
    if obs.to_call > 0:
        need = obs.to_call / pot_after
        if can_raise and adj > 0.86 - 0.04 * (obs.street == Street.PREFLOP):
            return Action.RAISE_POT if rng.random() < 0.6 * p["aggr"] else Action.RAISE_HALF
        if adj > need + 0.08 or (obs.street == Street.PREFLOP and obs.to_call <= 1 and adj > 0.25):
            return Action.CHECK_CALL
        return Action.FOLD
    if can_raise:
        if adj > 0.78:
            return Action.RAISE_POT if rng.random() < 0.5 * p["aggr"] else Action.RAISE_HALF
        if adj > 0.62 and rng.random() < 0.5 * p["aggr"]:
            return Action.RAISE_HALF
        if adj < 0.3 and rng.random() < p["bluff"]:
            return Action.RAISE_HALF
    return Action.CHECK_CALL

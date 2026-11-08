"""Shared helpers for the sparring opponents in bots/opponents/ (arena only)."""
import random

from game import Action
from plo5bot import abstraction as A
from plo5bot.heuristic import heuristic_action


def make(params=None, name="Heuristic", caller=False, check_fold=False):
    def new_half(config, memory):
        memory["rng"] = random.Random(12345)

    def choose_action(obs, memory):
        if check_fold:
            return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        if caller:
            hs = A.features(obs.hole_cards, obs.board, 0.5)[0]
            if obs.board == ():
                hs = min(1.0, max(0.0, (hs - 0.38) / 0.24))
            need = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0
            if obs.to_call == 0 or hs > need + 0.05 or (not obs.board and obs.to_call <= 1):
                return Action.CHECK_CALL
            return Action.FOLD
        return heuristic_action(obs, memory["rng"], params=params)

    def on_hand_end(result, memory):
        pass

    return new_half, choose_action, on_hand_end

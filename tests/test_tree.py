"""The betting tree must agree with game.Game on every step of random hands."""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluator as ev  # noqa: E402
from game import Action, Game, make_deal  # noqa: E402
from plo5bot.tree import get_tree  # noqa: E402


def test_tree_size():
    t = get_tree()
    assert t.n_local == [62, 810, 3678, 11402]


def test_tree_matches_game_on_random_hands():
    t = get_tree()
    rng = random.Random(0)
    for hand in range(3000):
        button = hand % 2
        g = Game(make_deal("tree", hand), button, hand)
        node = 0
        while not g.is_terminal:
            seat = g.to_act
            pos = 0 if seat == button else 1
            assert t.kind[node] == 0 and t.player[node] == pos
            assert t.street[node] == int(g.street)
            legal = [int(a) for a in g.legal_actions()]
            assert legal == t.legal(node)
            obs = g.observation(seat)
            assert obs.pot == t.pot[node] and obs.to_call == t.to_call[node]
            weights = [1, 3, 2, 2] if rng.random() < 0.5 else [1, 1, 1, 1]
            a = rng.choices(legal, [weights[x] for x in legal])[0]
            if a >= 2:
                assert g.raise_to(Action(a)) == obs.raise_to[Action(a)]
            g.apply(Action(a))
            node = int(t.child[node, a])
        res = g.result()
        assert t.kind[node] == (2 if res.went_to_showdown else 1)
        s_btn = ev.omaha_score(g.deal.hole_cards[button], g.deal.board)
        s_bb = ev.omaha_score(g.deal.hole_cards[1 - button], g.deal.board)
        sd = (s_btn > s_bb) - (s_btn < s_bb)
        assert t.payoff0(node, sd) == res.nets[button]


def test_node_from_observation_history():
    t = get_tree()
    g = Game(make_deal("x", 3), 0, 3)
    acts = []
    for a in (Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL, Action.CHECK_CALL):
        acts.append(int(a))
        g.apply(a)
    obs = g.observation(g.to_act)
    hist = [int(h.action) for h in obs.action_history]
    assert hist == acts
    node = t.node_from_history(hist)
    assert t.street[node] == 1 and t.pot[node] == obs.pot

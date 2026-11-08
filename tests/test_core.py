"""Correctness tests for plo5bot.core against the referee's evaluator.py."""

import os
import random
import sys
from itertools import combinations

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluator as ev  # noqa: E402
from plo5bot import core  # noqa: E402


def _ordinal(score):
    return core.ORDINAL[score]


def test_ordinals_preserve_order():
    scores = sorted(core.ORDINAL)
    assert [core.ORDINAL[s] for s in scores] == list(range(1, len(scores) + 1))


def test_all_five_card_hands_exhaustive_numpy():
    """Every one of the C(36,5) = 376,992 five-card hands."""
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    key = core.POW5[combos >> 2].sum(axis=1)
    suits = combos & 3
    flush = (suits == suits[:, :1]).all(axis=1)
    ours = np.where(flush, core.FL[key], core.NF[key])
    table = ev.get_table()
    masks = (np.int64(1) << combos).sum(axis=1)
    ref = np.array([core.ORDINAL[table[int(m)]] for m in masks])
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(core.BACKEND != "c", reason="C backend not available")
def test_all_five_card_hands_exhaustive_c():
    table = ev.get_table()
    for combo in combinations(range(36), 5):
        m = (1 << combo[0]) | (1 << combo[1]) | (1 << combo[2]) | (1 << combo[3]) | (1 << combo[4])
        assert core.five(combo) == core.ORDINAL[table[m]]


def _random_hands(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        cards = rng.sample(range(36), 10)
        out.append((cards[:5], cards[5:]))
    return out


def test_omaha_random_vs_referee():
    hands = _random_hands(30000, 1)
    holes = np.array([h for h, _ in hands])
    boards = np.array([b for _, b in hands])
    np_scores = core.omaha_np(holes, boards)
    for (h, b), s in zip(hands, np_scores):
        ref = _ordinal(ev.omaha_score(h, b))
        assert s == ref
        assert core.omaha(h, b) == ref


def test_omaha_requires_exactly_two_hole_cards():
    # Four aces in hand on a 6-high board: only a PAIR of aces plays (not quads).
    hole = ev.cards_from_str("Ac Ad Ah As 7c")
    board = ev.cards_from_str("6d 8h Ts Qc Kd")
    s = core.omaha(hole, board)
    assert ev.category_of(ev.omaha_score(hole, board)) == ev.ONE_PAIR
    assert s == _ordinal(ev.omaha_score(hole, board))
    # Four-flush on board + one suited hole card is NOT a flush in Omaha.
    hole = ev.cards_from_str("Ah 6c 7d 8s 9c")
    board = ev.cards_from_str("Kh Qh Jh Th 6d")
    assert ev.category_of(ev.omaha_score(hole, board)) != ev.FLUSH
    assert core.omaha(hole, board) == _ordinal(ev.omaha_score(hole, board))


def test_short_deck_category_order():
    flush = ev.cards_from_str("Ah Kh 9h 7h 6h")
    full_house = ev.cards_from_str("Ac Ad As Kc Kd")
    trips = ev.cards_from_str("Qc Qd Qs 7c 6d")
    straight = ev.cards_from_str("Ac Kd Qs Jc Td")
    wheel = ev.cards_from_str("Ac 6d 7s 8c 9d")
    six_high = ev.cards_from_str("6c 7d 8s 9c Td")
    assert core.five(flush) > core.five(full_house)
    assert core.five(trips) > core.five(straight)
    assert core.five(six_high) > core.five(wheel)       # A-6-7-8-9 is the lowest straight


def test_ties():
    a = ev.cards_from_str("Ac Kd 9s 7c 6d")
    b = ev.cards_from_str("Ad Kc 9h 7s 6c")
    assert core.five(a) == core.five(b)


def test_hs_features_river_is_exact_fraction():
    # On the river with the nut straight flush the hand beats every opponent hand.
    hole = ev.cards_from_str("Ah Kh 6c 7d 8s")
    board = ev.cards_from_str("Qh Jh Th 6d 9c")
    e, e2, nut = core.hs_features(hole, board, 1, 300, seed=3)
    assert e == 1.0 and e2 == 1.0 and nut == 1.0


def test_hs_features_numpy_matches_c_statistically():
    hole = ev.cards_from_str("Ah Kh Qd Jd 9s")
    board = ev.cards_from_str("Th 8h 6c")
    # R=2000 runouts: the standard error of the difference is ~0.008, so 0.03 is ~4 sigma
    a = core.hs_features(hole, board, 2000, 20, seed=1)
    b = core._hs_features_np(hole, board, 2000, 20, seed=2, dead=())
    assert abs(a[0] - b[0]) < 0.03
    assert abs(a[1] - b[1]) < 0.03


def test_equity_vs_hands_symmetry():
    rng = random.Random(5)
    cards = rng.sample(range(36), 13)
    h1, h2, board = cards[:5], cards[5:10], cards[10:13]
    e1, _ = core.equity_vs_hands(h1, board, np.array([h2]), np.array([1.0]), 4000, seed=1)
    e2, _ = core.equity_vs_hands(h2, board, np.array([h1]), np.array([1.0]), 4000, seed=2)
    assert abs(e1 + e2 - 1.0) < 0.04


def test_equity_vs_hands_river_exact():
    rng = random.Random(9)
    for _ in range(200):
        cards = rng.sample(range(36), 15)
        h1, h2, board = cards[:5], cards[5:10], cards[10:]
        eq, per = core.equity_vs_hands(h1, board, np.array([h2]), np.array([1.0]), 1)
        s1, s2 = ev.omaha_score(h1, board), ev.omaha_score(h2, board)
        assert eq == (1.0 if s1 > s2 else 0.5 if s1 == s2 else 0.0)


def test_equity_blocked_hands_excluded():
    hole = [0, 1, 2, 3, 4]
    opp = np.array([[4, 5, 6, 7, 8], [10, 11, 12, 13, 14]])
    eq, per = core.equity_vs_hands(hole, [20, 21, 22], opp, np.array([1.0, 1.0]), 50)
    assert per[0] == -1 and 0 <= per[1] <= 1

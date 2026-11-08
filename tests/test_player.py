"""Behavioural tests for the runtime engine (plo5bot.player / exploit)."""

import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import Action, AgentConfig, Game, HandResult, make_deal  # noqa: E402
from plo5bot import exploit  # noqa: E402
from plo5bot.player import Player  # noqa: E402
from plo5bot.tree import get_tree  # noqa: E402

PLAYER = Player()


def _play(player, n_hands, seed, opp_raise=0.3):
    """Our player in seat 0 vs a random-ish opponent. Returns our actions."""
    mem = {}
    player.new_half(AgentConfig(), mem)
    rng = random.Random(seed)
    out = []
    mem["net"] = 0
    for h in range(n_hands):
        g = Game(make_deal(seed, h), h % 2, h)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            if g.to_act == 0:
                a = player.choose_action(obs, mem)
                assert a in obs.legal_actions
                out.append(int(a))
            else:
                raises = [x for x in obs.legal_actions if x >= 2]
                a = rng.choice(raises) if raises and rng.random() < opp_raise else Action.CHECK_CALL
            g.apply(a)
        player.on_hand_end(g.hand_result(0), mem)
        mem["net"] += g.result().nets[0]
    return out, mem


def test_reproducible_mixing(monkeypatch):
    # Mixing is seeded (per-half RNG + card-derived feature seeds). Only the
    # time manager depends on wall-clock, so pin it for this test.
    import plo5bot.player as pm
    monkeypatch.setattr(pm, "DETERMINISTIC", True)
    a, _ = _play(PLAYER, 60, "repro")
    b, _ = _play(PLAYER, 60, "repro")
    assert a == b


def test_no_exceptions_or_fallbacks():
    _, mem = _play(PLAYER, 150, "clean", opp_raise=0.6)
    st = mem["_stats"]
    assert st["exceptions"] == 0
    assert st["fallbacks"] == 0


def test_decision_time_is_small():
    _, mem = _play(PLAYER, 100, "speed")
    st = mem["_stats"]
    assert st["time"] / st["decisions"] < 0.004       # average well under the ~3 ms budget
    assert st["max_time"] < 0.2


def test_infers_opponent_fold_to_our_bet():
    t = get_tree()
    model = exploit.new_model()
    # Hand 5: we are the button; we open with a pot raise and the opponent folds.
    g = Game(make_deal("inf", 5), 0, 5)
    obs = g.observation(0)
    exploit.observe_decision(model, obs, t, Action.RAISE_POT)
    g.apply(Action.RAISE_POT)
    g.apply(Action.FOLD)
    exploit.observe_result(model, g.hand_result(0), {})
    cls = exploit.situation_class(t, int(t.child[0, int(Action.RAISE_POT)]))
    assert cls == 2 and model["counts"][cls, int(Action.FOLD)] == 1


def test_infers_opponent_preflop_fold_without_our_decision():
    model = exploit.new_model()
    res = HandResult(hand_index=3, my_net=1, pot=3, went_to_showdown=False, board=(), opp_hole=None,
                     winner="me")
    exploit.observe_result(model, res, {})
    assert model["counts"][0, int(Action.FOLD)] == 1


def test_numpy_fallback_backend_runs():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from plo5bot import core; assert core.BACKEND == 'numpy'\n"
        "from tests.test_player import _play\n"
        "from plo5bot.player import Player\n"
        "out, mem = _play(Player(), 400, 'np', opp_raise=0.5)\n"
        "st = mem['_stats']\n"
        "assert st['exceptions'] == 0, st\n"
        "assert st['fallbacks'] == 0, st   # must never degrade to check/fold within budget\n"
        "print(mem['net'])\n"
        "print(st['time'] / st['decisions'])\n" % ROOT
    )
    env = dict(os.environ, PLO5_FORCE_NUMPY="1", PLO5_DETERMINISTIC="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert float(lines[-1]) < 0.01      # average seconds per decision
    # quality, not just liveness: 400 hands vs a never-folding random raiser
    # (measured: +1.9..+2.9 bb/hand for both backends; 150 hands was too noisy)
    assert float(lines[-2]) / 2 / 400 > 0.5


def test_startup_is_fast():
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", "import agent"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 10

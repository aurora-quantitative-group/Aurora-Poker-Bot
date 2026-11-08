"""
plo5bot.tree - the EXACT heads-up Short Deck PLO5 betting tree.

The game has only four actions and at most four raises per street, so the
whole betting tree (without cards) is small: 15,952 decision nodes. We use it
unabstracted - no action abstraction and no action translation are needed.

Players are POSITIONS, not seats: player 0 = button (small blind, acts first
preflop, last postflop), player 1 = big blind. The rules below mirror
game.Game exactly (tests/test_tree.py replays random games through both).

Node arrays (index = node id, root = 0):
    player[n]     0/1 for decision nodes, -1 terminal
    street[n]     0..3 (street of the decision / where the hand ended)
    child[n, a]   node reached by action a (FOLD, CHECK_CALL, RAISE_HALF,
                  RAISE_POT), -1 if illegal
    kind[n]       0 decision, 1 fold terminal, 2 showdown terminal
    contrib[n,p]  chips player p has put in (whole hand) at this node
    pot[n], to_call[n], stack[n,p], raises[n]
    local[n]      index of the decision node among decision nodes of its street
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SB, BB, STACK, CAP = 1, 2, 80, 4
FOLD, CALL, HALF, POT = 0, 1, 2, 3


@dataclass
class _S:
    street: int
    bets: list
    contrib: list
    stacks: list
    acted: list
    raises: int
    last_inc: int
    to_act: int

    def copy(self) -> "_S":
        return _S(self.street, self.bets[:], self.contrib[:], self.stacks[:], self.acted[:],
                  self.raises, self.last_inc, self.to_act)


def _to_call(s: _S, p: int) -> int:
    return max(0, s.bets[1 - p] - s.bets[p])


def _legal(s: _S) -> list[int]:
    p = s.to_act
    acts = []
    if _to_call(s, p) > 0:
        acts.append(FOLD)
    acts.append(CALL)
    if s.raises < CAP and s.stacks[p] > _to_call(s, p) and s.stacks[1 - p] > 0:
        acts += [HALF, POT]
    return acts


def raise_to(s: _S, a: int) -> int:
    p = s.to_act
    my_bet, opp_bet = s.bets[p], s.bets[1 - p]
    tc = _to_call(s, p)
    pot = s.contrib[0] + s.contrib[1]
    all_in_to = my_bet + s.stacks[p]
    pot_to = opp_bet + pot + tc
    half_to = opp_bet + (pot + tc) // 2
    min_to = opp_bet + max(s.last_inc, BB)
    target = half_to if a == HALF else pot_to
    return min(max(target, min_to), pot_to, all_in_to)


def _apply(s: _S, a: int):
    """Return (new_state, terminal_kind) with terminal_kind 0 = continue,
    1 = fold, 2 = showdown."""
    s = s.copy()
    p = s.to_act
    if a == FOLD:
        return s, 1
    if a == CALL:
        amt = min(_to_call(s, p), s.stacks[p])
    else:
        new_total = raise_to(s, a)
        amt = new_total - s.bets[p]
        s.last_inc = new_total - s.bets[1 - p]
        s.raises += 1
        s.acted[1 - p] = False
    s.stacks[p] -= amt
    s.bets[p] += amt
    s.contrib[p] += amt
    s.acted[p] = True
    if not (s.acted[0] and s.acted[1] and s.bets[0] == s.bets[1]):
        s.to_act = 1 - p
        return s, 0
    if s.street == 3 or 0 in s.stacks:
        return s, 2
    s.street += 1
    s.bets = [0, 0]
    s.acted = [False, False]
    s.raises = 0
    s.last_inc = BB
    s.to_act = 1   # big blind (non-button) acts first after the flop
    return s, 0


def root_state() -> _S:
    return _S(0, [SB, BB], [SB, BB], [STACK - SB, STACK - BB], [False, False], 0, BB, 0)


class Tree:
    """The full betting tree as flat numpy arrays (see module docstring)."""

    def __init__(self) -> None:
        player, street, kind, contrib, pot, to_call, stack, raises = [], [], [], [], [], [], [], []
        children: list[list[int]] = []
        states: list[_S] = []

        def add(s: _S, k: int, who: int) -> int:
            nid = len(player)
            player.append(who if k == 0 else -1)
            street.append(s.street)
            kind.append(k)
            contrib.append(tuple(s.contrib))
            pot.append(s.contrib[0] + s.contrib[1])
            to_call.append(_to_call(s, s.to_act) if k == 0 else 0)
            stack.append(tuple(s.stacks))
            raises.append(s.raises)
            children.append([-1, -1, -1, -1])
            states.append(s)
            return nid

        def build(s: _S, k: int) -> int:
            nid = add(s, k, s.to_act)
            if k == 0:
                for a in _legal(s):
                    ns, nk = _apply(s, a)
                    if a == FOLD:
                        ns.to_act = s.to_act        # remember who folded
                    children[nid][a] = build(ns, nk)
            return nid

        build(root_state(), 0)
        self.player = np.array(player, dtype=np.int8)
        self.street = np.array(street, dtype=np.int8)
        self.kind = np.array(kind, dtype=np.int8)
        self.child = np.array(children, dtype=np.int32)
        self.contrib = np.array(contrib, dtype=np.int32)
        self.pot = np.array(pot, dtype=np.int32)
        self.to_call = np.array(to_call, dtype=np.int32)
        self.stack = np.array(stack, dtype=np.int32)
        self.raises = np.array(raises, dtype=np.int8)
        self.states = states
        # who folded at fold terminals
        self.folder = np.array([st.to_act if k == 1 else -1 for st, k in zip(states, kind)], dtype=np.int8)
        n = len(player)
        self.n = n
        self.local = np.full(n, -1, dtype=np.int32)
        self.n_local = [0, 0, 0, 0]
        for i in range(n):
            if self.kind[i] == 0:
                s = int(self.street[i])
                self.local[i] = self.n_local[s]
                self.n_local[s] += 1

    def legal(self, node: int) -> list[int]:
        return [a for a in range(4) if self.child[node, a] >= 0]

    def payoff0(self, node: int, showdown_result: int) -> float:
        """Chips won by player 0 (button) at a terminal. showdown_result is +1
        if player 0's hand wins, -1 if it loses, 0 for a tie."""
        if self.kind[node] == 1:
            f = int(self.folder[node])
            return -float(self.contrib[node, 0]) if f == 0 else float(self.contrib[node, 1])
        return float(showdown_result * self.contrib[node, 1 if showdown_result > 0 else 0])

    def node_from_history(self, actions) -> int:
        """Follow a sequence of action ints from the root. Raises KeyError
        if an action is illegal in the tree (should never happen)."""
        n = 0
        for a in actions:
            c = int(self.child[n, int(a)])
            if c < 0:
                raise KeyError(f"illegal action {a} at node {n}")
            n = c
        return n


_TREE: Tree | None = None


def get_tree() -> Tree:
    global _TREE
    if _TREE is None:
        _TREE = Tree()
    return _TREE

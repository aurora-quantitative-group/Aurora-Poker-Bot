"""
plo5bot.abstraction - card abstraction shared by offline training and play.

Features (all against a uniformly random opponent hand, see core.hs_features):
    preflop  equity over a random 5-card runout (exact table, 376,992 hands)
    flop     E[HS], E[HS^2] over FLOP_R runouts x FLOP_K opponent hands
    turn     E[HS], E[HS^2] over TURN_R x TURN_K
    river    HS against RIVER_K opponent hands

Buckets:
    preflop  NB_PRE equal-mass bins of equity
    flop     NB_EQ equity bins x NB_POT potential bins (potential =
             Var[HS] = E[HS^2]-E[HS]^2, binned by quantiles WITHIN each equity
             bin, so drawing hands and made hands of equal equity separate)
    turn     same as flop
    river    NB_RIV equal-mass bins of HS
Bin edges are quantiles of the training distribution (train/gen_dataset.py)
and live in plo5bot/data/abstraction.npz. Training and play use the SAME
sample sizes so the noise in bucket assignment is the same in both.
"""

from __future__ import annotations

import os
from bisect import bisect_right
from math import comb

import numpy as np

from . import core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FLOP_R, FLOP_K = 50, 50
TURN_R, TURN_K = 40, 50
RIVER_K = 600

NB_PRE = 60
NB_EQ, NB_POT = 16, 6
NB_RIV = 64
NB = (NB_PRE, NB_EQ * NB_POT, NB_EQ * NB_POT, NB_RIV)

_C = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)
_CL = [[comb(n, k) for k in range(6)] for n in range(37)]


def colex_index_np(sorted_hands: np.ndarray) -> np.ndarray:
    """Colex index of ascending 5-card hands (N,5) -> (N,), range 0..376,991."""
    h = sorted_hands
    return _C[h[:, 0], 1] + _C[h[:, 1], 2] + _C[h[:, 2], 3] + _C[h[:, 3], 4] + _C[h[:, 4], 5]


def colex_index(hand) -> int:
    a, b, c, d, e = sorted(hand)
    return _CL[a][1] + _CL[b][2] + _CL[c][3] + _CL[d][4] + _CL[e][5]


_PRE: np.ndarray | None = None
_ABS: dict | None = None


def preflop_table() -> np.ndarray:
    global _PRE
    if _PRE is None:
        _PRE = np.load(os.path.join(DATA, "preflop_eq.npy")).astype(np.float32) / 65535.0
    return _PRE


def preflop_equity(hand) -> float:
    return float(preflop_table()[colex_index(hand)])


def load_abstraction(path: str | None = None) -> dict:
    global _ABS
    if _ABS is None or path is not None:
        z = np.load(path or os.path.join(DATA, "abstraction.npz"))
        _ABS = {k: z[k] for k in z.files}
    return _ABS


def card_seed(hole, board) -> int:
    """Deterministic per-situation seed (same cards -> same features)."""
    s = 1469598103934665603
    for c in list(hole) + [99] + list(board):
        s = ((s ^ (c + 1)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return s


def features(hole, board, scale: float = 1.0) -> tuple[float, float]:
    """(E[HS], E[HS^2]) for the current street. scale < 1 reduces samples
    (used by the time manager / numpy fallback)."""
    nb = len(board)
    seed = card_seed(hole, board)
    if nb == 0:
        e = preflop_equity(hole)
        return e, e * e
    if nb == 3:
        R, K = max(4, int(FLOP_R * scale)), max(4, int(FLOP_K * scale))
    elif nb == 4:
        R, K = max(4, int(TURN_R * scale)), max(4, int(TURN_K * scale))
    else:
        R, K = 1, max(20, int(RIVER_K * scale))
    e, e2, _ = core.hs_features(hole, board, R, K, seed)
    return e, e2


_EDGE_LISTS: dict = {}


def _edge_lists(a: dict) -> dict:
    """The bin edges as plain Python float lists (exact copies of the float64
    arrays): bisect on a short list is ~10x cheaper than np.searchsorted."""
    lists = _EDGE_LISTS.get(id(a))
    if lists is None:
        lists = {k: (v.tolist() if v.ndim == 1 else [row.tolist() for row in v])
                 for k, v in a.items() if k.endswith("_edges")}
        _EDGE_LISTS[id(a)] = lists
    return lists


def bucket_from_features(street: int, e: float, e2: float, abs_: dict | None = None) -> int:
    # bisect_right(list, x) == np.searchsorted(array, x, side="right")
    a = _edge_lists(abs_ or load_abstraction())
    if street == 0:
        return bisect_right(a["pre_edges"], e)
    if street == 3:
        return bisect_right(a["riv_edges"], e)
    key = "flop" if street == 1 else "turn"
    eb = bisect_right(a[f"{key}_eq_edges"], e)
    var = e2 - e * e
    pb = bisect_right(a[f"{key}_pot_edges"][eb], var)
    return eb * NB_POT + pb


def buckets_from_features_np(street: int, e: np.ndarray, e2: np.ndarray, a: dict) -> np.ndarray:
    """Vectorised bucket_from_features (training)."""
    if street == 0:
        return np.searchsorted(a["pre_edges"], e, side="right")
    if street == 3:
        return np.searchsorted(a["riv_edges"], e, side="right")
    key = "flop" if street == 1 else "turn"
    eb = np.searchsorted(a[f"{key}_eq_edges"], e, side="right")
    var = e2 - e * e
    pb = np.empty_like(eb)
    edges = a[f"{key}_pot_edges"]
    for b in range(NB_EQ):
        m = eb == b
        pb[m] = np.searchsorted(edges[b], var[m], side="right")
    return eb * NB_POT + pb


def quantile_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.asarray(qs, dtype=np.float64)

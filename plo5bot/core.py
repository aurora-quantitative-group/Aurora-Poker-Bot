"""
plo5bot.core - fast Short Deck PLO5 hand evaluation and Monte Carlo features.

Two backends with the same results:
  * "c":     plo5bot/libplo5core.so (built from csrc/plo5core.c), via ctypes
  * "numpy": a vectorised pure numpy fallback, used automatically if the shared
             library cannot be loaded (wrong platform, missing file, ...).

Scores are ORDINALS of evaluator.py's scores: higher is better, equal ties.
The ordinal tables are generated here from evaluator._score_ranks, the exact
function the referee uses, so the ordering can never drift from the rules.

Additive rank key: key(cards) = sum(5 ** rank). Ranks repeat at most 4 times,
so the key is a unique base-5 encoding of the rank multiset, and
key(hole pair) + key(board triple) = key(5-card hand) without sorting.
"""

from __future__ import annotations

import ctypes
import os
from itertools import combinations, combinations_with_replacement

import numpy as np

import evaluator as _ev

HERE = os.path.dirname(os.path.abspath(__file__))
NKEYS = 5 ** 9
POW5 = np.array([5 ** r for r in range(9)], dtype=np.int64)
PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPS = np.array(list(combinations(range(5), 3)), dtype=np.int64)    # (10, 3)


# ---------------------------------------------------------------------------
# Ordinal tables from the referee's own scoring function
# ---------------------------------------------------------------------------

def build_tables() -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return (NF, FL, ordinal_of_score).

    NF[key]: ordinal for a non-flush 5-card rank multiset.
    FL[key]: ordinal for a flush (5 distinct ranks, one suit).
    """
    nf_scores: dict[int, int] = {}
    fl_scores: dict[int, int] = {}
    for ranks in combinations_with_replacement(range(9), 5):
        if max(ranks.count(r) for r in ranks) > 4:
            continue
        key = sum(5 ** r for r in ranks)
        desc = tuple(sorted(ranks, reverse=True))
        nf_scores[key] = _ev._score_ranks(desc, False)
        if len(set(ranks)) == 5:
            fl_scores[key] = _ev._score_ranks(desc, True)
    distinct = sorted(set(nf_scores.values()) | set(fl_scores.values()))
    ordinal = {s: i + 1 for i, s in enumerate(distinct)}   # 0 = "no hand"
    nf = np.zeros(NKEYS, dtype=np.uint16)
    fl = np.zeros(NKEYS, dtype=np.uint16)
    for k, s in nf_scores.items():
        nf[k] = ordinal[s]
    for k, s in fl_scores.items():
        fl[k] = ordinal[s]
    return nf, fl, ordinal


NF, FL, ORDINAL = build_tables()
NUM_ORDINALS = len(ORDINAL)


# ---------------------------------------------------------------------------
# C backend
# ---------------------------------------------------------------------------

_lib = None
BACKEND = "numpy"
_I5 = ctypes.c_int * 5
_D3 = ctypes.c_double * 3


def _load_c() -> None:
    global _lib, BACKEND
    if os.environ.get("PLO5_FORCE_NUMPY"):
        return
    path = os.path.join(HERE, "libplo5core.so")
    if not os.path.isfile(path):
        return
    try:
        lib = ctypes.CDLL(path)
        lib.p5_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.p5_omaha.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.p5_omaha.restype = ctypes.c_int
        lib.p5_five.argtypes = [ctypes.c_void_p]
        lib.p5_five.restype = ctypes.c_int
        lib.p5_omaha_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.p5_hs_features.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                       ctypes.c_void_p]
        lib.p5_equity_vs_hands.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                           ctypes.c_void_p]
        lib.p5_equity_vs_hands.restype = ctypes.c_double
        lib.p5_gen_deals.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p]
        lib.p5_preflop_equity.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                          ctypes.c_void_p]
        lib.p5_init(NF.ctypes.data, FL.ctypes.data)
        # Self-check against the referee on a fixed hand before trusting it.
        hole, board = [0, 5, 10, 15, 20], [24, 29, 34, 3, 9]
        want = ORDINAL[_ev.omaha_score(hole, board)]
        if lib.p5_omaha(_I5(*hole), _I5(*board)) != want:
            return
        _lib = lib
        BACKEND = "c"
    except OSError:
        _lib = None


_load_c()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _keys_suits_pairs(holes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """holes (..., 5) -> pair keys (..., 10) and pair suits (..., 10) (-1 offsuit)."""
    r = holes >> 2
    s = holes & 3
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    key = POW5[r[..., a]] + POW5[r[..., b]]
    suit = np.where(s[..., a] == s[..., b], s[..., a], -1)
    return key, suit


def _keys_suits_trips(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """boards (..., 5) -> triple keys (..., 10) and suits (..., 10) (-1 not mono)."""
    r = boards >> 2
    s = boards & 3
    a, b, c = TRIPS[:, 0], TRIPS[:, 1], TRIPS[:, 2]
    key = POW5[r[..., a]] + POW5[r[..., b]] + POW5[r[..., c]]
    mono = (s[..., a] == s[..., b]) & (s[..., b] == s[..., c])
    suit = np.where(mono, s[..., a], -1)
    return key, suit


def omaha_np(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Vectorised Omaha ordinals: holes (N,5), boards (N,5) or (5,) -> (N,)."""
    holes = np.asarray(holes, dtype=np.int64)
    boards = np.asarray(boards, dtype=np.int64)
    pk, ps = _keys_suits_pairs(holes)           # (N,10)
    tk, ts = _keys_suits_trips(boards)          # (N,10) or (10,)
    if tk.ndim == 1:
        tk = tk[None, :]
        ts = ts[None, :]
    k = pk[:, :, None] + tk[:, None, :]          # (N,10,10)
    flush = (ps[:, :, None] >= 0) & (ps[:, :, None] == ts[:, None, :])
    v = np.where(flush, FL[k], NF[k])
    return v.reshape(len(holes), -1).max(axis=1)


def omaha(hole, board) -> int:
    """Best Omaha score ordinal (2 hole + 3 board) for one hand."""
    if _lib is not None:
        return _lib.p5_omaha(_I5(*hole), _I5(*board))
    return int(omaha_np(np.asarray([hole]), np.asarray(board))[0])


def five(cards) -> int:
    """Score ordinal of exactly five cards."""
    if _lib is not None:
        return _lib.p5_five(_I5(*cards))
    c = np.asarray(cards)
    key = int(POW5[c >> 2].sum())
    return int(FL[key] if len(set((c & 3).tolist())) == 1 else NF[key])


def hs_features(hole, board, R: int, K: int, seed: int = 0, dead=()) -> tuple[float, float, float]:
    """(E[HS], E[HS^2], P(HS>=0.9)) against a uniformly random opponent hand,
    over R runouts with K opponent hands each (see plo5core.c)."""
    nb = len(board)
    if _lib is not None:
        out = _D3()
        bd = _I5(*board) if nb else None
        dd = (ctypes.c_int * len(dead))(*dead) if dead else None
        _lib.p5_hs_features(_I5(*hole), bd, nb, dd, len(dead), R, K, seed & 0xFFFFFFFFFFFFFFFF, out)
        return out[0], out[1], out[2]
    return _hs_features_np(hole, board, R, K, seed, dead)


def _hs_features_np(hole, board, R, K, seed, dead=()):
    """Vectorised numpy version of p5_hs_features: all R runouts and R*K
    opponent hands are sampled and scored in two omaha_np calls."""
    rng = np.random.default_rng(seed)
    used = set(hole) | set(board) | set(dead)
    rem = np.array([c for c in range(36) if c not in used], dtype=np.int64)
    need = 5 - len(board)
    if need == 0:
        R = 1
    # per runout: a random permutation of the remaining cards; the first
    # `need` complete the board, the next 5*K (sampled per opponent) are hands
    keys = rng.random((R, len(rem)))
    perm = np.argsort(keys, axis=1)
    run = rem[perm[:, :need]]                                   # (R, need)
    boards = np.concatenate([np.broadcast_to(np.asarray(board, dtype=np.int64), (R, len(board))), run], axis=1)
    pool = rem[perm[:, need:]]                                  # (R, m)
    m = pool.shape[1]
    okeys = rng.random((R, K, m))
    oidx = np.argsort(okeys, axis=2)[:, :, :5]                  # (R, K, 5) distinct per opponent
    opp = np.take_along_axis(np.broadcast_to(pool[:, None, :], (R, K, m)), oidx, axis=2)
    me = np.broadcast_to(np.asarray(hole, dtype=np.int64), (R, 5))
    mine = omaha_np(me, boards)                                 # (R,)
    theirs = omaha_np(opp.reshape(R * K, 5), np.repeat(boards, K, axis=0)).reshape(R, K)
    hs = ((mine[:, None] > theirs) + 0.5 * (mine[:, None] == theirs)).mean(axis=1)
    return float(hs.mean()), float((hs * hs).mean()), float((hs >= 0.9).mean())


def equity_vs_hands(hole, board, opp: np.ndarray, weights: np.ndarray, R: int, seed: int = 0):
    """Weighted equity of hole vs opponent hands opp (n,5) on a partial board.
    Returns (equity, per_hand array with -1 for blocked hands)."""
    opp = np.ascontiguousarray(opp, dtype=np.int32)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    n = len(opp)
    per = np.empty(n, dtype=np.float64)
    nb = len(board)
    if _lib is not None:
        bd = _I5(*board) if nb else None
        eq = _lib.p5_equity_vs_hands(_I5(*hole), bd, nb, opp.ctypes.data, w.ctypes.data, n, R,
                                     seed & 0xFFFFFFFFFFFFFFFF, per.ctypes.data)
        return eq, per
    # numpy fallback (slower): sample R runouts shared across opponent hands
    rng = np.random.default_rng(seed)
    used = set(hole) | set(board)
    blocked = np.array([bool(set(h.tolist()) & used) for h in opp]) | (w <= 0)
    need = 5 - nb
    acc = np.zeros(n)
    reps = 1 if need == 0 else R
    me = np.asarray(hole, dtype=np.int64)[None, :]
    for _ in range(reps):
        full = list(board)
        if need:
            avail = [c for c in range(36) if c not in used]
            full += rng.choice(avail, need, replace=False).tolist()
        full_a = np.asarray(full)
        mine = omaha_np(me, full_a)[0]
        theirs = omaha_np(opp.astype(np.int64), full_a)
        clash = np.isin(opp, full_a[nb:]).any(axis=1) if need else np.zeros(n, bool)
        res = np.where(mine > theirs, 1.0, np.where(mine == theirs, 0.5, 0.0))
        res[clash] = 0.5
        acc += res
    per[:] = acc / reps
    per[blocked] = -1
    ok = ~blocked
    eq = float((w[ok] * per[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else 0.5
    return eq, per


def gen_deals(n: int, seed: int, Rf: int, Kf: int, Rt: int, Kt: int, Kr: int):
    """Offline training data (C backend only). See p5_gen_deals."""
    if _lib is None:
        raise RuntimeError("gen_deals needs the C backend")
    cards = np.empty((n, 15), dtype=np.uint8)
    feats = np.empty((n, 2, 7), dtype=np.float32)
    res = np.empty(n, dtype=np.int8)
    _lib.p5_gen_deals(n, seed, Rf, Kf, Rt, Kt, Kr, cards.ctypes.data, feats.ctypes.data, res.ctypes.data)
    return cards, feats, res


def preflop_equity(holes: np.ndarray, R: int, seed: int = 0) -> np.ndarray:
    """Equity vs a random hand for many 5-card hands (C backend only)."""
    holes = np.ascontiguousarray(holes, dtype=np.int32)
    out = np.empty(len(holes), dtype=np.float64)
    _lib.p5_preflop_equity(holes.ctypes.data, len(holes), R, seed, out.ctypes.data)
    return out

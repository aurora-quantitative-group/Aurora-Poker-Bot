"""plo5bot.blueprint - the MCCFR blueprint strategy table (see train/train_blueprint.py)."""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Blueprint:
    def __init__(self, path: str | None = None):
        path = path or os.path.join(HERE, "data", "blueprint.npz")
        z = np.load(path)
        self.tables = [z[f"s{s}"] for s in range(4)]
        self.nb = tuple(int(x) for x in z["nb"])
        self.iters = int(z["iters"][0]) if "iters" in z.files else 0

    def probs(self, street: int, local: int, bucket: int) -> list:
        """Action probabilities [FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT].
        Plain Python on the 4 quantised ints: the same IEEE divisions as the
        numpy version, without numpy's per-call overhead on tiny arrays."""
        q = self.tables[street][local, bucket].tolist()
        s = sum(q)
        return [x / s for x in q] if s > 0 else [float(x) for x in q]


def try_load(path: str | None = None):
    try:
        return Blueprint(path)
    except Exception:
        return None

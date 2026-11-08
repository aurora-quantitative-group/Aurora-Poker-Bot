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

    def probs(self, street: int, local: int, bucket: int) -> np.ndarray:
        """Action probabilities [FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT]."""
        q = self.tables[street][local, bucket].astype(np.float64)
        s = q.sum()
        return q / s if s > 0 else q


def try_load(path: str | None = None):
    try:
        return Blueprint(path)
    except Exception:
        return None

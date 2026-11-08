"""Sparring opponent for tools/arena.py: HeuristicNit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make  # noqa: E402

AGENT_NAME = "HeuristicNit"
new_half, choose_action, on_hand_end = make(params={'tight': 0.15, 'bluff': 0.0, 'aggr': 0.7}, name=AGENT_NAME)

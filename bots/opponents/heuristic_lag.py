"""Sparring opponent for tools/arena.py: HeuristicLAG."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make  # noqa: E402

AGENT_NAME = "HeuristicLAG"
new_half, choose_action, on_hand_end = make(params={'aggr': 2.0, 'bluff': 0.35}, name=AGENT_NAME)

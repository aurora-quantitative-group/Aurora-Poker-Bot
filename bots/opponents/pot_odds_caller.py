"""Sparring opponent for tools/arena.py: PotOddsCaller."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make  # noqa: E402

AGENT_NAME = "PotOddsCaller"
new_half, choose_action, on_hand_end = make(caller=True, name=AGENT_NAME)

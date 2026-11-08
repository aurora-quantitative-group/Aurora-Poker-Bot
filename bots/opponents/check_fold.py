"""Sparring opponent for tools/arena.py: CheckFold."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make  # noqa: E402

AGENT_NAME = "CheckFold"
new_half, choose_action, on_hand_end = make(check_fold=True, name=AGENT_NAME)

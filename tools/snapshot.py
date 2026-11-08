"""
tools/snapshot.py - freeze the current bot as bots/<name>/ for regression matches.

    python tools/snapshot.py v2_blueprint "MCCFR blueprint, 400M iters"

Copies agent.py and the plo5bot package (with its data files). The arena
puts bots/<name>/ first on sys.path, so the snapshot keeps using its own
frozen plo5bot code and tables. bots/ is excluded from the Docker image.
"""

import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    name = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else ""
    dst = os.path.join(ROOT, "bots", name)
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists; never overwrite a version - pick a new name")
    os.makedirs(dst)
    shutil.copy2(os.path.join(ROOT, "agent.py"), dst)
    shutil.copytree(os.path.join(ROOT, "plo5bot"), os.path.join(dst, "plo5bot"),
                    ignore=shutil.ignore_patterns("__pycache__", "csrc"))
    sys.path.insert(0, ROOT)
    from tools.arena import git_commit
    json.dump({"name": name, "note": note, "commit": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(dst, "VERSION.json"), "w"), indent=1)
    print(f"snapshot -> {dst}")


if __name__ == "__main__":
    main()

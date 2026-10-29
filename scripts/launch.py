#!/usr/bin/env python
"""Build, print or submit the training command of one cell (thin wrapper over marl_workflows.launch).

Examples::

    python scripts/launch.py --task math --workflow voting --policy ip --scale 1.7B --print
    python scripts/launch.py --paper-run lm7qcvrf --print
    python scripts/launch.py --task math --workflow voting --policy ip --scale 1.7B \\
        --slurm-config configs/slurm/dgxh_preempt_h100.conf --submit --chain 3
    python scripts/launch.py eval --slurm-config configs/slurm/dgxh_preempt_h100.conf -- --task-type math ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a checkout without installing: put the repository root on the import path.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from marl_workflows.launch import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli())

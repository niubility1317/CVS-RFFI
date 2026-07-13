from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import launch_phase1_dgleo_hiercore8_20260713 as launcher
import queue_phase1_dgleo_p0closed8_20260713 as capacity


capacity.launcher = launcher
capacity.DEFAULT_RUN_ID = launcher.DEFAULT_RUN_ID


if __name__ == "__main__":
    raise SystemExit(capacity.main())

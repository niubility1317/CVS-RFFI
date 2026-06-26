from __future__ import annotations

from pathlib import Path
import sys

"""Few-shot fine-tuning entrypoint.

This module intentionally reuses `train_ra.py` mechanics. In a real run, pass a
config whose dataset section selects the target receiver and shots per TX.
"""

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.ra_collab.train_ra import main


if __name__ == "__main__":
    main()

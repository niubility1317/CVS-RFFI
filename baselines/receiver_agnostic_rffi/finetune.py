from __future__ import annotations

"""Few-shot fine-tuning entrypoint.

This module intentionally reuses `train_ra.py` mechanics. In a real run, pass a
config whose dataset section selects the target receiver and shots per TX.
"""

from baselines.receiver_agnostic_rffi.train_ra import main


if __name__ == "__main__":
    main()

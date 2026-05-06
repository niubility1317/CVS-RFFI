"""Compatibility entry point for SGC-Adapter experiments.

The root project keeps the full trainer in train.py. This wrapper preserves the
planned `python train_sgc.py ...` command form without duplicating the training
loop.
"""

from train import main


if __name__ == "__main__":
    main()

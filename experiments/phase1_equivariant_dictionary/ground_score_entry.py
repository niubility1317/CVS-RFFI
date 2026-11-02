"""Expose the independent dictionary scorer to the truth-last audit controller."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.phase1_equivariant_dictionary.ground import main


if __name__ == '__main__':
    main()

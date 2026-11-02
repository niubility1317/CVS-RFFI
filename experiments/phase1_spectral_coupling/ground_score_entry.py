"""Expose the spectral model's fixed ground scorer to the independent audit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.phase1_spectral_coupling.ground import main
if __name__ == '__main__': main()

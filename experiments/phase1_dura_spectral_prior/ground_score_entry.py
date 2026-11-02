"""Expose the fixed spectral DURA scorer to the truth-last audit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.phase1_dura_spectral_prior.ground import main
if __name__ == '__main__':
    main()

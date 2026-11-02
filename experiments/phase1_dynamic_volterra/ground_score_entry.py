"""Expose the fixed dynamic Volterra scorer to the truth-last audit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.phase1_dynamic_volterra.ground import main
if __name__ == '__main__': main()

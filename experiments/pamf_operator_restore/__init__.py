"""Source-only feasibility of a reception-operator restoration branch."""
import sys
from pathlib import Path

CODE=Path(__file__).resolve().parents[2]/'code'
if str(CODE) not in sys.path:
    sys.path.insert(0,str(CODE))

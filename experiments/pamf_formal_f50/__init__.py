from pathlib import Path
import sys

_code = str(Path(__file__).resolve().parents[2]/'code')
if _code not in sys.path:
    sys.path.append(_code)

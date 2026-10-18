"""Fixed E50 replication; inherit unchanged runtime helpers from bundled seed01 package."""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parents[1]/'cvs_practical_budget50'))

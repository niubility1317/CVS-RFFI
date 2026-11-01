"""Check invariant-frame phase averaging on synthetic and real source IQ."""
import subprocess
from pathlib import Path

p = Path(__file__).parent
script = p / 'phase1_source_invariant_dura.py'
assert script.exists(), 'invariant-frame source pilot must be built first'
proc = subprocess.run(['E:/type10-7/local_envs/phase1-cu128/Scripts/python.exe', '-X', 'utf8', str(script), '--smoke'], capture_output=True, timeout=320)
out = proc.stdout.decode('utf8', errors='replace')
err = proc.stderr.decode('utf8', errors='replace')
print(out)
print(err)
assert proc.returncode == 0 and 'VERIFIED_SOURCE_INVARIANT_DURA_SMOKE 4' in out

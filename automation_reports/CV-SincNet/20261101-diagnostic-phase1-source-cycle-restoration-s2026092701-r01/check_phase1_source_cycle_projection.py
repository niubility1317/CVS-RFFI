"""Test periodic projection with phase drift and real source waveforms."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.concatenate((np.flatnonzero(split==1)[:2],np.flatnonzero(split==2)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = 'for view in views:\n if view=='
new = "base=np.arange(1,21,dtype=np.float64)+1j*np.arange(20,dtype=np.float64);test=(base[residues]*np.exp(.17j*cycles))[None,:];approx,angle=project(test);assert np.max(abs(approx-test))<1e-10 and abs(angle[0]-.17)<1e-10\n" + old
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "print('ALL_CYCLE_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + "\nassert len(data)==4 and all(x['probs'].shape==(6,4,6) for x in data.values())\nprint('VERIFIED_SOURCE_CYCLE_SMOKE',n,'test_reads',0,flush=True)\n"
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=300)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=320)
print(proc.stdout.decode('utf8', errors='replace'))
print(proc.stderr.decode('utf8', errors='replace'))
raise SystemExit(proc.returncode)

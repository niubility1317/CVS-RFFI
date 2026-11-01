"""Check identity channel and real-source pseudo-pilot candidates."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_channel_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
rtree = ast.parse(ast.literal_eval(node))
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.concatenate((np.flatnonzero(split==1)[:2],np.flatnonzero(split==2)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = 'for view in views:\n if view=='
new = "test_rng=np.random.default_rng(2026110439);test=test_rng.normal(size=(3,256))+1j*test_rng.normal(size=(3,256));xx,aa,HH,rr=channel_from_cycles(test,test);assert np.max(abs(HH-1))<1e-10 and np.max(abs(invert(xx,aa,HH,rr,1.)-test))<1e-9\n" + old
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "print('ALL_CYCLE_CHANNEL_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + "\nassert len(data)==4 and all(v['probs'].shape==(12,4,6) for v in data.values())\nprint('VERIFIED_SOURCE_CYCLE_CHANNEL_SMOKE',n,'test_reads',0,flush=True)\n"
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

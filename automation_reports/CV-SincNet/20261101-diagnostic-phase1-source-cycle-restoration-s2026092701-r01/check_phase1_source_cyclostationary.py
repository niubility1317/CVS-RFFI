import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cyclostationary_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = 'size=min(2048,len(available))'
assert inner.count(old) == 1
inner = inner.replace(old, 'size=4')
old = 'reference=correlation(clean_iq)'
new = old + "\nsynthetic=np.zeros((1,2,256),dtype=np.float32);synthetic[:,0,:]=1;assert np.allclose(correlation(synthetic),1,atol=1e-12)"
assert inner.count(old) == 1
inner = inner.replace(old, new)
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=300)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=320)
out = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in out.splitlines() if not s.startswith('SOURCE_CYCLOSTATIONARY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
data = json.loads(next(s[len('SOURCE_CYCLOSTATIONARY '):] for s in out.splitlines() if s.startswith('SOURCE_CYCLOSTATIONARY ')))
assert data['source_ids'] == 4 and data['U_s_labels_read'] == data['V_reads'] == data['test_IQ_reads'] == 0
assert len(data['results']) == 4 and all(len(x) >= 12 for x in data['results'].values())
print('VERIFIED_SOURCE_CORRELATION_SMOKE', data['source_ids'])

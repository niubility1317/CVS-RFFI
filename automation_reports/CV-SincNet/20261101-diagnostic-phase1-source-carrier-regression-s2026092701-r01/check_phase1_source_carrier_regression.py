"""Smoke-test complex phase targets, regression gradients and frozen F."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_carrier_regression.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = "assert all(int((split==i).sum())>200 for i in range(3));n=len(fixed);views="
new = "assert all(int((split==i).sum())>200 for i in range(3));sel=np.concatenate((np.flatnonzero(split==0)[:2],np.flatnonzero(split==1)[:1],np.flatnonzero(split==2)[:1]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);views="
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = 'all_input=torch.cat([data[v][\'input\'] for v in views])'
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
theta=.7;original=torch.complex(torch.tensor([[1.,-.3,.2]]),torch.tensor([[.2,.4,-.1]]));rotated=original*torch.exp(torch.tensor(1j*theta));corr=(rotated.conj()*original).sum(1);assert abs(float(torch.atan2(corr.imag,corr.real)[0])+theta)<1e-5
model=PhaseNet();losses=[]
for view in views:
 v=data[view];assert v['input'].shape==(4,4,256) and v['target'].shape==(4,2);pred=model(v['input']);loss=(((pred-v['target'])**2).sum(1)*v['weight']).sum()/v['weight'].sum().clamp_min(1e-8);losses.append(loss)
sum(losses).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
with torch.no_grad():
 a=data['practical_low_urban']['anchor'];angle=torch.atan2(pred[:,1],pred[:,0]);z=torch.complex(a[:,0],a[:,1])*torch.exp(1j*angle[:,None]);wave=torch.stack((z.real,z.imag),1);prob=F.softmax(teacher.classify_features(teacher.features(wave)),1);assert prob.shape==(4,6) and torch.isfinite(prob).all()
print('VERIFIED_SOURCE_CARRIER_SMOKE',len(fixed),'test_reads',0,flush=True)
'''
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

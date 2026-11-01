"""Smoke-test zero initialization, cycle constraint, gradients and frozen F."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_residual_net.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote = ast.literal_eval(node)
rtree = ast.parse(remote)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.concatenate((np.flatnonzero(split==0)[:2],np.flatnonzero(split==1)[:1],np.flatnonzero(split==2)[:1]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "train_input=torch.cat([data[v]['input'] for v in views])"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
model=CycleNet();v=data['practical_low_urban'];assert v['input'].shape==(4,6,13,20) and v['target'].shape==(4,2,256);initial=model(v['input']);assert torch.equal(initial,torch.zeros_like(initial))
opt=torch.optim.AdamW(model.parameters(),lr=.0005);loss=(initial-v['target']).square().mean();assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step();delta=model(v['input']);assert all(float(delta[:,:,r::20].mean(2).abs().max())<1e-6 for r in range(20))
with torch.no_grad():
 d=np.from_dlpack(delta.detach().contiguous());dz=(d[:,0].astype(np.float64)+1j*d[:,1])*v['rms'][:,None]/v['rotation'];wave=v['anchor']+v['gate'][:,None]*dz;wiq=np.stack((wave.real,wave.imag),1).astype(np.float32);x=core.tensor(wiq,'cpu');prob=F.softmax(teacher.classify_features(teacher.features(x)),1);assert prob.shape==(4,6) and torch.isfinite(prob).all()
print('VERIFIED_SOURCE_CYCLE_NET_SMOKE',n,'test_reads',0,flush=True)
'''
remote_smoke = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=300)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', 'N607', 'python3', '-'], input=remote_smoke.encode(), capture_output=True, timeout=320)
print(proc.stdout.decode('utf8', errors='replace'))
print(proc.stderr.decode('utf8', errors='replace'))
raise SystemExit(proc.returncode)

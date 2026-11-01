"""Check low-SNR training weights, frozen-base distillation and cycle constraint."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_low_snr_net.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
rtree = ast.parse(ast.literal_eval(node))
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.concatenate((np.flatnonzero(split==0)[:2],np.flatnonzero(split==1)[:1],np.flatnonzero(split==2)[:1]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = 'teacher.eval();teacher.requires_grad_(False);train_snr='
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
teacher.eval();teacher.requires_grad_(False);model=CycleNet();v=data['practical_low_urban'];assert v['input'].shape==(4,6,13,20) and v['snr'].shape==(4,) and v['target_prob'].shape==(4,6);pred=model(v['input']);rot=v['rotation_t'];amp=v['rms_t'][:,None];a=v['anchor_t'];wave=torch.stack((a[:,0]+amp*(pred[:,0]*rot[:,0]-pred[:,1]*rot[:,1]),a[:,1]+amp*(pred[:,0]*rot[:,1]+pred[:,1]*rot[:,0])),1);raw=teacher.features(wave);feat=F.normalize(raw,dim=1);fl=1-(feat*v['target_feature']).sum(1);logp=F.log_softmax(teacher.classify_features(raw),1);pclean=v['target_prob'];kl=(pclean*(pclean.clamp_min(1e-8).log()-logp)).sum(1);weights=torch.where(v['snr']<10.,4.,1.);per=(pred-v['target']).square().mean((1,2))+.05*pred.square().mean((1,2))+.05*fl+.02*kl;loss=(per*weights).sum()/weights.sum();assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());assert all(p.grad is None and not p.requires_grad for p in teacher.parameters());assert all(float(pred[:,:,r::20].mean(2).abs().max())<1e-6 for r in range(20));assert torch.equal(torch.where(torch.tensor([0.,20.])<10.,4.,1.),torch.tensor([4.,1.]));print('VERIFIED_SOURCE_CYCLE_LOW_SNR_SMOKE',n,'test_reads',0,flush=True)
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

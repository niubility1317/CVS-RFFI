"""Smoke-test real phase-set features and masked class selection without labels."""
import ast
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_phase_set_selector.py').read_text(encoding='utf8'))
remote_node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote = ast.literal_eval(remote_node)
remote_tree = ast.parse(remote)
run = next(n for n in ast.walk(remote_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'assert all(int((split==i).sum())>200 for i in range(3))'
assert inner.count(old) == 1
inner = inner.replace(old, old + "\nsel=np.concatenate((np.flatnonzero(split==0)[:2],np.flatnonzero(split==1)[:1],np.flatnonzero(split==2)[:1]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];assert len(fixed)==4")
marker = "print('ALL_PHASE_UTILITY_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
for view in views:
 v=data[view];f=v['phase_feat'];prob=v['phase_prob'];feat=torch.cat((f.mean(1),f.std(1,unbiased=False),v['base_prob'],v['raw_prob'],prob.mean(1),prob.std(1,unbiased=False),v['energy']),1)
 assert feat.shape==(4,346) and v['ratio'].shape==(4,) and torch.isfinite(feat).all() and torch.isfinite(v['ratio']).all()
 q=np.from_dlpack(F.softmax(nn.Sequential(nn.Linear(346,128),nn.GELU(),nn.Linear(128,6))(feat),1).detach().contiguous());phase=np.from_dlpack(prob.contiguous());desired=q.argmax(1);phase_class=phase.argmax(2);eligible=phase_class==desired[:,None];candidate_prob=np.take_along_axis(phase,desired[:,None,None],axis=2)[:,:,0];masked=np.where(eligible,candidate_prob,-np.inf);choice=masked.argmax(1);assert choice.shape==(4,) and masked.shape==(4,12)
print('VERIFIED_PHASE_SET_SOURCE_SMOKE',len(fixed),'test_reads',0,flush=True)
'''
remote_smoke = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=300)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config', 'N607', 'python3', '-'], input=remote_smoke.encode(), capture_output=True, timeout=320)
print(proc.stdout.decode('utf8', errors='replace'))
print(proc.stderr.decode('utf8', errors='replace'))
raise SystemExit(proc.returncode)

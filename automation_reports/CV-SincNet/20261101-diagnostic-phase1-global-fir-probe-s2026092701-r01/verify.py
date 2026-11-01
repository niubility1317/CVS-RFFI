import json,subprocess
from pathlib import Path

p=Path(__file__).parent
original=json.loads((p/'phase1_source_global_filter_probe.json').read_text(encoding='utf8'))
current=json.loads((p/'phase1_source_global_filter_probe_v2.json').read_text(encoding='utf8'))
coeff=current['selected_filter_coefficients'];check=dict(current);check.pop('selected_filter_coefficients')
assert check==original and current['selected']['kind']=='shared_widely_linear_fir'
remote=r'''
from pathlib import Path
import json,os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-filter-student-E30-s2026092701-r01-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner],cwd=release,env=env,check=True,timeout=90)
'''
inner=r'''
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a
from experiments.pamf_phase1_flow.filter_student import apply_filter_delta,project_delta
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
groups=[]
for role,count,n in (('L_s',6300,256),('U_s',56700,768)):
 all_ix=np.flatnonzero(roles==role);first=all_ix[np.linspace(0,count-1,n,dtype=int)]
 remaining=np.setdiff1d(all_ix,first,assume_unique=True);second=remaining[np.linspace(0,len(remaining)-1,n,dtype=int)]
 remaining=np.setdiff1d(remaining,second,assume_unique=True);groups.append(remaining[np.linspace(0,len(remaining)-1,n,dtype=int)])
selected=np.concatenate(groups);order=np.random.default_rng(2026110130).permutation(len(selected));ix=selected[np.sort(order[819:])]
coeff=torch.tensor(json.loads(__COEFF_PAYLOAD__),dtype=torch.float32);expected=json.loads(__EXPECTED_PAYLOAD__)
assert coeff.shape==(1,2,17,2) and torch.isfinite(coeff).all() and len(ix)==205
with torch.no_grad():
 for view in (1,20):
  raw=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
  stats={'old':0,'new':0,'repairs':0,'damages':0,'old_nmse':0.,'new_nmse':0.}
  for start in range(0,len(ix),32):
   batch=ix[start:start+32];received=core.tensor(raw[batch],'cpu');truth=core.tensor(clean[batch],'cpu');anchor=dura(received)
   y=project_delta(anchor,anchor+apply_filter_delta(anchor,coeff.expand(len(batch),-1,-1,-1)),epsilon=.2)
   ref=teacher.classify_features(teacher.features(truth)).argmax(1);old=teacher.classify_features(teacher.features(anchor)).argmax(1)==ref
   new=teacher.classify_features(teacher.features(y)).argmax(1)==ref
   stats['old']+=int(old.sum());stats['new']+=int(new.sum());stats['repairs']+=int((~old&new).sum());stats['damages']+=int((old&~new).sum())
   energy=truth.square().mean((1,2)).clamp_min(1e-4)
   stats['old_nmse']+=float(((anchor-truth).square().mean((1,2))/energy).sum())
   stats['new_nmse']+=float(((y-truth).square().mean((1,2))/energy).sum())
  e=next(row for row in expected if row['view']==view)
  assert (stats['old'],stats['new'],stats['repairs'],stats['damages'])==(e['baseline_teacher_match'],e['probe_teacher_match'],e['repairs'],e['damages'])
  assert abs(stats['old_nmse']/len(ix)-e['baseline_iq_nmse'])<1e-6 and abs(stats['new_nmse']/len(ix)-e['probe_iq_nmse'])<1e-6
  print('VERIFIED_FIXED_COEFFICIENTS_SOURCE_VIEW',view,stats)
print('VERIFIED_ONLY_SOURCE_INFERENCE_FIT_NOT_REPEATED')
'''.replace('__COEFF_PAYLOAD__',repr(json.dumps(coeff))).replace('__EXPECTED_PAYLOAD__',repr(json.dumps(current['results'])))
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=110)
print(proc.stdout.decode('utf8',errors='replace'));print(proc.stderr.decode('utf8',errors='replace'));raise SystemExit(proc.returncode)

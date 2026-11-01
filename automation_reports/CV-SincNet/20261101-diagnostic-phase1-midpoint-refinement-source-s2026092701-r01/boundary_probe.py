"""Read-only source probe of epoch 27; never selects a test checkpoint."""
import json
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-boundary-student-E30-s2026092701-r02-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='f72f9618859fce30a1983a3b4b10c8c869b2dbba'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=300)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,boundary_student_experiment as b
from experiments.pamf_phase1_flow.boundary_student import make_source_oracle
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
from experiments.pamf_phase1_flow.feature_conditioned_experiment import source_labels
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
checkpoint=root/'runs/20261101-phase1-boundary-student-E30-s2026092701-r02/flow/checkpoints/epoch-27.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert (state['epoch'],state['updates'],state['contract']['commit'])==(27,13311,'f72f9618859fce30a1983a3b4b10c8c869b2dbba')
student=b.make_student('cpu');student.load_state_dict(state['model'],strict=True);student.eval().requires_grad_(False)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
records={};states={}
for view in ('clean','01','20'):
 if view!='clean':
  with np.load(root/'runs'/PARENT/f'DURA_default-{int(view)}.npz',allow_pickle=False) as z:
   saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
  assert np.array_equal(saved['source_positions'],fixed) and np.array_equal(saved['physical_ids'],ids[fixed])
  assert np.array_equal(saved['roles'],roles[fixed]) and np.array_equal(saved['clean'],clean[fixed])
  raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False);states[view]=saved['state']
 obs={k:[] for k in ('reference','base','candidate','oracle','delta_ratio')}
 for start in range(0,len(fixed),32):
  ix=fixed[start:start+32];x=core.tensor(clean[ix] if view=='clean' else raw[ix],'cpu');target=core.tensor(clean[ix],'cpu')
  with torch.no_grad():
   anchor=dura(x)
   if view!='clean':torch.testing.assert_close(anchor,core.tensor(saved['native'][start:start+len(ix)],'cpu'),rtol=1e-4,atol=1e-5)
   y=student(x,anchor)
   ratio=(y-anchor).flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8)
   assert torch.all(ratio<=.050001)
   ref=teacher.classify_features(teacher.features(target)).argmax(1);old=teacher.classify_features(teacher.features(anchor)).argmax(1)
   candidate=teacher.classify_features(teacher.features(y)).argmax(1)
  oracle,wrong=make_source_oracle(teacher,anchor,target,epsilon=.05,steps=12)
  with torch.no_grad():oracle_pred=teacher.classify_features(teacher.features(oracle)).argmax(1)
  for key,value in [('reference',ref),('base',old),('candidate',candidate),('oracle',oracle_pred),('delta_ratio',ratio)]:obs[key].append(value.cpu())
 records[view]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
 print('SOURCE_PREDICTIONS_FIXED',view,flush=True)
# Source labels are read only after candidate and source-oracle predictions are fixed.
labels=source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s'
assert np.all(labels[fixed[128:]]==-1)
def paired(reference,before,after,mask):
 old=before==reference;new=after==reference
 return dict(count=int(mask.sum()),before=int((mask&old).sum()),after=int((mask&new).sum()),
  repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()))
out=[]
for view in ('clean','01','20'):
 z=records[view];masks=[('all',np.ones(len(fixed),dtype=bool)),('L_s',labeled)]
 if view!='clean':masks.append(('blocked',states[view]=='BLOCKED'))
 rows=[]
 for group,mask in masks:
  for kind in ('candidate','oracle'):
   rows.append(dict(group=group,kind=kind,teacher=paired(z['reference'],z['base'],z[kind],mask),
    L_s_truth=paired(labels[fixed],z['base'],z[kind],mask&labeled)))
  rows[-2]['delta_ratio_p50']=float(np.quantile(z['delta_ratio'][mask],.5));rows[-2]['delta_ratio_p95']=float(np.quantile(z['delta_ratio'][mask],.95))
 out.append(dict(view=view,groups=rows))
print('BOUNDARY_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_INTERMEDIATE_CHECKPOINT',epoch=27,updates=13311,
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_records=512,views=['clean',1,20],
 source_L_s_labels_materialized=6300,source_L_s_labels_scored=128,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 clean_target_used_only_for_source_oracle_and_scoring=True,final_model_selection='fixed epoch 30',results=out,
 caveat='Same source training records and observed views; oracle uses privileged clean target and is not deployable. Teacher agreement is not TX accuracy.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('BOUNDARY_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('BOUNDARY_SOURCE_PROBE '))
 path=Path(__file__).parent/'boundary.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)

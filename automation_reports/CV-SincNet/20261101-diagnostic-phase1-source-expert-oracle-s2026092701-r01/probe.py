"""Source-only privileged union of fixed experts; never used as a selector."""
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
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=200)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,boundary_student_experiment as bs,feature_conditioned_experiment as fs
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
digest=a.sha(root/a.DURA);assert digest=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
commit='f72f9618859fce30a1983a3b4b10c8c869b2dbba'
boundary_path=root/'runs/20261101-phase1-boundary-student-E30-s2026092701-r02/flow/final.pt'
feature_path=root/'runs/20261101-phase1-feature-conditioned-E30-s2026092701-r02/flow/final.pt'
boundary=bs.load(boundary_path,'cpu',commit,digest);feature=fs.load(feature_path,'cpu',commit,digest)
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
assert len(fixed)==len(np.unique(fixed))==512
views={};states={};experts=('raw','dura','boundary','feature')
with torch.no_grad():
 for view in ('clean','01','20'):
  if view!='clean':
   with np.load(root/'runs'/PARENT/f'DURA_default-{int(view)}.npz',allow_pickle=False) as z:
    saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
   assert np.array_equal(saved['source_positions'],fixed) and np.array_equal(saved['physical_ids'],ids[fixed])
   assert np.array_equal(saved['roles'],roles[fixed]) and np.array_equal(saved['clean'],clean[fixed])
   raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False);states[view]=saved['state']
  decisions={k:[] for k in experts}
  for start in range(0,len(fixed),32):
   ix=fixed[start:start+32];x=core.tensor(clean[ix] if view=='clean' else raw[ix],'cpu');anchor=dura(x)
   if view!='clean':torch.testing.assert_close(anchor,core.tensor(saved['native'][start:start+len(ix)],'cpu'),rtol=1e-4,atol=1e-5)
   feat=teacher.features(anchor);base=teacher.classify_features(feat)
   values={'raw':x,'dura':anchor,'boundary':boundary(x,anchor),'feature':feature(x,anchor,feat,base)}
   for name,wave in values.items():
    if name=='boundary':assert torch.all((wave-anchor).flatten(1).norm(dim=1)<=.05*anchor.flatten(1).norm(dim=1)+1e-5)
    if name=='feature':assert torch.all((wave-anchor).flatten(1).norm(dim=1)<=.1*anchor.flatten(1).norm(dim=1)+1e-5)
    score=teacher.classify_features(teacher.features(wave));assert torch.isfinite(score).all();decisions[name].append(score.argmax(1).cpu())
  views[view]={k:np.from_dlpack(torch.cat(v)) for k,v in decisions.items()}
  print('SOURCE_EXPERT_DECISIONS_FIXED',view,flush=True)
# All experts are fixed before source labels are materialized.
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[128:]]==-1)
reference=views['clean']['raw'];results=[]
for view in ('clean','01','20'):
 pred=views[view];masks=[('all',np.ones(len(fixed),dtype=bool)),('L_s',labeled)]
 if view!='clean':masks.append(('blocked',states[view]=='BLOCKED'))
 rows=[]
 for group,mask in masks:
  for ref_name,ref,valid in [('clean_teacher',reference,mask),('L_s_truth',labels[fixed],mask&labeled)]:
   right={name:pred[name]==ref for name in experts};base=right['dura'];union=np.logical_or.reduce([right[name] for name in experts]);three=np.logical_or.reduce([right[name] for name in ('dura','boundary','feature')])
   rows.append(dict(group=group,reference=ref_name,count=int(valid.sum()),experts={name:int((valid&right[name]).sum()) for name in experts},
    three_oracle=int((valid&three).sum()),all_four_oracle=int((valid&union).sum()),
    available_rescues=int((valid&~base&union).sum()),raw_unique_rescues=int((valid&~three&right['raw']).sum())))
 results.append(dict(view=view,groups=rows))
print('SOURCE_EXPERT_ORACLE '+json.dumps(dict(status='SOURCE_ONLY_PRIVILEGED_EXPERT_ORACLE',source_records=512,views=['clean',1,20],
 experts=list(experts),F_run=core.F_RUN,F_row=core.F_ROW,DURA_sha256=digest,
 boundary_checkpoint_sha256=hashlib.sha256(boundary_path.read_bytes()).hexdigest(),feature_checkpoint_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
 source_L_s_labels_materialized=6300,source_L_s_labels_scored=128,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 all_predictions_before_source_labels=True,models_updated=0,results=results,
 caveat='Per-record clean teacher or L_s truth is used only for privileged upper bound; not a deployable selector. Experts trained on source records; clean teacher agreement is not TX accuracy.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=220)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('SOURCE_EXPERT_ORACLE '):] for s in stdout.splitlines() if s.startswith('SOURCE_EXPERT_ORACLE '))
 path=Path(__file__).parent/'result.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)

"""Source-only predicted-class input refinement; no clean target enters refinement."""
import json
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-class-hypothesis-E30-s2026092701-r01-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='54e93c9b8ea1f6419d94a161cbcea5ffa63f444b'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=300)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,class_hypothesis_experiment as b
from experiments.pamf_phase1_flow.boundary_source_oracle import projected_targeted_steps
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
from experiments.pamf_phase1_flow.feature_conditioned_experiment import source_labels
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
checkpoint=root/'runs/20261101-phase1-class-hypothesis-E30-s2026092701-r01/flow/checkpoints/epoch-14.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert (state['epoch'],state['updates'],state['contract']['commit'])==(14,6902,'54e93c9b8ea1f6419d94a161cbcea5ffa63f444b')
model=b.make_model('cpu');model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
records={};states={}
def classify(x):return teacher.classify_features(teacher.features(x))
for view in ('clean','01','20'):
 if view!='clean':
  with np.load(root/'runs'/PARENT/f'DURA_default-{int(view)}.npz',allow_pickle=False) as z:
   saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
  assert np.array_equal(saved['source_positions'],fixed) and np.array_equal(saved['physical_ids'],ids[fixed])
  assert np.array_equal(saved['roles'],roles[fixed]) and np.array_equal(saved['clean'],clean[fixed])
  raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False);states[view]=saved['state']
 obs={k:[] for k in ('base','q','confidence','generated','refined')}
 for start in range(0,len(fixed),32):
  ix=fixed[start:start+32];x=core.tensor(clean[ix] if view=='clean' else raw[ix],'cpu')
  with torch.no_grad():
   anchor=dura(x)
   if view!='clean':torch.testing.assert_close(anchor,core.tensor(saved['native'][start:start+len(ix)],'cpu'),rtol=1e-4,atol=1e-5)
   feature=teacher.features(anchor);baseline=teacher.classify_features(feature);q,candidate=model(x,anchor,feature,baseline)
   confidence,target=torch.softmax(q,1).max(1);base=baseline.argmax(1);generated=classify(candidate).argmax(1)
   eligible=(target!=base)&(confidence>=.5)
  refined=anchor.detach().clone()
  if eligible.any():
   norm=anchor[eligible].flatten(1).norm(dim=1).clamp_min(1e-8)
   refined[eligible]=projected_targeted_steps(classify,anchor[eligible],target[eligible],norm,epsilon=.05,steps=12)
  with torch.no_grad():
   ratio=(refined-anchor).flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8)
   assert torch.all(ratio<=.050001)
   refined_pred=classify(refined).argmax(1)
  for key,value in [('base',base),('q',target),('confidence',confidence),('generated',generated),('refined',refined_pred)]:obs[key].append(value.cpu())
 records[view]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
 print('SOURCE_Q_AND_REFINEMENTS_FIXED',view,flush=True)
# Only now read clean-teacher reference and legal L_s labels for source scoring.
refs=[]
with torch.no_grad():
 for start in range(0,len(fixed),32):refs.append(classify(core.tensor(clean[fixed[start:start+32]],'cpu')).argmax(1).cpu())
ref=np.from_dlpack(torch.cat(refs));labels=source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s'
assert np.all(labels[fixed[128:]]==-1)
def paired(reference,before,after,mask):
 old=before==reference;new=after==reference
 return dict(count=int(mask.sum()),before=int((mask&old).sum()),after=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()))
out=[]
for view in ('clean','01','20'):
 z=records[view];masks=[('all',np.ones(len(fixed),dtype=bool)),('L_s',labeled)]
 if view!='clean':masks.append(('blocked',states[view]=='BLOCKED'))
 rows=[]
 for group,mask in masks:
  for threshold in (.5,.7,.8,.9,.95):
   propose=(z['q']!=z['base'])&(z['confidence']>=threshold)
   for kind in ('generated','refined'):
    accepted=propose&(z[kind]==z['q']);after=np.where(accepted,z['q'],z['base'])
    rows.append(dict(group=group,threshold=threshold,method=kind,proposed=int((mask&propose).sum()),accepted=int((mask&accepted).sum()),
     proposed_matches_teacher=int((mask&propose&(z['q']==ref)).sum()),teacher=paired(ref,z['base'],after,mask),
     L_s_truth=paired(labels[fixed],z['base'],after,mask&labeled)))
 out.append(dict(view=view,groups=rows))
print('SELF_REFINE_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_SELF_REFINEMENT_PROBE',epoch=14,updates=6902,
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_records=512,views=['clean',1,20],
 thresholds=[.5,.7,.8,.9,.95],predicted_class_only=True,steps=12,epsilon=.05,
 source_L_s_labels_materialized=6300,source_L_s_labels_scored=128,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 predictions_fixed_before_source_truth=True,final_checkpoint_fixed_epoch=30,results=out,
 caveat='Source training records; refinement can be adversarial and is not guaranteed physically faithful. Clean teacher agreement is not TX accuracy.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('SELF_REFINE_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('SELF_REFINE_SOURCE_PROBE '))
 path=Path(__file__).parent/'self_refine.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)

"""Read-only Phase1 source diagnostic; never access validation or test data."""
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='20261101-phase1-class-hypothesis-E30-s2026092701-r01'
release=root/'releases'/(run+'-release');native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
ckpt=root/'runs'/run/'flow/checkpoints/epoch-01.pt'
assert ckpt.is_file()
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=r"""
from pathlib import Path
import hashlib,json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,class_hypothesis_experiment as b
from experiments.pamf_phase1_flow.class_hypothesis import select_verified
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
ckpt=root/'runs/20261101-phase1-class-hypothesis-E30-s2026092701-r01/flow/checkpoints/epoch-01.pt'
state=torch.load(ckpt,map_location='cpu',weights_only=False)
assert (state['epoch'],state['updates'],state['contract']['commit'])==(1,493,'54e93c9b8ea1f6419d94a161cbcea5ffa63f444b')
torch.set_num_threads(2)
model=b.make_model('cpu');model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
labels=b.source_labels(root,cache,clean,roles,contract)
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
assert len(fixed)==512 and np.all(labels[fixed[128:]]==-1)
results=[]
with torch.no_grad():
 for view in (1,20):
  with np.load(root/'runs'/PARENT/f'DURA_default-{view}.npz',allow_pickle=False) as z:
   saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native')}
  assert np.array_equal(saved['source_positions'],fixed) and np.array_equal(saved['physical_ids'],ids[fixed])
  assert np.array_equal(saved['roles'],roles[fixed]) and np.array_equal(saved['clean'],clean[fixed])
  received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
  collect={k:[] for k in ('ref','old','new','posterior','accepted','confident','verified','confidence','candidate_class')}
  for start in range(0,len(fixed),32):
   sl=slice(start,start+32);ix=fixed[sl];x=core.tensor(received[ix],'cpu');c=core.tensor(saved['clean'][sl],'cpu')
   anchor=dura(x);torch.testing.assert_close(anchor,core.tensor(saved['native'][sl],'cpu'),rtol=1e-4,atol=1e-5)
   f=teacher.features(anchor);old=teacher.classify_features(f);q,candidate=model(x,anchor,f,old)
   checked=teacher.classify_features(teacher.features(candidate))
   y,accepted=select_verified(anchor,candidate,torch.softmax(q,1),old.argmax(1),checked.argmax(1),threshold=.95)
   new=teacher.classify_features(teacher.features(y));ref=teacher.classify_features(teacher.features(c))
   for k,v in [('ref',ref.argmax(1)),('old',old.argmax(1)),('new',new.argmax(1)),('posterior',q.argmax(1)),('accepted',accepted),('confident',torch.softmax(q,1).max(1).values>=.95),('verified',checked.argmax(1)==q.argmax(1)),('confidence',torch.softmax(q,1).max(1).values),('candidate_class',checked.argmax(1))]:collect[k].append(v.cpu())
  z={k:np.from_dlpack(torch.cat(v).to(torch.int64) if k in ('accepted','confident','verified') else torch.cat(v)) for k,v in collect.items()}
  def paired(reference,mask,after=None):
   old=z['old']==reference;new=(z['new'] if after is None else after)==reference
   return dict(count=int(mask.sum()),old_matches=int((mask&old).sum()),new_matches=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()))
  allrows=np.ones(512,dtype=bool);labeled=saved['roles']=='L_s'
  disagree=z['posterior']!=z['old'];confident=z['confident'].astype(bool);verified=z['verified'].astype(bool)
  assert np.array_equal(z['accepted'].astype(bool),disagree&confident&verified)
  counterfactuals=[]
  for threshold in (.50,.70,.80,.90,.95,.99):
   accept=disagree&verified&(z['confidence']>=threshold)
   after=np.where(accept,z['candidate_class'],z['old'])
   counterfactuals.append(dict(threshold=threshold,accepted=int(accept.sum()),clean_teacher_agreement=paired(z['ref'],allrows,after),L_s_label_accuracy=paired(labels[fixed],labeled,after)))
  results.append(dict(view=view,counterfactuals=counterfactuals,accepted=int(z['accepted'].sum()),proposed_disagreement=int(disagree.sum()),
    confident_disagreement=int((disagree&confident).sum()),verified_disagreement=int((disagree&verified).sum()),
    clean_teacher_agreement=paired(z['ref'],allrows),L_s_label_accuracy=paired(labels[fixed],labeled)))
print(json.dumps(dict(status='SOURCE_ONLY_DIAGNOSTIC',checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
 source_records=512,source_views=[1,20],source_truth_labeled=128,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 interpretation='Source-only counterfactual thresholds; same source training records; no change to fixed run; clean-teacher agreement is not TX accuracy.',results=results),ensure_ascii=False))
"""
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner],cwd=release,env=env,check=True,timeout=240)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=260)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
    path=Path(__file__).parent/'phase1_class_hypothesis_epoch01_source_thresholds.json'
    with path.open('x',encoding='utf8') as f:f.write(stdout)
raise SystemExit(p.returncode)

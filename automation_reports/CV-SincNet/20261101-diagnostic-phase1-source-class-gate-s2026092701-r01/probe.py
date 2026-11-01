"""Fit a source-only reliability gate; held-out IDs are held out only from the gate."""
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
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=450)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,class_hypothesis_experiment as b
from experiments.pamf_phase1_flow.feature_conditioned_experiment import source_labels
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
checkpoint=root/'runs/20261101-phase1-class-hypothesis-E30-s2026092701-r01/flow/checkpoints/epoch-14.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert (state['epoch'],state['updates'],state['contract']['commit'])==(14,6902,'54e93c9b8ea1f6419d94a161cbcea5ffa63f444b')
model=b.make_model('cpu');model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,1024),('U_s',56700,3072))])
assert len(fixed)==len(np.unique(fixed))==4096
order=np.random.default_rng(2026110140).permutation(len(fixed));split=np.empty(len(fixed),dtype=np.int64)
split[order[:2456]]=0;split[order[2456:3276]]=1;split[order[3276:]]=2
assert [int((split==i).sum()) for i in range(3)]==[2456,820,820]
records={}
with torch.no_grad():
 for view in ('clean','01','20'):
  if view!='clean':raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
  obs={k:[] for k in ('features','base','q','confidence')}
  for start in range(0,len(fixed),32):
   ix=fixed[start:start+32];x=core.tensor(clean[ix] if view=='clean' else raw[ix],'cpu');anchor=dura(x)
   feat=teacher.features(anchor);base_logits=teacher.classify_features(feat);q_logits,candidate=model(x,anchor,feat,base_logits)
   generated=teacher.classify_features(teacher.features(candidate));pb=F.softmax(base_logits,1);pq=F.softmax(q_logits,1);pg=F.softmax(generated,1)
   base=pb.argmax(1);confidence,q=pq.max(1)
   energy_x=x.square().mean((1,2)).clamp_min(1e-8);energy_a=anchor.square().mean((1,2)).clamp_min(1e-8)
   verified=(pg.argmax(1)==q).to(pb.dtype)
   features=torch.cat((pb,pq,pg,energy_x.log()[:,None],energy_a.log()[:,None],verified[:,None]),1)
   for key,value in [('features',features),('base',base),('q',q),('confidence',confidence)]:obs[key].append(value.cpu())
  records[view]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
  print('SOURCE_GATE_INPUTS_FIXED',view,flush=True)
# Clean teacher and legal L_s labels are accessed only after the inference inputs and proposals are fixed.
clean_scores=[]
with torch.no_grad():
 for start in range(0,len(fixed),32):
  scores=teacher.classify_features(teacher.features(core.tensor(clean[fixed[start:start+32]],'cpu')))
  clean_scores.append(F.softmax(scores,1).cpu())
clean_probs=np.from_dlpack(torch.cat(clean_scores));ref=clean_probs.argmax(1);ref_conf=clean_probs.max(1)
labels=source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s'
assert np.all(labels[fixed[1024:]]==-1)
target=np.where(labeled,labels[fixed],ref);valid=labeled|(ref_conf>=.95)
features=np.concatenate([records[v]['features'] for v in ('01','20')]);base=np.concatenate([records[v]['base'] for v in ('01','20')]);q=np.concatenate([records[v]['q'] for v in ('01','20')]);conf=np.concatenate([records[v]['confidence'] for v in ('01','20')])
split2=np.tile(split,2);target2=np.tile(target,2);valid2=np.tile(valid,2);eligible=(q!=base)&(conf>=.5)
train=(split2==0)&eligible&valid2
assert train.sum()>=30 and np.unique(q[train]==target2[train]).size==2
mean=features[train].mean(0);std=features[train].std(0).clip(min=1e-4)
def design(x):return np.column_stack((np.ones(len(x)),np.clip((x-mean)/std,-10,10))).astype(np.float64)
x=design(features[train]);truth=(q[train]==target2[train]).astype(np.float64);weight=np.zeros(x.shape[1]);penalty=np.eye(x.shape[1]);penalty[0,0]=0
for _ in range(40):
 z=np.clip(x@weight,-40,40);prob=1/(1+np.exp(-z));grad=x.T@(prob-truth)+weight@penalty
 hess=x.T@((prob*(1-prob))[:,None]*x)+penalty+np.eye(x.shape[1])*1e-8
 delta=np.linalg.solve(hess,grad);weight-=delta
 if np.linalg.norm(delta)<1e-8:break
def risk_prob(feat):
 z=np.clip(design(feat)@weight,-40,40);return 1/(1+np.exp(-z))
prob=risk_prob(features);clean_z=records['clean'];clean_prob=risk_prob(clean_z['features'])
thresholds=(.5,.6,.7,.8,.9,.95);validation=[];best={'kind':'identity','validation_net':0};chosen=None
for threshold in thresholds:
 gate=eligible&(prob>=threshold);mask=(split2==1)&valid2
 benefit=((q==target2).astype(np.int64)-(base==target2).astype(np.int64))
 net=int((benefit*gate*mask).sum());accepted=int((gate&mask).sum())
 clean_gate=(split==1)&(clean_z['q']!=clean_z['base'])&(clean_z['confidence']>=.5)&(clean_prob>=threshold)
 clean_delta=((clean_z['q']==ref).astype(np.int64)-(clean_z['base']==ref).astype(np.int64))
 clean_net=int((clean_delta*clean_gate).sum());clean_count=int((split==1).sum())
 validation.append(dict(threshold=threshold,accepted=accepted,net=net,clean_net=clean_net,clean_count=clean_count))
 if net>best['validation_net'] and clean_net>=-.01*clean_count:
  best=dict(kind='source_gate',threshold=threshold,validation_net=net);chosen=threshold
def outcome(mask,accepted,before_pred,after_pred,reference,labeled_mask):
 before=before_pred==reference;new=after_pred==reference;lm=mask&labeled_mask
 return dict(valid_records=int(mask.sum()),accepted=int((mask&accepted).sum()),
  before=int((mask&before).sum()),after=int((mask&new).sum()),repairs=int((mask&~before&new).sum()),damages=int((mask&before&~new).sum()),
  L_s_reference_count=int(lm.sum()),L_s_reference_before=int((lm&before).sum()),L_s_reference_after=int((lm&new).sum()))
out=[]
for i,view in enumerate(('01','20')):
 z=records[view];sl=slice(i*len(fixed),(i+1)*len(fixed));view_prob=prob[sl]
 mask=(split==2)&valid;accepted=mask&(z['q']!=z['base'])&(z['confidence']>=.5)&(view_prob>=chosen) if chosen is not None else np.zeros(len(fixed),dtype=bool)
 verified=accepted&(z['features'][:,-1]>.5)
 counter=outcome(mask,accepted,z['base'],np.where(accepted,z['q'],z['base']),target,labeled)
 generated=outcome(mask,verified,z['base'],np.where(verified,z['q'],z['base']),target,labeled)
 out.append(dict(view=int(view),counterfactual=counter,verified_generated=generated,teacher_confident_U_count=int((mask&~labeled).sum())))
clean_mask=split==2
clean_accepted=clean_mask&(clean_z['q']!=clean_z['base'])&(clean_z['confidence']>=.5)&(clean_prob>=chosen) if chosen is not None else np.zeros(len(fixed),dtype=bool)
clean_verified=clean_accepted&(clean_z['features'][:,-1]>.5)
clean_holdout=dict(counterfactual=outcome(clean_mask,clean_accepted,clean_z['base'],np.where(clean_accepted,clean_z['q'],clean_z['base']),ref,labeled),
 verified_generated=outcome(clean_mask,clean_verified,clean_z['base'],np.where(clean_verified,clean_z['q'],clean_z['base']),ref,labeled))
clean_holdout['L_s_truth_counterfactual']=outcome(clean_mask&labeled,clean_accepted,clean_z['base'],np.where(clean_accepted,clean_z['q'],clean_z['base']),labels[fixed],labeled)
clean_holdout['L_s_truth_generated']=outcome(clean_mask&labeled,clean_verified,clean_z['base'],np.where(clean_verified,clean_z['q'],clean_z['base']),labels[fixed],labeled)
print('GATE_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_HELDOUT_GATE_PROBE',checkpoint_epoch=14,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 source_ids=4096,train_ids=2456,validation_ids=820,heldout_ids=820,source_L_s_ids=1024,source_U_s_ids=3072,
 train_proposals=int(train.sum()),train_positive=int((train&(q==target2)).sum()),validation=validation,selected=best,heldout=out,clean_holdout=clean_holdout,
 calibrated_gate=dict(mean=mean.tolist(),std=std.tolist(),weights=weight.tolist()) if chosen is not None else None,
 source_L_s_labels_materialized=6300,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,truth_read_after_proposals=True,
 caveat='Verified-generated decisions use actual parent candidates; other switches are counterfactual. Generated and clean holdout analyses were added after observing initial counterfactual results. Held out from gate fitting, but epoch-14 parent saw source records. U_s targets are high-confidence clean-teacher pseudo-labels.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=470)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(s[len('GATE_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('GATE_SOURCE_PROBE '))
 path=Path(__file__).parent/'phase1_class_gate_source_probe_v2.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)

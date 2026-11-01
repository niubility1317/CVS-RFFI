"""Recompute frozen source gate decisions without refitting the gate or parent."""
import json,subprocess
from pathlib import Path

result=json.loads((Path(__file__).parent/'phase1_class_gate_source_probe_v2.json').read_text(encoding='utf8'))
assert result['selected']['kind']=='source_gate' and result['calibrated_gate'] is not None
payload={k:result[k] for k in ('selected','calibrated_gate','heldout','clean_holdout','checkpoint_sha256')}
remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-class-hypothesis-E30-s2026092701-r01-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='54e93c9b8ea1f6419d94a161cbcea5ffa63f444b'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=180)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,class_hypothesis_experiment as b
from experiments.pamf_phase1_flow.feature_conditioned_experiment import source_labels
expected=json.loads(__PAYLOAD__);root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
path=root/'runs/20261101-phase1-class-hypothesis-E30-s2026092701-r01/flow/checkpoints/epoch-14.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['checkpoint_sha256']
state=torch.load(path,map_location='cpu',weights_only=False);assert (state['epoch'],state['updates'])==(14,6902)
model=b.make_model('cpu');model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,1024),('U_s',56700,3072))])
order=np.random.default_rng(2026110140).permutation(len(fixed));ix=fixed[np.sort(order[3276:])];assert len(ix)==820
gate=expected['calibrated_gate'];mean=np.array(gate['mean']);std=np.array(gate['std']);weight=np.array(gate['weights']);threshold=expected['selected']['threshold']
records={}
with torch.no_grad():
 for view in ('clean','01','20'):
  if view!='clean':raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
  obs={k:[] for k in ('features','base','q','confidence')}
  for start in range(0,len(ix),32):
   batch=ix[start:start+32];x=core.tensor(clean[batch] if view=='clean' else raw[batch],'cpu');anchor=dura(x)
   feat=teacher.features(anchor);base_logits=teacher.classify_features(feat);q_logits,candidate=model(x,anchor,feat,base_logits)
   generated=teacher.classify_features(teacher.features(candidate));pb=F.softmax(base_logits,1);pq=F.softmax(q_logits,1);pg=F.softmax(generated,1)
   base=pb.argmax(1);confidence,q=pq.max(1);ex=x.square().mean((1,2)).clamp_min(1e-8);ea=anchor.square().mean((1,2)).clamp_min(1e-8)
   verified=(pg.argmax(1)==q).to(pb.dtype);features=torch.cat((pb,pq,pg,ex.log()[:,None],ea.log()[:,None],verified[:,None]),1)
   for key,value in [('features',features),('base',base),('q',q),('confidence',confidence)]:obs[key].append(value.cpu())
  records[view]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
 print('FROZEN_GATE_DECISIONS_FIXED',flush=True)
# Only score after all frozen decisions are computed.
refs=[]
with torch.no_grad():
 for start in range(0,len(ix),32):
  scores=teacher.classify_features(teacher.features(core.tensor(clean[ix[start:start+32]],'cpu')))
  refs.append(F.softmax(scores,1).cpu())
ref_prob=np.from_dlpack(torch.cat(refs));ref=ref_prob.argmax(1);labels=source_labels(root,cache,clean,roles,contract);labeled=roles[ix]=='L_s'
assert np.all(labels[ix[~labeled]]==-1) and int(labeled.sum())==205
target=np.where(labeled,labels[ix],ref);valid=labeled|(ref_prob.max(1)>=.95)
def outcome(mask,accepted,before_pred,after_pred,reference):
 old=before_pred==reference;new=after_pred==reference;lm=mask&labeled
 return dict(valid_records=int(mask.sum()),accepted=int((mask&accepted).sum()),before=int((mask&old).sum()),after=int((mask&new).sum()),
  repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_reference_count=int(lm.sum()),
  L_s_reference_before=int((lm&old).sum()),L_s_reference_after=int((lm&new).sum()))
for view in ('clean','01','20'):
 z=records[view];design=np.column_stack((np.ones(len(ix)),np.clip((z['features']-mean)/std,-10,10))).astype(np.float64)
 score=1/(1+np.exp(-np.clip(design@weight,-40,40)));accept=(z['q']!=z['base'])&(z['confidence']>=.5)&(score>=threshold)
 verified=accept&(z['features'][:,-1]>.5);mask=np.ones(len(ix),dtype=bool) if view=='clean' else valid;reference=ref if view=='clean' else target
 counter=outcome(mask,accept,z['base'],np.where(accept,z['q'],z['base']),reference)
 generated=outcome(mask,verified,z['base'],np.where(verified,z['q'],z['base']),reference)
 exp=expected['clean_holdout'] if view=='clean' else next(row for row in expected['heldout'] if row['view']==int(view))
 assert counter==exp['counterfactual'] and generated==exp['verified_generated']
 if view=='clean':
  actual_counter=outcome(labeled,accept,z['base'],np.where(accept,z['q'],z['base']),labels[ix])
  actual_generated=outcome(labeled,verified,z['base'],np.where(verified,z['q'],z['base']),labels[ix])
  assert actual_counter==exp['L_s_truth_counterfactual'] and actual_generated==exp['L_s_truth_generated']
 print('VERIFIED_FROZEN_GATE_SOURCE_VIEW',view,'counterfactual',counter['before'],counter['after'],'generated',generated['before'],generated['after'],flush=True)
print('VERIFIED_HELDOUT_SOURCE_WITHOUT_REFITTING_GATE',flush=True)
'''.replace('__PAYLOAD__',repr(json.dumps(payload)))
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=200)
print(p.stdout.decode('utf8',errors='replace'));print(p.stderr.decode('utf8',errors='replace'));raise SystemExit(p.returncode)

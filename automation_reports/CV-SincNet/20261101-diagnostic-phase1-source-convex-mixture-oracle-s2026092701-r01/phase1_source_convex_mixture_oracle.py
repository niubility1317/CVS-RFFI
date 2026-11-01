"""Privileged source-only convex mixture opportunity; not deployable inference."""
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
r=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261101-phase1-physical-rich-router-s2026092701-r01'
release=r/'releases'/(run+'-release');native=r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',__INNER_SCRIPT__],cwd=release,env=env,check=True,timeout=1200)
'''
inner=r'''
from pathlib import Path
import json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,boundary,feature,_,_,_=models
cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
groups=[]
for role,count,initial,n in (('L_s',6300,128,1024),('U_s',56700,384,3072)):
 available=np.flatnonzero(roles==role);prior=available[np.linspace(0,count-1,initial,dtype=int)];available=np.setdiff1d(available,prior,assume_unique=True)
 for _ in range(3):
  old=available[np.linspace(0,len(available)-1,n,dtype=int)];available=np.setdiff1d(available,old,assume_unique=True)
 take=128 if role=='L_s' else 384;prior=available[np.linspace(0,len(available)-1,take,dtype=int)];available=np.setdiff1d(available,prior,assume_unique=True)
 for _ in range(2):
  old=available[np.linspace(0,len(available)-1,n,dtype=int)];available=np.setdiff1d(available,old,assume_unique=True)
 take=64 if role=='L_s' else 192;groups.append(available[np.linspace(0,len(available)-1,take,dtype=int)])
ix=np.concatenate(groups);assert len(ix)==len(np.unique(ix))==256
scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-convex-oracle-2026110211';views={}
with torch.no_grad():
 for view in ('clean',)+scenes:
  if view=='clean':received=np.asarray(clean[ix],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[]
   for j in ix:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110212,session)
    value=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(value[0]+1j*value[1])
    result,meta=ResidualChannel(cfg,stable_seed(2026110211,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied']
    received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  batches=[];probabilities=[]
  for start in range(0,len(ix),32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x);feat=teacher.features(anchor);base=teacher.classify_features(feat)
   outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,base));signals=torch.stack(outputs,dim=1)
   probs=torch.stack([F.softmax(teacher.classify_features(teacher.features(y)),1) for y in outputs],dim=1)
   batches.append(signals.cpu());probabilities.append(probs.cpu())
  views[view]=dict(signals=torch.cat(batches),probs=torch.cat(probabilities))
  print('SOURCE_CONVEX_EXPERTS_FIXED',view,flush=True)
print('ALL_SOURCE_EXPERTS_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[ix]=='L_s';assert np.all(labels[ix[~labeled]]==-1)
ref=np.from_dlpack(views['clean']['probs'][:,0,:]);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95)
target=np.where(labeled,labels[ix],teacher_target);results=[]
for view in scenes:
 signals=views[view]['signals'];probs=np.from_dlpack(views[view]['probs']);pred=probs.argmax(2);truth=target
 old=pred[:,1]==truth;expert_oracle=(pred==truth[:,None]).any(1);best=old.copy();best_weights=np.zeros((len(ix),4),np.float32);best_weights[:,1]=1.
 for start in range(0,len(ix),16):
  end=min(start+16,len(ix));batch=signals[start:end].detach();y=torch.tensor(truth[start:end].tolist(),dtype=torch.long)
  mask=torch.tensor(valid[start:end].tolist(),dtype=torch.bool)
  if not bool(mask.any()):continue
  u=torch.zeros((len(batch),4),dtype=torch.float32);u[:,1]=2.;u.requires_grad_(True);optimizer=torch.optim.Adam([u],lr=.2)
  for step in range(31):
   weights=F.softmax(u,dim=1);mixed=(batch*weights[:,:,None,None]).sum(1)
   logits=teacher.classify_features(teacher.features(mixed));assert torch.isfinite(logits).all()
   now=np.from_dlpack(logits.detach().argmax(1).cpu());new=(now==truth[start:end])&valid[start:end]&~best[start:end]
   if new.any():
    best[start:end][new]=True;w=np.from_dlpack(weights.detach().cpu().contiguous());best_weights[start:end][new]=w[new]
   if step==30:break
   loss=F.cross_entropy(logits[mask],y[mask]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
 assert np.all(best_weights>=0) and np.allclose(best_weights.sum(1),1,atol=1e-5)
 row=dict(view=view,valid=int(valid.sum()),labeled=int((valid&labeled).sum()),pseudo=int((valid&~labeled).sum()),
  dura=int((valid&old).sum()),four_expert_oracle=int((valid&expert_oracle).sum()),
  convex_oracle=int((valid&best).sum()),convex_new_beyond_four=int((valid&best&~expert_oracle).sum()),
  convex_repairs=int((valid&~old&best).sum()),labeled_dura=int((valid&labeled&old).sum()),
  labeled_convex=int((valid&labeled&best).sum()))
 results.append(row);print('PRIVILEGED_SOURCE_CONVEX_RESULT',json.dumps(row),flush=True)
print('SOURCE_CONVEX_ORACLE '+json.dumps(dict(status='PRIVILEGED_SOURCE_ONLY_CONVEX_ORACLE',source_ids=256,results=results,
 labels='L_s actual; U_s high-confidence clean-teacher pseudo-targets',steps=30,optimizer='Adam lr 0.2; per-record convex expert weights',
 U_s_labels_read=0,V_reads=0,test_IQ_reads=0,expert_outputs_before_source_labels=True,
 caveat='Per-record optimization and best-iterate selection use source targets and are not deployable. Simulator is not validated real satellite data; source experts saw source records. An observed Phase1 test oracle motivated this exploratory method.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=1220)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(s[len('SOURCE_CONVEX_ORACLE '):] for s in stdout.splitlines() if s.startswith('SOURCE_CONVEX_ORACLE '))
 path=Path(__file__).parent/'phase1_source_convex_mixture_oracle.json'
 if path.exists():
  import json
  assert json.loads(path.read_text(encoding='utf8'))==json.loads(line)
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)

"""Train a source-only convex waveform mixer; validate on disjoint physical IDs."""
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
r=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261101-phase1-physical-rich-router-s2026092701-r01'
release=r/'releases'/(run+'-release');native=r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',__INNER_SCRIPT__],cwd=release,env=env,check=True,timeout=3600)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch import nn
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
 take=64 if role=='L_s' else 192;prior=available[np.linspace(0,len(available)-1,take,dtype=int)];available=np.setdiff1d(available,prior,assume_unique=True)
 take=512 if role=='L_s' else 1536;groups.append(available[np.linspace(0,len(available)-1,take,dtype=int)])
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==2048
order=np.random.default_rng(2026110221).permutation(len(fixed));split=np.empty(len(fixed),dtype=np.int64)
split[order[:1228]]=0;split[order[1228:1638]]=1;split[order[1638:]]=2
assert [int((split==i).sum()) for i in range(3)]==[1228,410,410]
view_names=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-convex-student-2026110222';views={}
with torch.no_grad():
 for view in view_names:
  if view=='clean':received=np.asarray(clean[fixed],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[]
   for j in fixed:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110225,session)
    v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110222,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  obs={k:[] for k in ('signals','probs','features')}
  for start in range(0,len(fixed),32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x);fa=teacher.features(anchor);base=teacher.classify_features(fa)
   outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,fa,base));signals=torch.stack(outputs,dim=1)
   rawfeat=teacher.features(x);logits=(teacher.classify_features(rawfeat),base,teacher.classify_features(teacher.features(outputs[2])),teacher.classify_features(teacher.features(outputs[3])))
   probs=torch.stack([F.softmax(z,dim=1) for z in logits],dim=1)
   energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),1)
   def spectrum(z):
    wave=torch.complex(z[:,0,:],z[:,1,:]);power=torch.fft.fft(wave,dim=-1).abs().square().reshape(len(z),16,16).mean(-1)
    return power/power.sum(1,keepdim=True).clamp_min(1e-8)
   feats=torch.cat((probs.reshape(len(x),24),energy,F.normalize(rawfeat,dim=1),F.normalize(fa,dim=1),spectrum(x),spectrum(x-anchor)),1)
   assert feats.shape==(len(x),378) and torch.isfinite(feats).all()
   obs['signals'].append(signals.cpu());obs['probs'].append(probs.cpu());obs['features'].append(feats.cpu())
  views[view]={k:torch.cat(v) for k,v in obs.items()};print('CONVEX_STUDENT_SOURCE_EXPERTS_FIXED',view,flush=True)
print('ALL_SOURCE_EXPERTS_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[512:]]==-1)
clean_probs=np.from_dlpack(views['clean']['probs'][:,0,:]);teacher_ref=clean_probs.argmax(1);confidence=clean_probs.max(1)
target=np.where(labeled,labels[fixed],teacher_ref);valid=labeled|(confidence>=.95)
n=len(fixed);signals=torch.cat([views[v]['signals'] for v in view_names]);probs=np.from_dlpack(torch.cat([views[v]['probs'] for v in view_names]));pred=probs.argmax(2)
features=np.from_dlpack(torch.cat([views[v]['features'] for v in view_names]));all_target=np.tile(target,len(view_names));all_valid=np.tile(valid,len(view_names));all_split=np.tile(split,len(view_names))
baseline_correct=pred[:,1]==all_target;train=(all_split==0)&all_valid;mean=features[train].mean(0);std=features[train].std(0).clip(min=1e-4)
scaled=np.clip((features-mean)/std,-10,10).astype(np.float32);base_tensor=core.tensor(scaled,'cpu');target_t=torch.tensor(all_target.tolist(),dtype=torch.long)
weights_t=torch.tensor(np.where(baseline_correct,1.,4.).astype(np.float32).tolist(),dtype=torch.float32)
torch.manual_seed(2026110223);student=nn.Sequential(nn.Linear(378,96),nn.GELU(),nn.Linear(96,4))
nn.init.zeros_(student[-1].weight);nn.init.zeros_(student[-1].bias)
with torch.no_grad():student[-1].bias[1]=6.
optimizer=torch.optim.AdamW(student.parameters(),lr=.0005,weight_decay=.0001);rng=np.random.default_rng(2026110224);train_ix=np.flatnonzero(train)
thresholds=(.25,.5,1.);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def evaluate(model,alpha,part):
 predictions=[]
 with torch.no_grad():
  for start in range(0,len(signals),128):
   b=signals[start:start+128];w=F.softmax(model(base_tensor[start:start+128]),1);mix=(b*w[:,:,None,None]).sum(1)
   y=b[:,1]+alpha*(mix-b[:,1]);logits=teacher.classify_features(teacher.features(y));predictions.append(logits.argmax(1).cpu())
 output=np.from_dlpack(torch.cat(predictions));per={};mask=(split==part)&valid
 for i,name in enumerate(view_names):
  sl=slice(i*n,(i+1)*n);old=pred[sl,1]==target;new=output[sl]==target;lm=mask&labeled
  per[name]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),
   baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),
   L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
 clean_out=output[:n];clean_base=pred[:n,1];cm=split==part
 clean_net=int((((clean_out==teacher_ref).astype(np.int64)-(clean_base==teacher_ref).astype(np.int64))*cm).sum())
 return per,clean_net
for epoch in range(1,21):
 permutation=rng.permutation(train_ix)
 for start in range(0,len(permutation),32):
  index=torch.tensor(permutation[start:start+32].tolist(),dtype=torch.long);b=signals[index];w=F.softmax(student(base_tensor[index]),dim=1)
  mixed=(b*w[:,:,None,None]).sum(1);logits=teacher.classify_features(teacher.features(mixed))
  per=F.cross_entropy(logits,target_t[index],reduction='none');wgt=weights_t[index];loss=(per*wgt).sum()/wgt.sum().clamp_min(1)+.05*(1-w[:,1]).mean()
  if not torch.isfinite(loss):raise FloatingPointError('Finite source mixture loss required')
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);optimizer.step()
 if epoch in (2,5,10,20):
  for alpha in thresholds:
   per,clean_net=evaluate(student,alpha,1);changes={v:per[v]['student']-per[v]['baseline'] for v in view_names[1:]};net=sum(changes.values())
   validation.append(dict(epoch=epoch,alpha=alpha,net=net,per_view=changes,clean_teacher_net=clean_net,clean_count=410))
   if net>best['validation_net'] and clean_net>=-4.1 and min(changes.values())>=-2:
    best=dict(kind='convex_student',epoch=epoch,alpha=alpha,validation_net=net);best_state={k:v.detach().clone() for k,v in student.state_dict().items()}
  print('CONVEX_STUDENT_VALIDATION_COMPLETE',epoch,flush=True)
if best_state is not None:
 student.load_state_dict(best_state);holdout,clean_net=evaluate(student,best['alpha'],2)
 state={k:v.detach().tolist() for k,v in student.state_dict().items()};m=mean.tolist();sd=std.tolist()
else:
 holdout=None;clean_net=None;state=None;m=None;sd=None
print('CONVEX_STUDENT_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_CONVEX_STUDENT_PROBE',source_ids=2048,train_ids=1228,validation_ids=410,heldout_ids=410,
 source_L_s_ids=512,source_U_s_ids=1536,excluded_previous_source_ids=21760,training_valid_views=int(train.sum()),feature_dim=378,
 validation=validation,selected=best,heldout=holdout,heldout_clean_teacher_net=clean_net,student_state=state,feature_mean=m,feature_std=sd,
 boundary_checkpoint_sha256=g.sha(root/'runs'/g.BOUNDARY_RUN/'flow/final.pt'),feature_checkpoint_sha256=g.sha(root/'runs'/g.FEATURE_RUN/'flow/final.pt'),
 U_s_labels_read=0,V_reads=0,test_IQ_reads=0,expert_outputs_before_source_labels=True,
 caveat='Source holdout is only for mixer; fixed experts saw source records. Physical simulator is not validated real satellite data. U_s uses confident clean-teacher pseudo-labels. This direction follows an already observed Phase1 test expert-oracle analysis.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=3620)
stdout=p.stdout.decode('utf8',errors='replace');stderr=p.stderr.decode('utf8',errors='replace')
for line in stdout.splitlines():
 if not line.startswith('CONVEX_STUDENT_SOURCE_PROBE '):print(line)
print(stderr)
if p.returncode==0:
 line=next(s[len('CONVEX_STUDENT_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('CONVEX_STUDENT_SOURCE_PROBE '))
 path=Path(__file__).parent/'phase1_source_convex_student_probe.json'
 with path.open('x',encoding='utf8') as f:f.write(line+'\n')
 print('SAVED_SOURCE_CONVEX_STUDENT_RESULT',path)
raise SystemExit(p.returncode)

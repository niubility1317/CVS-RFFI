"""Source-only pairwise repair-vs-damage utility on physical channels."""
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
import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,boundary,feature,_,_,_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
rng=np.random.default_rng(2026110241);groups=[]
for role,take in (('L_s',2048),('U_s',6144)):
 available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==8192
order=np.random.default_rng(2026110242).permutation(len(fixed));split=np.empty(len(fixed),dtype=np.int64)
split[order[:4912]]=0;split[order[4912:6552]]=1;split[order[6552:]]=2
assert [int((split==i).sum()) for i in range(3)]==[4912,1640,1640]
view_names=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-switch-utility-2026110243';views={}
with torch.no_grad():
 for view in view_names:
  if view=='clean':received=np.asarray(clean[fixed],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[]
   for j in fixed:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110244,session)
    v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110243,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  obs={k:[] for k in ('probs','features')}
  for start in range(0,len(fixed),32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x);fa=teacher.features(anchor);base=teacher.classify_features(fa)
   outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,fa,base));emb=(teacher.features(outputs[0]),fa,teacher.features(outputs[2]),teacher.features(outputs[3]))
   probs=torch.stack([F.softmax(teacher.classify_features(z),dim=1) for z in emb],dim=1)
   energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),1)
   def spectrum(z):
    wave=torch.complex(z[:,0,:],z[:,1,:]);power=torch.fft.fft(wave,dim=-1).abs().square().reshape(len(z),16,16).mean(-1)
    return power/power.sum(1,keepdim=True).clamp_min(1e-8)
   feats=torch.cat((probs.reshape(len(x),24),energy,*[F.normalize(z,dim=1) for z in emb],spectrum(x),spectrum(x-anchor)),1)
   assert feats.shape==(len(x),698) and torch.isfinite(feats).all()
   obs['probs'].append(probs.cpu());obs['features'].append(feats.cpu())
  views[view]={k:torch.cat(v) for k,v in obs.items()};print('UTILITY_SOURCE_EXPERTS_FIXED',view,flush=True)
print('ALL_SOURCE_EXPERTS_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[2048:]]==-1)
ref=np.from_dlpack(views['clean']['probs'][:,0,:]);teacher_ref=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_ref)
n=len(fixed);probs=np.from_dlpack(torch.cat([views[v]['probs'] for v in view_names]));pred=probs.argmax(2)
features=np.from_dlpack(torch.cat([views[v]['features'] for v in view_names]));all_target=np.tile(target,len(view_names));all_valid=np.tile(valid,len(view_names));all_split=np.tile(split,len(view_names))

base_correct=pred[:,1]==all_target;raw_correct=pred[:,0]==all_target;feature_correct=pred[:,3]==all_target
outcomes=np.stack(((~base_correct)&raw_correct,base_correct&(~raw_correct),(~base_correct)&feature_correct,base_correct&(~feature_correct)),1).astype(np.float32)
diff_raw=pred[:,0]!=pred[:,1];diff_feature=pred[:,3]!=pred[:,1]
masks=np.stack((diff_raw,diff_raw,diff_feature,diff_feature),1).astype(np.float32)
train=(all_split==0)&all_valid;assert int((train&diff_raw).sum())>100 and int((train&diff_feature).sum())>100
mean=features[train].mean(0);std=features[train].std(0).clip(min=1e-4);scaled=np.clip((features-mean)/std,-10,10).astype(np.float32)
base_tensor=core.tensor(scaled,'cpu');target_t=core.tensor(outcomes,'cpu');mask_t=core.tensor(masks,'cpu')
torch.manual_seed(2026110245);model=nn.Sequential(nn.Linear(698,128),nn.GELU(),nn.Linear(128,4))
optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);rng=np.random.default_rng(2026110246);train_ix=np.flatnonzero(train)
thresholds=(0.,.005,.01,.02,.04,.08,.12,.2);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def evaluate(p,threshold,part):
 utility=np.stack((p[:,0]-p[:,1],p[:,2]-p[:,3]),1);utility[~diff_raw,0]=-np.inf;utility[~diff_feature,1]=-np.inf
 selected=utility.argmax(1);scores=utility[np.arange(len(utility)),selected];choice=np.where(scores>threshold,np.where(selected==0,0,3),1)
 output=pred[np.arange(len(pred)),choice];per={};mask=(split==part)&valid
 for i,view in enumerate(view_names):
  sl=slice(i*n,(i+1)*n);o=pred[sl,1]==target;z=output[sl]==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),
   baseline=int((mask&o).sum()),student=int((mask&z).sum()),repairs=int((mask&~o&z).sum()),damages=int((mask&o&~z).sum()),
   L_s_baseline=int((lm&o).sum()),L_s_student=int((lm&z).sum()),raw_choices=int((mask&(choice[sl]==0)).sum()),feature_choices=int((mask&(choice[sl]==3)).sum()))
 cm=split==part;clean_out=output[:n];clean_base=pred[:n,1]
 clean_net=int((((clean_out==teacher_ref).astype(np.int64)-(clean_base==teacher_ref).astype(np.int64))*cm).sum())
 return per,clean_net
for epoch in range(1,31):
 perm=rng.permutation(train_ix)
 for start in range(0,len(perm),128):
  ix=torch.tensor(perm[start:start+128].tolist(),dtype=torch.long);logits=model(base_tensor[ix]);mask=mask_t[ix]
  loss=(F.binary_cross_entropy_with_logits(logits,target_t[ix],reduction='none')*mask).sum()/mask.sum().clamp_min(1)
  if not torch.isfinite(loss):raise FloatingPointError('Finite source utility loss required')
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
 if epoch in (2,5,10,20,30):
  with torch.no_grad():prob=np.from_dlpack(torch.sigmoid(model(base_tensor)).cpu().contiguous())
  for threshold in thresholds:
   per,clean_net=evaluate(prob,threshold,1);changes={v:per[v]['student']-per[v]['baseline'] for v in view_names[1:]};net=sum(changes.values())
   validation.append(dict(epoch=epoch,threshold=threshold,net=net,per_view=changes,clean_teacher_net=clean_net,clean_count=1640))
   if net>best['validation_net'] and clean_net>=-16.4 and min(changes.values())>=-8:
    best=dict(kind='multi_utility',epoch=epoch,threshold=threshold,validation_net=net);best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  print('MULTI_UTILITY_SOURCE_VALIDATION_COMPLETE',epoch,flush=True)
if best_state is not None:
 model.load_state_dict(best_state)
 with torch.no_grad():prob=np.from_dlpack(torch.sigmoid(model(base_tensor)).cpu().contiguous())
 heldout,clean_net=evaluate(prob,best['threshold'],2);state={k:v.detach().tolist() for k,v in model.state_dict().items()};m=mean.tolist();sd=std.tolist()
else:heldout=None;clean_net=None;state=None;m=None;sd=None
print('MULTI_UTILITY_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_MULTI_UTILITY_PROBE',source_ids=8192,train_ids=4912,validation_ids=1640,heldout_ids=1640,
 source_L_s_ids=2048,source_U_s_ids=6144,feature_dim=698,training_valid_views=int(train.sum()),
 training_raw_disagreements=int((train&diff_raw).sum()),training_feature_disagreements=int((train&diff_feature).sum()),
 training_outcome_counts=[int((outcomes[:,i]*train).sum()) for i in range(4)],validation=validation,selected=best,heldout=heldout,
 heldout_clean_teacher_net=clean_net,model_state=state,feature_mean=m,feature_std=sd,
 U_s_labels_read=0,V_reads=0,test_IQ_reads=0,expert_outputs_before_source_labels=True,
 caveat='This utility model uses disjoint source training/validation/holdout physical IDs, but IDs may overlap earlier source diagnostics and fixed experts saw source records. U_s uses confident clean-teacher pseudo-labels. Simulator is not validated real satellite data. Previously observed Phase1 test oracle informed the direction.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=3620)
stdout=p.stdout.decode('utf8',errors='replace');stderr=p.stderr.decode('utf8',errors='replace')
for line in stdout.splitlines():
 if not line.startswith('MULTI_UTILITY_SOURCE_PROBE '):print(line)
print(stderr)
if p.returncode==0:
 line=next(s[len('MULTI_UTILITY_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('MULTI_UTILITY_SOURCE_PROBE '))
 path=Path(__file__).parent/'phase1_source_multi_utility_probe.json'
 with path.open('x',encoding='utf8') as f:f.write(line+'\n')
 print('SAVED_SOURCE_UTILITY_RESULT',path)
raise SystemExit(p.returncode)

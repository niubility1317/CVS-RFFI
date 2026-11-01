"""Noise-augmented source router with new physical-ID validation and holdout."""
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
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=1200)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,boundary_student_experiment as bs,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
digest=a.sha(root/a.DURA);assert digest=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
commit='f72f9618859fce30a1983a3b4b10c8c869b2dbba'
bp=root/'runs/20261101-phase1-boundary-student-E30-s2026092701-r02/flow/final.pt'
fp=root/'runs/20261101-phase1-feature-conditioned-E30-s2026092701-r02/flow/final.pt'
boundary=bs.load(bp,'cpu',commit,digest);feature=fs.load(fp,'cpu',commit,digest)
groups=[]
for role,count,old_n,n in (('L_s',6300,128,1024),('U_s',56700,384,3072)):
 available=np.flatnonzero(roles==role);previous=available[np.linspace(0,count-1,old_n,dtype=int)]
 available=np.setdiff1d(available,previous,assume_unique=True); old=available[np.linspace(0,len(available)-1,n,dtype=int)]
 available=np.setdiff1d(available,old,assume_unique=True);groups.append(available[np.linspace(0,len(available)-1,n,dtype=int)])
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096
order=np.random.default_rng(2026110161).permutation(len(fixed));split=np.empty(len(fixed),dtype=np.int64)
split[order[:2456]]=0;split[order[2456:3276]]=1;split[order[3276:]]=2
assert [int((split==i).sum()) for i in range(3)]==[2456,820,820]
experts=('raw','dura','boundary','feature');views={};view_names=('clean','01','01_plus_15dB','01_plus_5dB','20','20_plus_15dB','20_plus_5dB')
view_specs=(('clean',None),('01',None),('01',15),('01',5),('20',None),('20',15),('20',5))
with torch.no_grad():
 for view,snr in view_specs:
  name=view if snr is None else view+'_plus_'+str(snr)+'dB'
  generator=torch.Generator(device='cpu').manual_seed(2026110162+(0 if view=='clean' else int(view)*100)+(snr or 0))
  if view!='clean':raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
  obs={k:[] for k in ('probs','energy')}
  for start in range(0,len(fixed),32):
   ix=fixed[start:start+32];x=core.tensor(clean[ix] if view=='clean' else raw[ix],'cpu')
   if snr is not None:
    rms=x.square().mean((1,2),keepdim=True).sqrt();noise=torch.randn(x.shape,generator=generator,dtype=x.dtype)*rms*(10**(-snr/20));x=x+noise
    x=x*(rms/x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-12))
   anchor=dura(x)
   feat=teacher.features(anchor);base=teacher.classify_features(feat)
   outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,base))
   assert torch.all((outputs[2]-anchor).flatten(1).norm(dim=1)<=.05*anchor.flatten(1).norm(dim=1)+1e-5)
   assert torch.all((outputs[3]-anchor).flatten(1).norm(dim=1)<=.1*anchor.flatten(1).norm(dim=1)+1e-5)
   probs=torch.stack([F.softmax(teacher.classify_features(teacher.features(y)),1) for y in outputs],dim=1)
   energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),dim=1)
   assert torch.isfinite(probs).all() and torch.isfinite(energy).all()
   obs['probs'].append(probs.cpu());obs['energy'].append(energy.cpu())
  views[name]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
  print('SOURCE_EXPERTS_FIXED',name,flush=True)
# Experts and their probabilities are fixed before any source labels are read.
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[1024:]]==-1)
clean_probs=views['clean']['probs'][:,0,:];teacher_ref=clean_probs.argmax(1);confidence=clean_probs.max(1)
target=np.where(labeled,labels[fixed],teacher_ref);valid=labeled|(confidence>=.95)
def features(v):return np.concatenate((v['probs'].reshape(len(fixed),24),v['energy']),axis=1)
all_features=np.concatenate([features(views[v]) for v in view_names]);all_probs=np.concatenate([views[v]['probs'] for v in view_names]);pred=all_probs.argmax(2)
all_target=np.tile(target,len(view_names));all_valid=np.tile(valid,len(view_names));all_split=np.tile(split,len(view_names));correct=pred==all_target[:,None]
baseline_correct=correct[:,1];repairable=(~baseline_correct)&correct.any(1)
dist=np.zeros((len(pred),4),dtype=np.float32);dist[:,1]=1.
dist[repairable]=correct[repairable]/correct[repairable].sum(1,keepdims=True)
weights=np.where(repairable,8.,1.).astype(np.float32);train=(all_split==0)&all_valid
mean=all_features[train].mean(0);std=all_features[train].std(0).clip(min=1e-4)
scaled=np.clip((all_features-mean)/std,-10,10).astype(np.float32)
torch.manual_seed(2026110151);router=nn.Sequential(nn.Linear(26,64),nn.GELU(),nn.Linear(64,4))
nn.init.zeros_(router[-1].weight);nn.init.zeros_(router[-1].bias)
with torch.no_grad():router[-1].bias[1]=2.
optimizer=torch.optim.AdamW(router.parameters(),lr=.001,weight_decay=.0001)
train_ix=np.flatnonzero(train);rng=np.random.default_rng(2026110152)
base_tensor=core.tensor(scaled,'cpu');dist_tensor=core.tensor(dist,'cpu');weight_tensor=core.tensor(weights,'cpu')
thresholds=(.8,.6,.4,.2,0.);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def route(prob,threshold):
 alt=prob.copy();alt[:,1]=-1.;choice=alt.argmax(1);gap=alt[np.arange(len(alt)),choice]-prob[:,1]
 return np.where(gap>=threshold,choice,1)
def evaluate(router,threshold):
 with torch.no_grad():prob=np.from_dlpack(F.softmax(router(base_tensor),1).cpu().contiguous())
 selected=route(prob,threshold);selected_pred=pred[np.arange(len(pred)),selected]
 val=(all_split==1)&all_valid;corrupt=val.copy();corrupt[:len(fixed)]=False
 benefit=(selected_pred==all_target).astype(np.int64)-baseline_correct.astype(np.int64)
 net=int((benefit*corrupt).sum());accepted=int(((selected!=1)&corrupt).sum())
 clean_view=selected_pred[:len(fixed)];clean_base=pred[:len(fixed),1];mask=split==1
 clean_net=int((((clean_view==teacher_ref).astype(np.int64)-(clean_base==teacher_ref).astype(np.int64))*mask).sum())
 return net,accepted,clean_net,prob
for epoch in range(1,21):
 for start in range(0,len(train_ix),256):
  if start==0:permutation=rng.permutation(train_ix)
  index=torch.tensor(permutation[start:start+256].tolist(),dtype=torch.long)
  logits=router(base_tensor[index]);per=-(dist_tensor[index]*F.log_softmax(logits,1)).sum(1)
  loss=(per*weight_tensor[index]).sum()/weight_tensor[index].sum().clamp_min(1)
  if not torch.isfinite(loss):raise FloatingPointError('Finite source router loss required')
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(router.parameters(),1.,error_if_nonfinite=True);optimizer.step()
 if epoch in (2,5,10,20):
  for threshold in thresholds:
   net,accepted,clean_net,_=evaluate(router,threshold)
   validation.append(dict(epoch=epoch,threshold=threshold,net=net,accepted=accepted,clean_net=clean_net,clean_count=820))
   if net>best['validation_net'] and clean_net>=-8.2:
    best=dict(kind='source_router',epoch=epoch,threshold=threshold,validation_net=net)
    best_state={k:v.detach().clone() for k,v in router.state_dict().items()}
  print('ROUTER_VALIDATION_COMPLETE',epoch,flush=True)
if best_state is not None:router.load_state_dict(best_state)
with torch.no_grad():prob=np.from_dlpack(F.softmax(router(base_tensor),1).cpu().contiguous())
selected=np.full(len(pred),1,dtype=np.int64) if best_state is None else route(prob,best['threshold'])
selected_pred=pred[np.arange(len(pred)),selected]
def metrics(view):
 i=view_names.index(view);sl=slice(i*len(fixed),(i+1)*len(fixed));mask=(split==2)&valid
 base=pred[sl,1];out=selected_pred[sl];sel=selected[sl];old=base==target;new=out==target;lm=mask&labeled
 truth_teacher=teacher_ref;ctold=base==truth_teacher;ctnew=out==truth_teacher;cm=split==2
 return dict(view=view,valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),
  expert_choices={name:int((mask&(sel==j)).sum()) for j,name in enumerate(experts)},
  baseline=int((mask&old).sum()),router=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),
  L_s_baseline=int((lm&old).sum()),L_s_router=int((lm&new).sum()),
  clean_teacher_baseline=int((cm&ctold).sum()),clean_teacher_router=int((cm&ctnew).sum()))
state={k:v.detach().tolist() for k,v in router.state_dict().items()} if best_state is not None else None
print('ROUTER_SOURCE_PROBE '+json.dumps(dict(status='SOURCE_ONLY_NOISE_AUGMENTED_ROUTER_PROBE',source_ids=4096,train_ids=2456,validation_ids=820,heldout_ids=820,
 source_L_s_ids=1024,source_U_s_ids=3072,excluded_previous_source_ids=4608,experts=list(experts),
 training_source_valid_views=int(train.sum()),training_repairable_views=int((train&repairable).sum()),validation=validation,selected=best,
 heldout=[metrics(v) for v in view_names],router_state=state,feature_mean=mean.tolist() if state else None,feature_std=std.tolist() if state else None,
 boundary_checkpoint_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),feature_checkpoint_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
 source_L_s_labels_materialized=6300,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,expert_decisions_before_source_labels=True,
 caveat='Source holdout is only for the router; fixed experts saw these records. Added noise is a synthetic stress, not a validated satellite channel. U_s uses confident clean-teacher pseudo-labels. The idea is exploratory after inspecting Phase1 test oracle bounds.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=1220)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(s[len('ROUTER_SOURCE_PROBE '):] for s in stdout.splitlines() if s.startswith('ROUTER_SOURCE_PROBE '))
 path=Path(__file__).parent/'phase1_source_noise_router_probe.json'
 with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)

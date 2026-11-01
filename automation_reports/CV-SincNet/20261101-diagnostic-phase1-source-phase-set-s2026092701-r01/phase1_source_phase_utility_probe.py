"""Source-only repair/damage utility learning for global-phase candidates."""
import subprocess
from pathlib import Path

inner=r'''
from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);rng=np.random.default_rng(2026110371);groups=[]
for role,q_count,h_count,guard_count,take in (('L_s',2048,512,2048,1024),('U_s',6144,1536,6144,3072)):
 available=np.flatnonzero(roles==role);q_ids=q_rng.choice(available,size=q_count,replace=False);h_ids=h_rng.choice(available,size=h_count,replace=False)
 without_h=np.setdiff1d(available,np.sort(h_ids),assume_unique=True);guard_ids=guard_rng.choice(without_h,size=guard_count,replace=False)
 excluded=np.unique(np.concatenate((q_ids,h_ids,guard_ids)));available=np.setdiff1d(available,excluded,assume_unique=True);assert len(available)>=take
 groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096
source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64)
split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0;split[np.isin(source_rx,[4,6])&(source_day==3)]=1;split[(source_rx==8)&(source_day==3)]=2
assert all(int((split==i).sum())>200 for i in range(3))
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-phase-utility-2026110372';phases=np.arange(12,dtype=np.float64)*(2*np.pi/12);data={}
with torch.no_grad():
 for view in views:
  if view=='clean':received=np.asarray(clean[fixed],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
   for j in fixed:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110373,session)
    v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110372,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  buckets={k:[] for k in ('base_feat','base_prob','raw_prob','phase_feat','phase_prob','energy')}
  for start in range(0,len(fixed),32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);bf=teacher.features(a);rf=teacher.features(x)
   bp=F.softmax(teacher.classify_features(bf),1);rp=F.softmax(teacher.classify_features(rf),1);z=torch.complex(a[:,0,:],a[:,1,:]);ef=[];ep=[]
   for phase in phases:
    y=z*complex(np.cos(phase),np.sin(phase));wave=torch.stack((y.real,y.imag),1);f=teacher.features(wave)
    ef.append(F.normalize(f,dim=1).cpu());ep.append(F.softmax(teacher.classify_features(f),1).cpu())
   buckets['base_feat'].append(F.normalize(bf,dim=1).cpu());buckets['base_prob'].append(bp.cpu());buckets['raw_prob'].append(rp.cpu())
   buckets['phase_feat'].append(torch.stack(ef,1).cpu());buckets['phase_prob'].append(torch.stack(ep,1).cpu())
   buckets['energy'].append(torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),a.square().mean((1,2)).clamp_min(1e-8).log()),1).cpu())
  data[view]={k:torch.cat(v).contiguous() for k,v in buckets.items()}
  assert torch.equal(data[view]['base_prob'].argmax(1),data[view]['phase_prob'][:,0,:].argmax(1))
  print('PHASE_UTILITY_SOURCE_CANDIDATES_FIXED',view,flush=True)
print('ALL_PHASE_UTILITY_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1)
ref=np.from_dlpack(data['clean']['raw_prob']);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);n=len(fixed)
features=[];baselines=[];candidate_predictions=[]
angles=core.tensor(np.stack((np.cos(phases),np.sin(phases)),1).astype(np.float32),'cpu')
for view in views:
 v=data[view];bf=v['base_feat'][:,None,:].expand(-1,12,-1);bp=v['base_prob'][:,None,:].expand(-1,12,-1);rp=v['raw_prob'][:,None,:].expand(-1,12,-1);energy=v['energy'][:,None,:].expand(-1,12,-1);ang=angles[None,:,:].expand(n,-1,-1)
 feat=torch.cat((bf,v['phase_feat'],bp,rp,v['phase_prob'],energy,ang),2);assert feat.shape==(n,12,342)
 features.append(feat);baselines.append(v['base_prob'].argmax(1).cpu());candidate_predictions.append(v['phase_prob'].argmax(2).cpu())
features=torch.cat(features);base=np.from_dlpack(torch.cat(baselines).contiguous());candidate=np.from_dlpack(torch.cat(candidate_predictions).contiguous());all_target=np.tile(target,len(views));all_split=np.tile(split,len(views));all_valid=np.tile(valid,len(views))
base_right=base==all_target;right=candidate==all_target[:,None];disagree=candidate!=base[:,None]
outcome=np.where(~base_right[:,None]&right,1,np.where(base_right[:,None]&~right,2,0));train=(all_split[:,None]==0)&all_valid[:,None]&disagree
train_flat=np.flatnonzero(train.reshape(-1));counts=np.bincount(outcome.reshape(-1)[train_flat],minlength=3);assert np.all(counts>0)
weights=np.sqrt(counts.sum()/counts.astype(np.float64));weights/=weights.mean();class_weight=torch.tensor(weights.tolist(),dtype=torch.float32)
flat=features.reshape(-1,342);training=flat[torch.tensor(train_flat.tolist(),dtype=torch.long)];mean=training.mean(0);std=training.std(0).clamp_min(1e-4)
scaled=((flat-mean)/std).clamp(-10,10);target_t=torch.tensor(outcome.reshape(-1).tolist(),dtype=torch.long)
torch.manual_seed(2026110374);model=nn.Sequential(nn.Linear(342,128),nn.GELU(),nn.Linear(128,3));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);rng=np.random.default_rng(2026110375)
thresholds=(.0,.01,.03,.05,.1,.2);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def score_all():
 scores=[]
 with torch.no_grad():
  for start in range(0,len(scaled),4096):
   logits=model(scaled[start:start+4096])-class_weight.log();p=F.softmax(logits,1);scores.append((p[:,1]-p[:,2]).cpu())
 return np.from_dlpack(torch.cat(scores).contiguous()).reshape(len(views)*n,12)
def evaluate(scores,threshold,part,group=None):
 masked=scores.copy();masked[~disagree]=-np.inf;choice=masked.argmax(1);bestscore=masked[np.arange(len(masked)),choice];out=base.copy();use=np.isfinite(bestscore)&(bestscore>threshold);out[use]=candidate[np.arange(len(out))[use],choice[use]]
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for i,view in enumerate(views):
  sl=slice(i*n,(i+1)*n);old=base[sl]==target;new=out[sl]==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),switches=int((mask&use[sl]).sum()))
  if view=='clean':clean_net=int((cm&(out[sl]==teacher_target)).sum()-(cm&(base[sl]==teacher_target)).sum())
 return per,clean_net
for epoch in range(1,31):
 order=rng.permutation(train_flat)
 for start in range(0,len(order),512):
  ix=torch.tensor(order[start:start+512].tolist(),dtype=torch.long);logits=model(scaled[ix]);loss=F.cross_entropy(logits,target_t[ix],weight=class_weight)
  if not torch.isfinite(loss):raise FloatingPointError('Finite phase utility loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 if epoch in (2,5,10,20,30):
  scores=score_all()
  for threshold in thresholds:
   per,clean=evaluate(scores,threshold,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
   for group in (4,6):
    gp,gclean=evaluate(scores,threshold,1,group);count=int(((split==1)&(source_rx==group)).sum());gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=count,net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
   validation.append(dict(epoch=epoch,threshold=threshold,net=net,per_view=changes,clean_teacher_net=clean,groups=groups_v))
   guards=all(z['net']>=0 and z['clean_teacher_net']>=-.01*z['n'] and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
   if net>best['validation_net'] and guards:
    best=dict(kind='phase_utility',epoch=epoch,threshold=threshold,validation_net=net);best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  print('PHASE_UTILITY_SOURCE_VALIDATION_COMPLETE',epoch,flush=True)
if best_state is None:heldout=None;clean_net=None;state=None
else:
 model.load_state_dict(best_state);heldout,clean_net=evaluate(score_all(),best['threshold'],2);state={k:v.detach().tolist() for k,v in model.state_dict().items()}
print('SOURCE_PHASE_UTILITY_PROBE '+json.dumps(dict(status='SOURCE_ONLY_PHASE_UTILITY_PROBE',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),excluded_ids=int((split<0).sum()),train_receivers=[1,3,4],validation_receivers=[4,6],heldout_receivers=[8],train_days=[1,2],validation_days=[3],heldout_days=[3],phase_count=12,feature_dim=342,training_disagreements=int(len(train_flat)),training_outcome_counts=counts.tolist(),class_weights=weights.tolist(),validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=clean_net,model_state=state,feature_mean=mean.tolist(),feature_std=std.tolist(),U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_labels=True,
 caveat='Phase candidates are global carrier rotations selected using only received IQ and frozen models at inference. The selector holdout receiver/day is disjoint only from selector training; fixed experts saw source records. IDs exclude three earlier samples but may overlap other source probes. Source simulator is not validated real satellite data; earlier Phase1 test observations motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER',repr(inner),1)
proc=subprocess.Popen(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_PHASE_UTILITY_PROBE ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('Source result marker missing')
 dest=Path(__file__).parent/'phase1_source_phase_utility_probe.json'
 with dest.open('x',encoding='utf8') as f:f.write(payload)
 print('SAVED_SOURCE_PHASE_UTILITY_RESULT',dest,flush=True)
raise SystemExit(rc)

import pathlib,sys,torch,numpy as np,json,time
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
class ExactGELU(torch.nn.Module):
 def forward(self,x):return .5*x*(1.+torch.erf(x*0.7071067811865476))
torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;cache,clean,rx,train,_,_=s.source_contract(root);r1,sc1=s.epoch_data(cache,11);r2,sc2=s.epoch_data(cache,12);eligible=train[(sc1[train]!='clean')&(sc2[train]!='clean')];ids=np.random.default_rng(2026112155).choice(eligible,size=2304,replace=False);training=ids[:2048];held=ids[2048:];f,_=core.teacher(root,'cpu');assert all(not p.requires_grad for p in f.parameters());models=[];hist=[]
for mode in ('single','paired'):
 torch.manual_seed(2026112154);m,gate=s.initial_models(root,'cpu');m.body.local[2]=ExactGELU();m.train();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.0001,weight_decay=.001);losses=[];active_total=0
 for start in range(0,len(training),32):
  i=training[start:start+32];x1=core.tensor(r1[i],'cpu');x2=core.tensor(r2[i],'cpu');y=core.tensor(clean[i],'cpu')
  with torch.no_grad():active=(gate(x1).sigmoid()>.8)&(gate(x2).sigmoid()>.8)
  if not active.any():continue
  x1=x1[active];x2=x2[active];y=y[active];active_total+=len(y)
  with torch.no_grad():cf=f.features(y);cl=f.classify_features(cf);soft=F.softmax(cl/2,1)
  def one(x):
   with torch.no_grad():aligned,_=s.align(y,m.anchor(x))
   out=m(x);feat=f.features(out);logit=f.classify_features(feat);kl=F.kl_div(F.log_softmax(logit/2,1),soft,reduction='none').sum(1).mean()*4;finger=(1.-F.cosine_similarity(feat,cf,dim=1,eps=1e-8)).mean();den=aligned.square().mean((1,2)).clamp_min(1e-8);iq=((out-aligned).square().mean((1,2))/den).mean();inv=s.complex_gain_invariant_error(out,aligned).mean();return kl+.2*finger+.2*iq+.4*inv,feat
  l1,z1=one(x1)
  if mode=='paired':
   l2,z2=one(x2);consistency=(1.-F.cosine_similarity(z1,z2,dim=1,eps=1e-8)).mean();loss=.5*(l1+l2)+.3*consistency
  else:loss=l1
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite multi-view source loss')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
 models.append(m.eval().requires_grad_(False));hist.append({'mode':mode,'updates':len(losses),'active_pairs':active_total,'loss_mean':float(np.mean(losses)),'last16_loss_mean':float(np.mean(losses[-16:]))})
 print('SOURCE_PILOT_TRAINED',mode,'updates',len(losses),'active',active_total,flush=True)
# Comparator is loaded only after both models are fixed; source holdout is excluded from updates.
d=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for start in range(0,len(held),32):
  idx=held[start:start+32];x=core.tensor(r1[idx],'cpu');y=core.tensor(clean[idx],'cpu');p=f.classify_features(f.features(y)).softmax(1);valid=p.max(1).values>=.95;target=p.argmax(1);pred=[]
  for w in (x,d(x),models[0](x),models[1](x)):
   pred.append((f.classify_features(f.features(w)).argmax(1)==target).long())
  rows.append(torch.stack((valid.long(),*pred),1))
v=np.from_dlpack(torch.cat(rows).contiguous());result={}
for view in s.VIEWS[1:]:
 a=v[sc1[held]==view];a=a[a[:,0]>0];result[view]={'valid':len(a),'raw':int(a[:,1].sum()),'DURA':int(a[:,2].sum()),'single_view':int(a[:,3].sum()),'paired_consistency':int(a[:,4].sum())}
z=v[v[:,0]>0];result['all']={'valid':len(z),'raw':int(z[:,1].sum()),'DURA':int(z[:,2].sum()),'single_view':int(z[:,3].sum()),'paired_consistency':int(z[:,4].sum())}
print('SOURCE_PILOT_RESULT',json.dumps({'source_train_only':True,'train_ids':len(training),'heldout_from_train':len(held),'two_view_epochs':[11,12],'same_clean_physical_id':True,'F_frozen':True,'phase_factor_prior':'source-trained previously with frozen DURA, not used at inference','DURA_loaded_only_after_training':True,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'objective':'paired clean F feature + output IQ losses, with cross-view feature consistency for paired model','train':hist,'results':result,'limitations':'small exploratory source pilot, reused source identities, pseudo targets not true labels; paired model processes twice as many training views; not Phase1 test'},allow_nan=False),flush=True)

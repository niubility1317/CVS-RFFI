"""Train a Phase1 IQ prior against source blind-posterior residuals."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from signal_prior import SignalPrior
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026125054)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior,assume_unique=True);fit=np.sort(np.random.default_rng(2026125055).choice(pool,size=8192,replace=False))
 if np.intersect1d(fit,np.r_[prior,validation,heldout]).size:raise ValueError('Independent source fit and heldout IDs required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);model=SignalPrior().to(a.device)
 blindstate=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p0=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if blindstate.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or blindstate.get('epoch')!=10 or p0.get('source_ids')!=8192:raise ValueError('Exact independent source models required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);stats=(p0['means'].to(a.device),p0['covariances'].to(a.device),p0['priors'].to(a.device));eye=torch.eye(512,device=a.device)
 selected=fit if not a.smoke else np.sort(np.r_[fit[roles[fit]=='L_s'][:32],fit[roles[fit]=='U_s'][:32]])
 if a.smoke and len(selected)!=64:raise ValueError('Mixed source roles required')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Matching source IDs required')
 residual=np.empty((len(fit),2,256),dtype=np.float32);assignment=np.full(len(fit),-1,dtype=np.int16);wanted=np.isin(fit,selected);tick=time.monotonic();scene_counts={}
 for epoch in (21,22,23,11,12,13,14,15,16,17,18,19,20):
  source=fresh if epoch>=21 else cache;info=core.read(source/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
  received,scene=s.epoch_data(source,epoch);positions=np.flatnonzero(wanted&(assignment<0)&(scene[fit]!='clean'))
  with torch.no_grad():
   for start in range(0,len(positions),8):
    loc=positions[start:start+8];ix=fit[loc];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*stats);parent=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);e=parent-y;rms=e.square().mean((1,2),keepdim=True).sqrt()
    if not torch.isfinite(e).all() or not torch.isfinite(rms).all() or (rms<=1e-8).any():raise ValueError('Nonzero finite source residual required')
    residual[loc]=np.from_dlpack((e/rms).cpu().contiguous());assignment[loc]=epoch
  for value in np.unique(scene[fit[positions]]):scene_counts[str(value)]=scene_counts.get(str(value),0)+int(np.count_nonzero(scene[fit[positions]]==value))
  print('STRUCTURED_RESIDUAL_EPOCH',epoch,'count',len(positions),'remaining',int(np.count_nonzero(wanted&(assignment<0))),'seconds',round(time.monotonic()-tick,1),flush=True)
  if np.all(assignment[wanted]>0):break
 if np.any(assignment[wanted]<0):raise ValueError('Each fit record requires an impaired source view')
 lookup=np.full(len(rx),-1,dtype=np.int64);lookup[fit]=np.arange(len(fit));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);order_rng=np.random.default_rng(2026125056);levels=torch.tensor([.03,.08,.16,.32],device=a.device);counts={'exposures':0,'L_s':0,'U_s':0};losses=[]
 for epoch in range(1,(2 if a.smoke else 9)):
  if a.smoke:
   known=fit[roles[fit]=='L_s'][:32];unknown=fit[roles[fit]=='U_s'][:32];order=np.r_[known,unknown]
   if len(known)!=32 or len(unknown)!=32:raise ValueError('Mixed source roles required')
  else:order=order_rng.permutation(fit)
  total=0.
  for start in range(0,len(order),32):
   ix=order[start:start+32];y=core.tensor(clean[ix],a.device);scale=y.square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt();sigma=levels[torch.randint(len(levels),(len(ix),),device=a.device)];direction=core.tensor(residual[lookup[ix]],a.device);noisy=y+scale*sigma[:,None,None]*direction
   with torch.no_grad():yf=teacher.features(y);soft=F.softmax(teacher.classify_features(yf)/2,1)
   out=model(noisy,sigma);feature=teacher.features(out);logits=teacher.classify_features(feature);den=y.flatten(1).square().sum(1).clamp_min(1e-8);wave=(out-y).flatten(1).square().sum(1)/den;semantic=1-F.cosine_similarity(feature,yf,dim=1,eps=1e-8);kl=F.kl_div(F.log_softmax(logits/2,1),soft,reduction='none').sum(1)*4;per=wave+.2*semantic+.1*kl;groups=rx[ix];loss=torch.stack([per[torch.tensor((groups==g).tolist(),device=a.device)].mean() for g in np.unique(groups)]).mean()
   opt.zero_grad(set_to_none=True);loss.backward()
   if model.output.weight.grad is None or not torch.isfinite(model.output.weight.grad).all() or (epoch==1 and start==0 and model.output.weight.grad.abs().sum()==0):raise ValueError('Finite nonzero prior gradient required')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   if not torch.isfinite(loss) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):raise ValueError('Finite loss and frozen F required')
   opt.step();total+=float(loss.detach())*len(ix)
   if epoch==1:counts['exposures']+=len(ix);counts['L_s']+=int(np.count_nonzero(roles[ix]=='L_s'));counts['U_s']+=int(np.count_nonzero(roles[ix]=='U_s'))
  losses.append(total/len(order));print('STRUCTURED_PRIOR_EPOCH',epoch,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',counts=counts,scene_counts=scene_counts,residual_finite=True,F_frozen=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('SIGNAL_PRIOR_SMOKE PASS',flush=True);return
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=model.state_dict(),counts=counts,losses=losses,fit_ids=fit,source_view_assignment=assignment),f)
 core.write(a.out/'result.json',dict(status='SOURCE_TRAINING_COMPLETE',commit=a.commit,counts=counts,losses=losses,scene_counts=scene_counts,assignment_counts={str(int(i)):int(np.count_nonzero(assignment==i)) for i in np.unique(assignment)},F_frozen=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,fit_ids=len(fit)))
 print('SIGNAL_PRIOR_TRAINING_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()

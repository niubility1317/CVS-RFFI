"""Source-only task-guided differentiable blind Bayesian restoration pilot."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from differentiable_posterior import posterior

VAL=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-blind-bayes-source-validation-s2026092701-r01-release')
if (VAL/'release_commit.txt').read_text().strip()!='bcc0c56686ad57d0c101d7b0a7fc7822dd6e5ccb':raise ValueError('Exact source validation dependency required')
sys.path.insert(0,str(VAL/'experiments/phase1_blind_bayes_validation'))
from blind_bayes_source_validation import BlindActionNet,action_matrix,s,core,source_prior,SOURCE,SHA
from blind_action_model import loss as action_loss
from conditional_bayes_oracle import SCENES

PRIOR_RUN='20261102-phase1-blind-bayes-source-validation-s2026092701-r01'
PRIOR_SHA='5e35516d54d3e3d1d1749933861f290c1f87c8ffdb4b863d2f9819914135ceb9'

def apply(model,x,means,covs,priors,eye):
 h,logvar=model(x);A=action_matrix(h);N=eye[None,:,:]*logvar.exp()[:,None,None];states,lp=posterior(x.reshape(len(x),512),A,N,means,covs,priors);return (states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x)

def objective(model,teacher,x,y,k,c,v,means,covs,priors,eye):
 with torch.no_grad():
  yf=teacher.features(y);yp=F.softmax(teacher.classify_features(yf)/2,1)
 z=apply(model,x,means,covs,priors,eye);zf=teacher.features(z);logits=teacher.classify_features(zf);den=y.square().sum((1,2)).clamp_min(1e-8);iq=((z-y).square().sum((1,2))/den).mean();finger=(1-F.cosine_similarity(zf,yf,dim=1,eps=1e-8)).mean();kl=F.kl_div(F.log_softmax(logits/2,dim=1),yp,reduction='batchmean')*4;physics=action_loss(model,x,k,c,v);return .8*iq+.4*finger+.2*kl+.1*physics

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026120554)
 checkpoint=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_path.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed blind model and prior required')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False)
 if saved['commit']!='94693a342bb2daa684c0f21415d7b949f9fb2107' or saved['epoch']!=10 or prior['source_ids']!=8192:raise ValueError('Exact source provenance required')
 means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);r_prior=np.random.default_rng(2026120355);prior_ids=np.sort(r_prior.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);rng=np.random.default_rng(2026120555);fit=np.sort(rng.choice(pool,size=4096,replace=False));received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(r_prior.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(np.r_[fit,prior_ids],np.r_[validation,heldout]).size or np.intersect1d(fit,prior_ids).size:raise ValueError('Disjoint prior, fit and source holdout IDs required')
 model=BlindActionNet().to(a.device);model.load_state_dict(saved['model'],strict=True);baseline=BlindActionNet().to(a.device);baseline.load_state_dict(saved['model'],strict=True);baseline.eval().requires_grad_(False);teacher,_=core.teacher(a.root,a.device);folder=cache/'epochs/20';rec,scene20=s.epoch_data(cache,20);kernel=np.load(folder/'kernel.npy',mmap_mode='r',allow_pickle=False);coeff=np.load(folder/'coefficients.npy',mmap_mode='r',allow_pickle=False);var=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
 index=fit[scene20[fit]!='clean'][:8]
 if len(index)!=8:raise ValueError('Eight impaired source records required')
 x=core.tensor(rec[index],a.device);y=core.tensor(clean[index],a.device);k=core.tensor(kernel[index],a.device);c=core.tensor(coeff[index],a.device);v=core.tensor(var[index],a.device);z=objective(model,teacher,x,y,k,c,v,means,covs,priors,eye);params=list(model.parameters());zero=torch.autograd.grad(z*0,params,retain_graph=True,allow_unused=True)
 if not torch.isfinite(z) or any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise ValueError('Zero-gradient source smoke failed')
 z.backward()
 if model.head[-1].weight.grad is None or not torch.isfinite(model.head[-1].weight.grad).all() or not model.head[-1].weight.grad.abs().sum()>0:raise ValueError('Finite nonzero joint gradient required')
 if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('F must remain frozen')
 model.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=len(index),zero_gradient=True,finite_nonzero_gradient=True,F_frozen=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0);print('JOINT_POSTERIOR_SMOKE',json.dumps(smoke),flush=True)
 if a.smoke:
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
  return
 opt=torch.optim.AdamW(model.parameters(),lr=.00002,weight_decay=.001);order=rng.permutation(fit);total=0.;used=0;updates=0;tick=time.monotonic();model.train()
 for start in range(0,len(order),8):
  idx=order[start:start+8];idx=idx[scene20[idx]!='clean']
  if not len(idx):continue
  x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);k=core.tensor(kernel[idx],a.device);c=core.tensor(coeff[idx],a.device);v=core.tensor(var[idx],a.device);z=objective(model,teacher,x,y,k,c,v,means,covs,priors,eye)
  if not torch.isfinite(z):raise FloatingPointError('Nonfinite joint source loss')
  opt.zero_grad(set_to_none=True);z.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();used+=len(idx);total+=float(z.detach())*len(idx);updates+=1
  if updates%64==0:print('JOINT_POSTERIOR_PROGRESS',updates,'active',used,'loss',total/used,'seconds',round(time.monotonic()-tick,1),flush=True)
 if updates<400 or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Source updates or F freeze violated')
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=model.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
 model.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);variants={'raw':x,'dura':dura(x),'parent':apply(baseline,x,means,covs,priors,eye),'joint':apply(model,x,means,covs,priors,eye)};den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
   for name,signal in variants.items():
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   out.append(torch.stack(cols,1).cpu())
   if (start+len(idx))%64==0:print('JOINT_POSTERIOR_EVALUATED',start+len(idx),flush=True)
 values=torch.cat(out);results={}
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()))
  for i,name in enumerate(('raw','dura','parent','joint')):cell[name]=dict(correct=int((mask&(rows[:,2+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,3+2*i].mean().item()))
  results[view]=cell
 result=dict(status='COMPLETE_SOURCE_ONLY_JOINT_PILOT',commit=a.commit,scenes=results,fit_ids=len(fit),used=used,updates=updates,mean_loss=total/used,probe_ids=len(ids),probe_receivers=[8],parent_sha256=SHA,prior_sha256=PRIOR_SHA,source_only=True,F_frozen=True,DURA_loaded_after_training=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source physical IDs previously seen in other views and pilot; target is high-confidence clean-F pseudo label, not true accuracy; no test IQ.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('JOINT_POSTERIOR_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Source-only learned blind action plus conditional Gaussian posterior pilot."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from blind_action_model import BlindActionNet,effective_kernel,loss

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-conditional-bayes-oracle-s2026092701-r04-release')
if (DEP/'release_commit.txt').read_text().strip()!='dff683121e48ba8ab4ae7619ea895dad690bd77c':raise ValueError('Exact source prior dependency required')
sys.path.insert(0,str(DEP/'experiments/phase1_conditional_bayes'))
from conditional_bayes_oracle import s,core,source_prior,fit_prior,real_operator,noise_covariance,SCENES
from gaussian_posterior import posterior

def data(cache,epoch):
 f=cache/'epochs'/('%02d'%epoch);received,scene=s.epoch_data(cache,epoch)
 return received,scene,*[np.load(f/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('kernel','coefficients','variance')]

def run_batch(net,cache,idx,epoch,device):
 received,scene,kernel,coefficients,variance=data(cache,epoch);idx=idx[scene[idx]!='clean']
 if not len(idx):return None
 x=core.tensor(received[idx],device);k=core.tensor(kernel[idx],device);c=core.tensor(coefficients[idx],device);v=core.tensor(variance[idx],device)
 return x,k,c,v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source-only output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026120454)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);r_prior=np.random.default_rng(2026120355);prior=np.sort(r_prior.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior,assume_unique=True);rng=np.random.default_rng(2026120455);fit=np.sort(rng.choice(pool,size=8192,replace=False));received,scene,*_=data(cache,19);ids=np.concatenate([np.sort(r_prior.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(fit,prior).size or np.intersect1d(np.r_[fit,prior],np.r_[validation,heldout]).size:raise ValueError('Disjoint source physical IDs required')
 model=BlindActionNet().to(a.device);smoke_batch=run_batch(model,cache,fit[:48],11,a.device)
 if smoke_batch is None:raise ValueError('Impaired source smoke required')
 x,k,c,v=smoke_batch;z=loss(model,x,k,c,v);zero=torch.autograd.grad(z*0,list(model.parameters()),retain_graph=True,allow_unused=True)
 if not torch.isfinite(z) or any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise ValueError('Zero gradient source smoke failed')
 z.backward()
 if model.head[-1].weight.grad is None or not torch.isfinite(model.head[-1].weight.grad).all() or not model.head[-1].weight.grad.abs().sum()>0:raise ValueError('Finite nonzero source gradient required')
 model.zero_grad(set_to_none=True);print('BLIND_ACTION_SMOKE',json.dumps(dict(status='PASS',impaired=len(x),zero_gradient=True,finite_nonzero_gradient=True,F_loaded=False,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0)),flush=True)
 if a.smoke:
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',impaired=len(x),source_only=True,zero_gradient=True,finite_nonzero_gradient=True,F_loaded=False,DURA_loaded=False,TX_labels_read=0,V_reads=0,test_reads=0),f,indent=2)
  return
 opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);losses=[];active_counts=[]
 for epoch in (1,2,3):
  rec,scene,k,c,v=data(cache,epoch+10);order=rng.permutation(fit);model.train();total=0.;count=0;tick=time.monotonic()
  for start in range(0,len(order),64):
   idx=order[start:start+64];idx=idx[scene[idx]!='clean']
   if not len(idx):continue
   x=core.tensor(rec[idx],a.device);kt=core.tensor(k[idx],a.device);ct=core.tensor(c[idx],a.device);vt=core.tensor(v[idx],a.device);z=loss(model,x,kt,ct,vt)
   if not torch.isfinite(z):raise FloatingPointError('Nonfinite blind action loss')
   opt.zero_grad(set_to_none=True);z.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(z.detach())*len(idx);count+=len(idx)
  if count<5000:raise ValueError('Insufficient impaired source training records')
  losses.append(total/count);active_counts.append(count);print('BLIND_ACTION_EPOCH',epoch,'count',count,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,epoch=3,model=model.state_dict()),f)
 model.eval().requires_grad_(False);teacher,_=core.teacher(a.root,a.device);gm,gc,cm,cc,prior_probs,counts,ridge,nprior=fit_prior(clean,prior,teacher,a.device);dura=source_prior.frozen_dura(a.root,a.device);received,scene,kernel,coefficients,variance=data(cache,19);out=[];eye=torch.eye(512,device=a.device);
 with torch.no_grad():
  for start in range(0,len(ids),2):
   idx=ids[start:start+2];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);k=core.tensor(kernel[idx],a.device);c=core.tensor(coefficients[idx],a.device);v=core.tensor(variance[idx],a.device);estimate,logvar=model(x);rms=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8);target_h=effective_kernel(k,c,rms)*rms[:,None,None];herr=(estimate-target_h).square().sum((1,2))/target_h.square().sum((1,2)).clamp_min(1e-12);verr=(logvar-v.clamp_min(1e-10).log()).abs()
   hs=np.from_dlpack(estimate.cpu().contiguous());ident=np.zeros((len(idx),4,256),dtype=np.float32);ident[:,0]=1;blind_A=core.tensor(np.stack([real_operator(h,c0) for h,c0 in zip(hs,ident)]),a.device);blind_N=eye[None,:,:]*logvar.exp()[:,None,None];flat=x.reshape(len(idx),512)
   bg,_=posterior(flat,blind_A,blind_N,gm,gc,torch.ones(1,device=a.device));bm,bp=posterior(flat,blind_A,blind_N,cm,cc,prior_probs);bm=(bm*bp.exp().unsqueeze(-1)).sum(1)
   true_c=coefficients[idx];true_A=core.tensor(np.stack([real_operator(h,c0) for h,c0 in zip(kernel[idx],true_c)]),a.device);true_N=core.tensor(np.stack([noise_covariance(c0) for c0 in true_c]),a.device);kg,_=posterior(flat,true_A,true_N,gm,gc,torch.ones(1,device=a.device));km,kp=posterior(flat,true_A,true_N,cm,cc,prior_probs);km=(km*kp.exp().unsqueeze(-1)).sum(1)
   prob=F.softmax(teacher.classify_features(teacher.features(y)),1);conf,target=prob.max(1);variants={'raw':x,'dura':dura(x),'blind_global':bg[:,0].reshape_as(x),'blind_mixture':bm.reshape_as(x),'known_global':kg[:,0].reshape_as(x),'known_mixture':km.reshape_as(x)};den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float(),herr,verr]
   for name,z in variants.items():
    pred=teacher.classify_features(teacher.features(z)).argmax(1);iq=(z-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   out.append(torch.stack(cols,1).cpu())
   if (start+len(idx))%64==0:print('BLIND_ACTION_EVALUATED',start+len(idx),flush=True)
 values=torch.cat(out);results={}
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()),mean_kernel_nmse=float(rows[:,2].mean().item()),mean_abs_log_noise_error=float(rows[:,3].mean().item()))
  for i,name in enumerate(('raw','dura','blind_global','blind_mixture','known_global','known_mixture')):cell[name]=dict(correct=int((mask&(rows[:,4+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,5+2*i].mean().item()))
  results[view]=cell
 result=dict(status='COMPLETE_SOURCE_ONLY_PILOT',commit=a.commit,scenes=results,train_ids=len(fit),prior_ids=len(prior),probe_ids=len(ids),probe_receivers=[8],losses=losses,active_counts=active_counts,prior_counts=counts,source_only=True,F_loaded_after_training=True,DURA_loaded_after_training=True,blind_inference_inputs='single received IQ and frozen source models only',known_variants_are_oracles=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,source_caveat='physical IDs may have appeared in earlier source views; pseudo targets are not true labels; simulator is not real satellite data')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('BLIND_ACTION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

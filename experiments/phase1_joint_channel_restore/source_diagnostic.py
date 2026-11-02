"""Source-only blind signal/channel alternating restoration diagnostic."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from channel_update import update_channel,write_fixed_predictions
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def signal_estimate(x,h,lv,means,covs,priors,eye):
 A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],means,covs,priors);return (states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x)

def predictions(x,blind,means,covs,priors,eye):
 h0,lv=blind(x);parent=signal_estimate(x,h0,lv,means,covs,priors,eye);signals=[parent];kernels=[h0]
 for alpha in (.1,1.,10.):
  h=h0;z=parent
  for _ in range(3):
   h=update_channel(z,x,h0,alpha);z=signal_estimate(x,h,lv,means,covs,priors,eye)
  signals.append(z);kernels.append(h)
 return signals,kernels

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_file.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact frozen source model and prior required')
 weights=torch.load(ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(weights['model'],strict=True);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(prior_ids,np.r_[validation,heldout]).size:raise ValueError('Source prior and source holdout must be disjoint')
 if a.smoke:
  x=core.tensor(received[ids[:8]],a.device)
  with torch.no_grad():signals,kernels=predictions(x,blind,means,covs,priors,eye)
  if not all(torch.isfinite(v).all() for v in signals+kernels):raise ValueError('Finite source channel and signal output required')
  smoke_values=torch.stack([q.mean((1,2)) for q in signals],1);smoke_kernels=torch.stack(kernels,1);saved_values,saved_kernels,digest=write_fixed_predictions(a.out/'source_smoke_predictions.npz',ids[:8],smoke_values,smoke_kernels)
  with np.load(a.out/'source_smoke_predictions.npz',allow_pickle=False) as f:
   if not np.array_equal(f['indices'],ids[:8]) or not np.array_equal(f['values'],saved_values) or not np.array_equal(f['kernels'],saved_kernels):raise ValueError('Source prediction serialization roundtrip required')
  result=dict(status='PASS',records=len(x),candidate_count=len(signals),prediction_roundtrip=True,prediction_sha256=digest,source_only=True,F_loaded=False,DURA_loaded=False,source_channel_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
  print('JOINT_CHANNEL_SMOKE',json.dumps(result),flush=True);return
 teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);values=[];kernel_values=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);signals,kernels=predictions(x,blind,means,covs,priors,eye)
   if not all(torch.isfinite(v).all() for v in signals+kernels):raise ValueError('Finite source candidate required')
   y=core.tensor(clean[idx],a.device);confidence,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);d=dura(x);den=y.square().sum((1,2)).clamp_min(1e-12);xden=x.square().sum((1,2)).clamp_min(1e-12);cols=[confidence,target.float()]
   for signal in [d]+signals:
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   for h,z in zip(kernels,signals):
    A=action_matrix(h);forward=torch.bmm(A,z.reshape(len(z),512,1)).reshape_as(x);cols.append((forward-x).square().sum((1,2))/xden)
   values.append(torch.stack(cols,1).cpu());kernel_values.append(torch.stack(kernels,1).cpu())
   if (start+len(idx))%64==0:print('JOINT_CHANNEL_PREDICTED',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 fixed=a.out/'source_predictions.npz';values,kernels,digest=write_fixed_predictions(fixed,ids,torch.cat(values),torch.cat(kernel_values));print('JOINT_CHANNEL_PREDICTIONS_FIXED',digest,flush=True)
 epochdir=cache/'epochs'/'19';true_kernel=np.load(epochdir/'kernel.npy',mmap_mode='r',allow_pickle=False);coeff=np.load(epochdir/'coefficients.npy',mmap_mode='r',allow_pickle=False);raw=true_kernel[ids];c=coeff[ids];hr=raw[:,0];hi=raw[:,1];ar=c[:,0,128,None];ai=c[:,1,128,None];truth=np.stack((ar*hr-ai*hi,ar*hi+ai*hr),1);den=np.maximum(np.square(truth).sum((1,2)),1e-12);herr=np.square(kernels-truth[:,None]).sum((2,3))/den[:,None]
 result={};names=('dura','parent','ridge_0.1','ridge_1','ridge_10')
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];kh=herr[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum()))
  for i,name in enumerate(names):cell[name]=dict(correct=int(np.count_nonzero(valid&(rows[:,2+2*i]==rows[:,1]))),mean_iq_nmse=float(rows[:,3+2*i].mean()))
  for i,name in enumerate(names[1:]):cell[name].update(mean_channel_nmse=float(kh[:,i].mean()),mean_forward_residual=float(rows[:,12+i].mean()))
  result[view]=cell
 out=dict(status='COMPLETE_SOURCE_ONLY_JOINT_CHANNEL',commit=a.commit,prediction_sha256=digest,source_only=True,probe_ids=len(ids),scene=result,regularization=[.1,1.,10.],iterations=3,source_channel_truth_after_predictions=True,source_channel_truth_in_inference=False,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Fixed source-only predictions; source channel metadata is accessed only afterward. Clean F pseudo labels and previously explored source IDs are not independent true-label test accuracy. Midpoint channel truth is an approximate effective operator.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print('JOINT_CHANNEL_COMPLETE',json.dumps(out),flush=True)
if __name__=='__main__':main()

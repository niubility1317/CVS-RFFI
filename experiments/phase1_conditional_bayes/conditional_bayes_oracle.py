"""Source-only diagnostic: conditional Gaussian IQ prior with known simulated action."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from gaussian_posterior import posterior

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-phase-shift-student-s2026092701-r02-release')
if (DEP/'release_commit.txt').read_text().strip()!='284317bd819df974395d3388b1920a41300937cb':raise ValueError('Exact Phase1 source dependency required')
sys.path.insert(0,str(DEP/'experiments/phase1_phase_shift_restorer'))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as source_prior
REFERENCE=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-DURA-known-action-Gaussian-r01')
reference_file=REFERENCE/'experiments/pamf_dura_efficiency/known_action_reference.py'
if (REFERENCE/'release_commit.txt').read_text().strip()!='f443942e33318b9f72733908dc2735203addaa96' or hashlib.sha256(reference_file.read_bytes()).hexdigest()!='e01efc635b4bbdc17d41014d72c9e6bbe3ad631553da9c3cdb6489a3058f1427':raise ValueError('Exact known-action reference required')
spec=importlib.util.spec_from_file_location('known_action_reference',reference_file);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
real_operator,noise_covariance=reference.real_operator,reference.noise_covariance

SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def fit_prior(clean,fit,teacher,device):
 logits=[]
 with torch.no_grad():
  for start in range(0,len(fit),128):
   x=core.tensor(clean[fit[start:start+128]],device);logits.append(teacher.classify_features(teacher.features(x)).cpu())
 p=torch.softmax(torch.cat(logits),1);confidence,labels=p.max(1);use=np.from_dlpack(confidence.contiguous())>=.95;labels=np.from_dlpack(labels.contiguous())[use]
 values=np.asarray(clean[fit[use]],dtype=np.float64).reshape(-1,512);mean=values.mean(0);d=values-mean;global_cov=d.T@d/len(values);global_cov=(global_cov+global_cov.T)/2;ridge=1e-4*np.trace(global_cov)/512;global_cov+=ridge*np.eye(512)
 means=[];covs=[];counts=[]
 for c in range(6):
  v=values[labels==c];counts.append(len(v))
  if len(v)<128:raise ValueError('At least 128 high-confidence source records per class required')
  m=v.mean(0);d=v-m;cov=d.T@d/len(v);cov=.8*((cov+cov.T)/2)+.2*global_cov+ridge*np.eye(512);means.append(m);covs.append(cov)
 to=lambda z:core.tensor(np.asarray(z,dtype=np.float32),device)
 return to(mean[None,:]),to(global_cov[None,:,:]),to(np.stack(means)),to(np.stack(covs)),to(np.array(counts)/sum(counts)),counts,float(ridge),int(np.count_nonzero(use))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source-only output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026120354);rng=np.random.default_rng(2026120355)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);fit=np.sort(rng.choice(training,size=8192,replace=False));received,scene=s.epoch_data(cache,19)
 if np.intersect1d(fit,np.r_[validation,heldout]).size:raise ValueError('Disjoint physical IDs required')
 per_scene=2 if a.smoke else 64;ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=per_scene,replace=False)) for v in SCENES]);teacher,_=core.teacher(a.root,a.device);gm,gc,cm,cc,prior,counts,ridge,fit_valid=fit_prior(clean,fit,teacher,a.device);print('PRIOR_FIT',json.dumps(dict(valid=fit_valid,counts=counts,ridge=ridge)),flush=True)
 dura=source_prior.frozen_dura(a.root,a.device);folder=cache/'epochs/19';kernel=np.load(folder/'kernel.npy',mmap_mode='r',allow_pickle=False);coefs=np.load(folder/'coefficients.npy',mmap_mode='r',allow_pickle=False);out=[];start_time=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),2):
   idx=ids[start:start+2];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);v=coefs[idx];acts=np.stack([real_operator(k,c) for k,c in zip(kernel[idx],v)]);noise=np.stack([noise_covariance(c) for c in v]);A=core.tensor(acts,a.device);N=core.tensor(noise,a.device);flat=x.reshape(len(idx),512)
   gaussian,_=posterior(flat,A,N,gm,gc,torch.ones(1,device=a.device));states,lp=posterior(flat,A,N,cm,cc,prior);mix=(states*lp.exp().unsqueeze(-1)).sum(1);choice=lp.argmax(1);mapout=states[torch.arange(len(idx),device=a.device),choice];cleanlog=teacher.classify_features(teacher.features(y));cleanprob=F.softmax(cleanlog,1);conf,target=cleanprob.max(1);conditional_oracle=states[torch.arange(len(idx),device=a.device),target]
   variants={'raw':x,'dura':dura(x),'global':gaussian[:,0].reshape_as(x),'mixture':mix.reshape_as(x),'map':mapout.reshape_as(x),'oracle_class':conditional_oracle.reshape_as(x)};den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float(),choice.float()]
   for name,z in variants.items():
    pred=teacher.classify_features(teacher.features(z)).argmax(1);iq=(z-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   row=torch.stack(cols,1);out.append(row.cpu());
   if (start+len(idx))%64==0:print('CONDITIONAL_GAUSSIAN_PROGRESS',start+len(idx),'elapsed',round(time.monotonic()-start_time,1),flush=True)
 values=torch.cat(out);result={}
 for j,view in enumerate(SCENES):
  rows=values[j*per_scene:(j+1)*per_scene];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()),class_map_agreement=int((mask&(rows[:,2]==rows[:,1])).sum().item()))
  for i,name in enumerate(('raw','dura','global','mixture','map','oracle_class')):cell[name]=dict(correct=int((mask&(rows[:,3+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,4+2*i].mean().item()))
  result[view]=cell
 result=dict(status='COMPLETE_SOURCE_ONLY_ORACLE_SMOKE' if a.smoke else 'COMPLETE_SOURCE_ONLY_ORACLE',commit=a.commit,scenes=result,fit_ids=len(fit),fit_valid=fit_valid,class_counts=counts,shrink_to_global=.2,ridge=ridge,probe_ids=len(ids),probe_receivers=[8],cache_epoch=19,source_only=True,F_frozen=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,known_action_at_inference=True,known_action_is_simulator_only=True,oracle_class_uses_clean_F_target=True,source_caveat='physical IDs may have appeared in other source views; AGC-dependent noise breaks exact likelihood; estimated mixture is not a certified upper bound')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('CONDITIONAL_GAUSSIAN_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

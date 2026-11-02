"""Source-only hierarchical class/waveform Gaussian prior pilot."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from hierarchical_prior import fit_hierarchical
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def fit_source_prior(clean,fit,teacher,device):
 logits=[]
 with torch.no_grad():
  for start in range(0,len(fit),128):
   x=core.tensor(clean[fit[start:start+128]],device);logits.append(teacher.classify_features(teacher.features(x)).cpu())
 conf,labels=F.softmax(torch.cat(logits),1).max(1);use=np.from_dlpack(conf.contiguous())>=.95;lab=np.from_dlpack(labels.contiguous())[use];values=np.asarray(clean[fit[use]],dtype=np.float64).reshape(-1,512);means,covs,priors,counts,classes,ridge=fit_hierarchical(values,lab)
 return means,covs,priors,counts,classes,ridge,int(use.sum())

@torch.no_grad()
def pair(x,blind,old,sub,eye):
 h,lv=blind(x);A=action_matrix(h);N=eye[None]*lv.exp()[:,None,None];a,lp=posterior(x.reshape(len(x),512),A,N,old[0],old[1],old[2]);b,lq=posterior(x.reshape(len(x),512),A,N,sub[0],sub[1],sub[2]);return (a*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x),(b*lq.exp().unsqueeze(-1)).sum(1).reshape_as(x)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start_time=time.monotonic()
 ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_file.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed source checkpoint and prior required')
 state=torch.load(ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model'],strict=True);blind.eval().requires_grad_(False);old=(prior['means'].to(a.device),prior['covariances'].to(a.device),prior['priors'].to(a.device));eye=torch.eye(512,device=a.device)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);fit=np.sort(rng.choice(training,size=8192,replace=False));received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(fit,np.r_[validation,heldout]).size:raise ValueError('Prior fit and source holdout IDs must be disjoint')
 teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);means,covs,priors,counts,classes,ridge,nvalid=fit_source_prior(clean,fit,teacher,a.device);convert=lambda v:core.tensor(np.asarray(v,dtype=np.float32),a.device);sub=(convert(means),convert(covs),convert(priors))
 print('HIERARCHICAL_PRIOR_FIT',json.dumps(dict(fit_ids=len(fit),valid=nvalid,counts=counts.tolist(),ridge=ridge,seconds=round(time.monotonic()-start_time,1))),flush=True)
 if a.smoke:
  x=core.tensor(received[ids[:8]],a.device);parent,proposed=pair(x,blind,old,sub,eye)
  if not torch.isfinite(parent).all() or not torch.isfinite(proposed).all():raise ValueError('Finite source posterior required')
  out=dict(status='PASS',records=len(x),components=len(counts),fit_ids=len(fit),high_conf_fit=nvalid,subcluster_counts=counts.tolist(),source_only=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
  print('HIERARCHICAL_PRIOR_SMOKE',json.dumps(out),flush=True);return
 with (a.out/'source_hierarchical_prior.pt').open('xb') as f:torch.save(dict(commit=a.commit,means=sub[0].cpu(),covariances=sub[1].cpu(),priors=sub[2].cpu(),counts=counts.tolist(),classes=classes.tolist(),fit_ids=len(fit),high_conf_fit=nvalid,ridge=ridge),f)
 dura=source_prior.frozen_dura(a.root,a.device);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);parent,proposed=pair(x,blind,old,sub,eye);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);signals=[x,dura(x),parent,proposed];den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
   for signal in signals:
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   batch=torch.stack(cols,1).cpu()
   if not torch.isfinite(batch).all():raise ValueError('Finite source prediction required')
   out.append(batch)
 values=np.from_dlpack(torch.cat(out).contiguous());result={}
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum()))
  for i,name in enumerate(('raw','dura','parent','hierarchical')):cell[name]=dict(correct=int(np.count_nonzero(valid&(rows[:,2+2*i]==rows[:,1]))),mean_iq_nmse=float(rows[:,3+2*i].mean()))
  result[view]=cell
 out=dict(status='COMPLETE_SOURCE_ONLY_HIERARCHICAL_PRIOR',commit=a.commit,scenes=result,fit_ids=len(fit),high_conf_fit=nvalid,subcluster_counts=counts.tolist(),classes=classes.tolist(),ridge=ridge,components=len(counts),probe_ids=len(ids),source_only=True,F_frozen=True,DURA_loaded_after_fit=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Prior clusters use only frozen F high-confidence source predictions and clean source IQ; source IDs previously explored; pseudo targets are not true test accuracy. More components can hallucinate identity.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print('HIERARCHICAL_PRIOR_COMPLETE',json.dumps(out),flush=True)
if __name__=='__main__':main()

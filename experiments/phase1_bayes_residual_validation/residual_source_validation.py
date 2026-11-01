"""Source-only fixed multi-view residual model validation and gated selection."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from residual_selection import choose,VIEWS

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact multi-view residual source release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import ResidualCorrector,BlindActionNet,prior_output,s,core,source_prior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA
RESIDUAL_RUN='20261102-phase1-bayes-residual-full-s2026092701-r01'
THRESHOLDS=(.8,.99)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 parent=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=a.root/'runs'/PRIOR_RUN/'source_prior.pt';residual_path=a.root/'runs'/RESIDUAL_RUN/'checkpoint_epoch03.pt';record=core.read(a.root/'runs'/RESIDUAL_RUN/'result.json')
 if hashlib.sha256(parent.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_path.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact parent and source prior required')
 if record.get('status')!='COMPLETE_SOURCE_ONLY_RESIDUAL_FULL' or record.get('commit')!='848c4b4c47b054f43c8303638e5aa549154ebfd4' or record.get('fit_ids')!=38951 or not record.get('F_frozen') or not record.get('blind_parent_frozen') or any(record.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Completed fixed multi-view source training required')
 residual_hash=hashlib.sha256(residual_path.read_bytes()).hexdigest();saved=torch.load(residual_path,map_location='cpu',weights_only=False);parent_state=torch.load(parent,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False)
 if saved.get('commit')!=record['commit'] or saved.get('updates',0)<6000 or saved.get('parent_sha256')!=SHA or saved.get('prior_sha256')!=PRIOR_SHA:raise ValueError('Fixed final source checkpoint provenance required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);teacher,_=core.teacher(a.root,a.device);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False);model=BlindActionNet().to(a.device);model.load_state_dict(parent_state['model'],strict=True);model.eval().requires_grad_(False);net=ResidualCorrector().to(a.device);net.load_state_dict(saved['model'],strict=True);net.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
 if a.smoke:
  ids=training[:32];x=core.tensor(received[ids],a.device);z,b=prior_output(model,x,means,covs,priors,eye);out=net(z,b);prob=gate(x).sigmoid();logits=teacher.classify_features(teacher.features(out))
  if not all(torch.isfinite(v).all() for v in (z,b,out,prob,logits)):raise ValueError('Finite fixed source inference required')
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(ids),active=int((prob>.8).sum().item()),source_only=True,residual_sha256=residual_hash,test_reads=0),f,indent=2)
  print('RESIDUAL_VALIDATION_SMOKE PASS',len(ids),flush=True);return
 dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];out=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);base=teacher.classify_features(teacher.features(d)).argmax(1);prob=gate(x).sigmoid();enabled=prob>.8;z=x.clone();corrected=x.clone()
   if enabled.any():
    v=x[enabled];pz,b=prior_output(model,v,means,covs,priors,eye);z[enabled]=pz;corrected[enabled]=net(pz,b)
   parentpred=teacher.classify_features(teacher.features(z)).argmax(1);pred=teacher.classify_features(teacher.features(corrected)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);_,pe=s.align(z,y);_,ce=s.align(corrected,y);out.append(torch.stack((target.float(),conf,base.float(),prob,de/den,raw.float(),re/den,parentpred.float(),pe/den,pred.float(),ce/den),1).cpu())
   if (start+len(idx))%256==0:print('RESIDUAL_VALIDATION_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(out).contiguous());result={}
 for threshold in THRESHOLDS:
  cells={}
  for part,receivers in ((1,(4,6)),(2,(8,))):
   for receiver in receivers:
    for view in VIEWS:
     mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
     if not len(v):raise ValueError('All source receiver/scene cells required')
     trusted=v[:,1]>=.95;active=v[:,3]>threshold;pp=np.where(active,v[:,7],v[:,5]);pred=np.where(active,v[:,9],v[:,5]);piq=np.where(active,v[:,8],v[:,6]);iq=np.where(active,v[:,10],v[:,6])
     cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=int(np.count_nonzero(trusted&(v[:,2]==v[:,0]))),parent_correct=int(np.count_nonzero(trusted&(pp==v[:,0]))),candidate_correct=int(np.count_nonzero(trusted&(pred==v[:,0]))),baseline_iq=float(np.mean(v[:,4])),parent_iq=float(np.mean(piq)),candidate_iq=float(np.mean(iq)))
  result[str(threshold)]=cells
 selection={name:choose(result,require_iq,reference) for name,require_iq,reference in (('primary_dura',True,'baseline'),('accuracy_dura',False,'baseline'),('primary_parent',True,'parent'),('accuracy_parent',False,'parent'))}
 summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,residual_sha256=residual_hash,parent_sha256=SHA,prior_sha256=PRIOR_SHA,validation_ids=len(validation),heldout_ids=len(heldout),validation=result,selection=selection,source_only=True,F_frozen=True,DURA_inference=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source IDs previously used in other channel views, ground F pseudo targets are not truth; no target prediction.')
 with (a.out/'source_validation.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
 print('RESIDUAL_VALIDATION_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()

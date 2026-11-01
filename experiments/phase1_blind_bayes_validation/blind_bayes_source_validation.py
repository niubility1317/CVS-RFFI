"""Frozen blind model source-only per-receiver and scene validation."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from blind_action_matrix import action_matrix

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-blind-bayes-action-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='94693a342bb2daa684c0f21415d7b949f9fb2107':raise ValueError('Exact blind model source release required')
sys.path.insert(0,str(DEP/'experiments/phase1_blind_bayes_full'))
from blind_action_full import BlindActionNet,s,core,fit_prior,posterior,source_prior,real_operator
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
THRESHOLDS=(.8,.99)
SOURCE='20261102-phase1-blind-bayes-action-full-s2026092701-r01'
SHA='0ef60a205f785bda45f9936d519e925179dbbddeef2fd6cd285a83b02a284338'

def select(results,require_iq):
 best=dict(kind='dura',validation_net=0)
 required={f'{part}:{rx}:{view}' for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for view in VIEWS}
 for threshold,cells in results.items():
  if set(cells)!=required:raise ValueError('All receiver/scene cells required')
  val=[(k,v) for k,v in cells.items() if k.startswith('1:')]
  if any(v['count']<=0 or v['valid_count']<=0 for _,v in val):raise ValueError('Valid source cells required')
  net=sum(v['candidate_correct']-v['baseline_correct'] for k,v in val if not k.endswith(':clean'))
  cls=all(v['candidate_correct']>=v['baseline_correct']-max(1,math.ceil(.005*v['valid_count'])) for _,v in val)
  clean=all(v['candidate_iq']<=v['baseline_iq']+.001 for k,v in val if k.endswith(':clean'))
  impaired=all(v['candidate_iq']<=1.2*v['baseline_iq']+1e-6 for k,v in val if not k.endswith(':clean'))
  if cls and clean and (impaired or not require_iq) and net>=10 and net>best['validation_net']:best=dict(kind='blind_bayes',threshold=float(threshold),validation_net=net)
 return best

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';record=core.read(a.root/'runs'/SOURCE/'result.json')
 if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=SHA or record['status']!='COMPLETE_SOURCE_ONLY_FULL' or record['train_ids']!=38951:raise ValueError('Exact fixed source checkpoint required')
 state=torch.load(ckpt,map_location='cpu',weights_only=False)
 if state['commit']!='94693a342bb2daa684c0f21415d7b949f9fb2107' or state['epoch']!=10:raise ValueError('Fixed final source weights required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False))
 if np.intersect1d(prior_ids,np.r_[validation,heldout]).size:raise ValueError('Disjoint prior and source validation required')
 teacher,_=core.teacher(a.root,a.device);_,_,means,covs,priors,counts,ridge,nvalid=fit_prior(clean,prior_ids,teacher,a.device);phase,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False);model=BlindActionNet().to(a.device);model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False);received,scene=s.epoch_data(cache,19)
 if a.smoke:
  ids=training[:32];x=core.tensor(received[ids],a.device);h,v=model(x);matrix=action_matrix(h);hn=np.from_dlpack(h.cpu().contiguous());identity=np.zeros((2,4,256),dtype=np.float32);identity[:,0]=1;reference=core.tensor(np.stack([real_operator(k,c) for k,c in zip(hn[:2],identity)]),a.device);delta=(matrix[:2]-reference).abs().max().item();g=gate(x).sigmoid();active=g>.8
  if delta>1e-6 or not torch.isfinite(matrix).all() or not torch.isfinite(v).all() or not torch.isfinite(g).all():raise ValueError('Frozen source action or gate smoke failed')
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(ids),active=int(active.sum()),action_matrix_max_abs=delta,source_only=True,checkpoint_sha256=SHA,test_reads=0),f,indent=2)
  print('BLIND_VALIDATION_SMOKE',delta,'active',int(active.sum()),flush=True);return
 with (a.out/'source_prior.pt').open('xb') as f:torch.save(dict(means=means.cpu(),covariances=covs.cpu(),priors=priors.cpu(),source_ids=8192,fit_valid=nvalid,counts=counts,ridge=ridge),f)
 dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];out=[];eye=torch.eye(512,device=a.device);tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);base=teacher.classify_features(teacher.features(d)).argmax(1);prob=gate(x).sigmoid();enabled=prob>.8;z=x.clone()
   if enabled.any():
    xe=x[enabled];h,lv=model(xe);A=action_matrix(h);N=eye[None,:,:]*lv.exp()[:,None,None];states,weights=posterior(xe.reshape(len(xe),512),A,N,means,covs,priors);z[enabled]=(states*weights.exp().unsqueeze(-1)).sum(1).reshape_as(xe)
   pred=teacher.classify_features(teacher.features(z)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);_,ze=s.align(z,y);out.append(torch.stack((target.float(),conf,base.float(),prob,de/den,raw.float(),re/den,pred.float(),ze/den),1).cpu())
   if (start+len(idx))%256==0:print('BLIND_VALIDATION_PROGRESS',start+len(idx),'elapsed',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(out).contiguous());result={}
 for threshold in THRESHOLDS:
  cells={}
  for part,receivers in ((1,(4,6)),(2,(8,))):
   for receiver in receivers:
    for view in VIEWS:
     mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
     if not len(v):raise ValueError('All source receiver/scene cells required')
     trusted=v[:,1]>=.95;active=v[:,3]>threshold;prediction=np.where(active,v[:,7],v[:,5]);iq=np.where(active,v[:,8],v[:,6])
     cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=int(np.count_nonzero(trusted&(v[:,2]==v[:,0]))),candidate_correct=int(np.count_nonzero(trusted&(prediction==v[:,0]))),baseline_iq=float(np.mean(v[:,4])),candidate_iq=float(np.mean(iq)))
  result[str(threshold)]=cells
 primary=select(result,True);accuracy=select(result,False)
 summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,checkpoint_sha256=SHA,source_prior_counts=counts,source_prior_ids=len(prior_ids),validation_ids=len(validation),heldout_ids=len(heldout),validation=result,selected_primary=primary,selected_accuracy=accuracy,source_only=True,prior_uses_F_pseudo=True,DURA_inference=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='source physical IDs previously observed in other views; clean F pseudo targets are not true accuracy; no target predictions')
 with (a.out/'source_validation.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
 print('BLIND_VALIDATION_COMPLETE',json.dumps(dict(primary=primary,accuracy=accuracy)),flush=True)
if __name__=='__main__':main()

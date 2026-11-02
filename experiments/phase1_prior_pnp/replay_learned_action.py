"""Fixed source evaluation saving all learned-action restoration predictions."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pipeline_learned_action import load_fixed,evaluate_wave,s,core
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from validate_free_full import source_prior,choose,VIEWS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 models=load_fixed(a.root,a.device);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);state=torch.load(a.root/'runs/20261102-phase1-observation-unroll-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if np.intersect1d(state['fit_ids'],np.r_[validation,heldout]).size:raise ValueError('Source validation must be disjoint from prior training')
 received,scene=s.epoch_data(cache,19)
 if a.smoke:
  x=core.tensor(received[validation[:8]],a.device);z,c,on=evaluate_wave(x,models)
  if not torch.isfinite(z).all() or not torch.isfinite(c).all():raise ValueError('Finite fixed source inference required')
  core.write(a.out/'result.json',dict(status='PASS',records=len(x),source_only=True,DURA_in_candidate=False,test_reads=0));print('LEARNED_ACTION_SOURCE_SMOKE PASS',flush=True);return
 dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];out=[];tick=time.monotonic();scales=(.25,.5,1.)
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);dp=teacher.classify_features(teacher.features(d)).argmax(1);prob=models[3](x).sigmoid();parent,cand,_=evaluate_wave(x,models);pp=teacher.classify_features(teacher.features(parent)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);_,pe=s.align(parent,y);cols=[target.float(),conf,dp.float(),prob,de/den,raw.float(),re/den,pp.float(),pe/den]
   for scale in scales:
    wave=parent+scale*(cand-parent);pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,ce=s.align(wave,y);cols.extend((pred.float(),ce/den))
   batch=torch.stack(cols,1).cpu()
   if not torch.isfinite(batch).all():raise ValueError('Finite source results required')
   out.append(batch)
   if (start+len(idx))%512==0:print('LEARNED_ACTION_SOURCE_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(out).contiguous());saved=a.out/'source_predictions.npz'
 with saved.open('xb') as f:np.savez(f,indices=ids,split=split,rx=rx[ids],scene=scene[ids],values=values)
 core.write(a.out/'prediction_complete.json',dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),source_only=True,DURA_comparator_only=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0))
 results={}
 for i,scale in enumerate(scales):
  result={}
  for threshold in (.8,.99):
   cells={}
   for part,receivers in ((1,(4,6)),(2,(8,))):
    for receiver in receivers:
     for view in VIEWS:
      mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
      if not len(v):raise ValueError('All source cells required')
      trusted=v[:,1]>=.95;active=v[:,3]>threshold;par=np.where(active,v[:,7],v[:,5]);piq=np.where(active,v[:,8],v[:,6]);pred=np.where(active,v[:,9+2*i],v[:,5]);iq=np.where(active,v[:,10+2*i],v[:,6])
      cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=int(np.count_nonzero(trusted&(v[:,2]==v[:,0]))),parent_correct=int(np.count_nonzero(trusted&(par==v[:,0]))),candidate_correct=int(np.count_nonzero(trusted&(pred==v[:,0]))),baseline_iq=float(np.mean(v[:,4])),parent_iq=float(np.mean(piq)),candidate_iq=float(np.mean(iq)))
   result[str(threshold)]=cells
  results[str(scale)]=result
 selection={k:{name:choose(v,req,ref) for name,req,ref in (('primary_dura',True,'baseline'),('accuracy_dura',False,'baseline'),('primary_parent',True,'parent'),('accuracy_parent',False,'parent'))} for k,v in results.items()}
 summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,validation_ids=len(validation),heldout_ids=len(heldout),validation=results,selection=selection,source_only=True,F_frozen=True,DURA_comparator_inference=True,DURA_in_candidate=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source IDs previously explored; clean F pseudo-targets not truth; no ground test.')
 core.write(a.out/'source_validation.json',summary);print('LEARNED_ACTION_SOURCE_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()

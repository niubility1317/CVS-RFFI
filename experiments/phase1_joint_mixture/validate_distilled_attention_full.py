"""Predeclared source-only validation of direct teacher-guided IQ attention network."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from attention_restorer import AttentionRestorer
from validate_free_full import s,core,source_prior,choose,VIEWS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 run=a.root/'runs/20261102-phase1-distilled-attention-full-s2026092701-r01';complete=core.read(run/'result.json');state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
 if complete.get('status')!='TRAINING_COMPLETE' or state.get('commit')!='4066269354c6192e4592cafaad1cb77ac12babf7':raise ValueError('Completed fixed source checkpoint required')
 net=AttentionRestorer().to(a.device);net.load_state_dict(state['model']);net.eval().requires_grad_(False);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph)
 if np.intersect1d(state['fit_ids'],np.r_[validation,heldout]).size:raise ValueError('Source validation excluded from fit required')
 received,scene=s.epoch_data(cache,19);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False)
 if a.smoke:
  x=core.tensor(received[validation[:8]],a.device);y=net(x)
  if not torch.isfinite(y).all():raise ValueError('Finite waveform required')
  core.write(a.out/'result.json',dict(status='PASS',records=len(x),source_only=True,test_reads=0));print('DISTILLED_FULL_ATTENTION_VALIDATION_SMOKE PASS',flush=True);return
 dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];tick=time.monotonic();outputs=[];scales=(.25,.5,1.)
 with torch.no_grad():
  for start in range(0,len(ids),16):
   ix=ids[start:start+16];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);dp=teacher.classify_features(teacher.features(d)).argmax(1);prob=gate(x).sigmoid();predicted=net(x);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);cols=[target.float(),conf,dp.float(),prob,de/den,raw.float(),re/den]
   for scale in scales:
    wave=x+scale*(predicted-x);pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,e=s.align(wave,y);cols.extend((pred.float(),e/den))
   z=torch.stack(cols,1).cpu()
   if not torch.isfinite(z).all():raise ValueError('Finite validation output required')
   outputs.append(z)
   if (start+len(ix))%512==0:print('DISTILLED_FULL_ATTENTION_VALIDATION_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(outputs).contiguous());results={}
 for i,scale in enumerate(scales):
  result={}
  for threshold in (.8,.99):
   cells={}
   for part,receivers in ((1,(4,6)),(2,(8,))):
    for receiver in receivers:
     for view in VIEWS:
      mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
      if not len(v):raise ValueError('All source receiver/scene cells required')
      trusted=v[:,1]>=.95;active=v[:,3]>threshold;pred=np.where(active,v[:,7+2*i],v[:,5]);iq=np.where(active,v[:,8+2*i],v[:,6])
      cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=int(np.count_nonzero(trusted&(v[:,2]==v[:,0]))),candidate_correct=int(np.count_nonzero(trusted&(pred==v[:,0]))),baseline_iq=float(np.mean(v[:,4])),candidate_iq=float(np.mean(iq)))
   result[str(threshold)]=cells
  results[str(scale)]=result
 selection={k:{'primary_dura':choose(v,True,'baseline'),'accuracy_dura':choose(v,False,'baseline')} for k,v in results.items()}
 summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,validation_ids=len(validation),heldout_ids=len(heldout),validation=results,selection=selection,source_only=True,F_frozen=True,DURA_comparator_inference=True,candidate_uses_DURA=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Previously explored source IDs; clean F pseudo targets are not truth; no target prediction.')
 core.write(a.out/'source_validation.json',summary);print('DISTILLED_FULL_ATTENTION_VALIDATION_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()

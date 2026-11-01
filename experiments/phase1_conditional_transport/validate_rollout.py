"""Fixed source-only receiver/scenario validation of four-step transport."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_transport import s,core,source_prior,BlindActionNet,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,initial
from transport_model import ConditionalTransport,integrate
SEL=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-source-validation-s2026092701-r01-release')
if (SEL/'release_commit.txt').read_text().strip()!='86acfb414d70c638ef086e318739cc158a9b6601':raise ValueError('Exact source selection contract required')
sys.path.insert(0,str(SEL/'experiments/phase1_bayes_residual_validation'))
from residual_selection import choose,VIEWS
RUN='20261102-phase1-conditional-transport-rollout-s2026092701-r01'
NET_SHA='c47bb4ad18cd31e3e06e7e92f1faee4262cee74e599ebdd246a061425cd243ec'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ckpt=a.root/'runs'/RUN/'checkpoint_epoch03.pt';blind_ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 for path,expected in ((ckpt,NET_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exact fixed source checkpoint required')
 netstate=torch.load(ckpt,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False)
 if netstate.get('commit')!='d9f4a4072b581bd12d9ae66eb4c60c06081b7df0' or netstate.get('training_mode')!='rollout4':raise ValueError('Expected rollout training checkpoint')
 net=ConditionalTransport().to(a.device);net.load_state_dict(netstate['model']);net.eval().requires_grad_(False);blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False)
 if a.smoke:
  x=core.tensor(received[validation[:8]],a.device);parent,A,lv=initial(blind,x,means,covs,priors,eye);choices=[integrate(net,x,parent,A,lv,k) for k in (1,4,8)]
  if not all(torch.isfinite(v).all() for v in (parent,A,lv,*choices)):raise ValueError('Finite fixed source inference required')
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(x),source_only=True,test_reads=0),f,indent=2)
  print('TRANSPORT_VALIDATION_SMOKE PASS',flush=True);return
 dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];out=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);base=teacher.classify_features(teacher.features(d)).argmax(1);prob=gate(x).sigmoid();enabled=prob>.8;z=x.clone();cand=[x.clone() for _ in range(3)]
   if enabled.any():
    x2=x[enabled];parent,A,lv=initial(blind,x2,means,covs,priors,eye);z[enabled]=parent
    for j,steps in enumerate((1,4,8)):
     cand[j][enabled]=integrate(net,x2,parent,A,lv,steps)
   pp=teacher.classify_features(teacher.features(z)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);_,pe=s.align(z,y);cols=[target.float(),conf,base.float(),prob,de/den,raw.float(),re/den,pp.float(),pe/den]
   for c in cand:
    pred=teacher.classify_features(teacher.features(c)).argmax(1);_,ce=s.align(c,y);cols.extend((pred.float(),ce/den))
   batch=torch.stack(cols,1).cpu()
   if not torch.isfinite(batch).all():raise ValueError('Finite source evaluation required')
   out.append(batch)
   if (start+len(idx))%512==0:print('TRANSPORT_VALIDATION_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(out).contiguous());results={}
 for i,k in enumerate((1,4,8)):
  result={}
  for threshold in (.8,.99):
   cells={}
   for part,receivers in ((1,(4,6)),(2,(8,))):
    for receiver in receivers:
     for view in VIEWS:
      mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
      if not len(v):raise ValueError('All source receiver/scene cells required')
      trusted=v[:,1]>=.95;active=v[:,3]>threshold;parentpred=np.where(active,v[:,7],v[:,5]);parentiq=np.where(active,v[:,8],v[:,6]);pred=np.where(active,v[:,9+2*i],v[:,5]);iq=np.where(active,v[:,10+2*i],v[:,6])
      cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=int(np.count_nonzero(trusted&(v[:,2]==v[:,0]))),parent_correct=int(np.count_nonzero(trusted&(parentpred==v[:,0]))),candidate_correct=int(np.count_nonzero(trusted&(pred==v[:,0]))),baseline_iq=float(np.mean(v[:,4])),parent_iq=float(np.mean(parentiq)),candidate_iq=float(np.mean(iq)))
   result[str(threshold)]=cells
  results[str(k)]=result
 selection={}
 for k,result in results.items():
  selection[k]={name:choose(result,require_iq,ref) for name,require_iq,ref in (('primary_dura',True,'baseline'),('accuracy_dura',False,'baseline'),('primary_parent',True,'parent'),('accuracy_parent',False,'parent'))}
 summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,model_sha256=NET_SHA,validation_ids=len(validation),heldout_ids=len(heldout),validation=results,selection=selection,source_only=True,F_frozen=True,DURA_inference=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source IDs previously used in other views; clean F pseudo targets not truth; no target prediction.')
 with (a.out/'source_validation.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
 print('TRANSPORT_VALIDATION_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()

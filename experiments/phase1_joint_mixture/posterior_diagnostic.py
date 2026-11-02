"""Source-only diagnostics for the learned posterior mixture; no test truth."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from source_pilot import s,core,BlindActionNet,physical,SOURCE,PRIOR_RUN,SCENES
from mixture_controller import MixtureController

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 bs=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False);st=torch.load(a.root/'runs/20261102-phase1-joint-mixture-pilot-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if bs.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or bs.get('epoch')!=10 or p.get('source_ids')!=8192 or st.get('commit')!='c67dc6d7b6a3880ffe2fe4e49357ee4da18eb4ef':raise ValueError('Fixed source model contract required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(bs['model']);blind.eval().requires_grad_(False);prior=(p['means'].to(a.device),p['covariances'].to(a.device),p['priors'].to(a.device));net=MixtureController().to(a.device);net.load_state_dict(st['model']);net.eval().requires_grad_(False);eye=torch.eye(512,device=a.device);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120355);_prior=np.sort(rng.choice(training,size=8192,replace=False));ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);ids=ids[:16] if a.smoke else ids
 out=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);states,lp,v,parent=physical(blind,teacher,x,prior,eye);pred0=teacher.classify_features(teacher.features(parent.reshape_as(x))).argmax(1);candidate,w,corr=net(v,lp,states);pred1=teacher.classify_features(teacher.features(candidate)).argmax(1);p0=lp.exp();top0=p0.argmax(1);top1=w.argmax(1);entropy=-(p0*lp).sum(1);tv=.5*(p0-w).abs().sum(1);oracle=(v[:,:,326:332].argmax(2)==target[:,None]).any(1);den=y.flatten(1).square().sum(1).clamp_min(1e-12);error0=(parent-y.flatten(1)).square().sum(1)/den;error1=(candidate-y).flatten(1).square().sum(1)/den
   row=torch.stack((conf,target.float(),pred0.float(),pred1.float(),p0.max(1).values,entropy,tv,(top0!=top1).float(),corr.abs().amax(1),oracle.float(),error0,error1),1)
   if not torch.isfinite(row).all():raise ValueError('Finite source diagnostic required')
   out.append(row.cpu())
 values=np.from_dlpack(torch.cat(out).contiguous())
 if a.smoke:
  print('JOINT_POSTERIOR_DIAGNOSTIC_SMOKE PASS',len(values),flush=True);return
 result={}
 for i,view in enumerate(SCENES):
  v=values[i*64:(i+1)*64];valid=v[:,0]>=.95;wrong=valid&(v[:,2]!=v[:,1]);n=int(valid.sum());r=dict(count=len(v),valid_count=n,parent_correct=int(np.count_nonzero(valid&(v[:,2]==v[:,1]))),candidate_correct=int(np.count_nonzero(valid&(v[:,3]==v[:,1]))),repairs=int(np.count_nonzero(wrong&(v[:,3]==v[:,1]))),damages=int(np.count_nonzero(valid&(v[:,2]==v[:,1])&(v[:,3]!=v[:,1]))),parent_wrong=int(wrong.sum()),parent_wrong_with_any_hypothesis_matching_pseudotarget=int(np.count_nonzero(wrong&(v[:,9]>0))),max_weight_median=float(np.median(v[:,4])),parent_entropy_median=float(np.median(v[:,5])),weight_tv_median=float(np.median(v[:,6])),weight_tv_mean=float(np.mean(v[:,6])),max_weight_over_0_95=int(np.count_nonzero(v[:,4]>.95)),top_component_changes=int(np.count_nonzero(v[:,7])),max_logit_correction_median=float(np.median(v[:,8])),parent_iq_mean=float(np.mean(v[:,10])),candidate_iq_mean=float(np.mean(v[:,11])))
  if n==0:raise ValueError('Valid source pseudo target required')
  result[view]=r
 output=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',commit=a.commit,count=len(ids),scenes=result,source_only=True,ungated=True,F_frozen=True,candidate_uses_DURA=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Previously explored source IDs, clean F pseudo targets are not actual test accuracy. Oracle uses pseudo target and is not deployable; it applies only to six fixed component waves.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
 print('JOINT_POSTERIOR_DIAGNOSTIC_COMPLETE',json.dumps(output),flush=True)
if __name__=='__main__':main()

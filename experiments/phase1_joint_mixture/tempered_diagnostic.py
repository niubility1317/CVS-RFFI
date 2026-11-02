"""Source-only fixed temperature and correction diagnostic; no test truth."""
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
 temperatures=(.01,.03,.1,.3,1.);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);states,lp,v,parent=physical(blind,teacher,x,prior,eye);correction=net.network(v.reshape(len(x),-1));den=y.flatten(1).square().sum(1).clamp_min(1e-12);cols=[conf,target.float()]
   for alpha in temperatures:
    for beta in (0.,1.):
     weight=F.softmax(alpha*lp+beta*correction,1);wave=(states*weight.unsqueeze(-1)).sum(1).reshape_as(x);pred=teacher.classify_features(teacher.features(wave)).argmax(1);iq=(wave-y).square().sum((1,2))/den;cols.extend((pred.float(),iq,weight.max(1).values))
   row=torch.stack(cols,1)
   if not torch.isfinite(row).all():raise ValueError('Finite source evaluation required')
   out.append(row.cpu())
 values=np.from_dlpack(torch.cat(out).contiguous())
 if a.smoke:print('TEMPERED_POSTERIOR_SMOKE PASS',len(values),flush=True);return
 scenes={}
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum()),candidates={})
  for i,alpha in enumerate(temperatures):
   for k,beta in enumerate((0.,1.)):
    off=2+3*(2*i+k);cell['candidates'][f'{alpha}:{beta}']=dict(correct=int(np.count_nonzero(valid&(rows[:,off]==rows[:,1]))),mean_iq_nmse=float(rows[:,off+1].mean()),max_weight_median=float(np.median(rows[:,off+2])))
  scenes[view]=cell
 result=dict(status='SOURCE_TEMPERATURE_DIAGNOSTIC_COMPLETE',commit=a.commit,count=len(ids),scenes=scenes,alpha_values=list(temperatures),learned_correction_scales=[0.,1.],source_only=True,ungated=True,F_frozen=True,DURA_used=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Temperatures proposed after examining same source records. Pseudo labels are not test truth; softening may harm physical fidelity. No target test or model selection.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('TEMPERED_POSTERIOR_DIAGNOSTIC_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

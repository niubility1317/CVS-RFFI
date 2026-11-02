"""Source-only paired flow matching; ground labels and DURA are not used."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from raw_conditioned_flow import RawConditionedFlow
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 torch.set_num_threads(2);torch.manual_seed(2026150054);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);fit=np.sort(np.random.default_rng(2026150055).choice(training,size=4096,replace=False));roles=np.load(cache/'train/roles.npy',allow_pickle=False)
 if np.intersect1d(fit,np.r_[validation,heldout]).size or not np.all(np.isin(roles[fit],['L_s','U_s'])):raise ValueError('Disjoint authorized source fit IDs required')
 if a.smoke:fit=np.sort(np.r_[fit[roles[fit]=='L_s'][:16],fit[roles[fit]=='U_s'][:16]])
 if a.smoke and len(fit)!=32:raise ValueError('Mixed source-role smoke required')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Common source IDs required')
 epochs=(21,) if a.smoke else (21,22,23)
 for epoch in epochs:
  info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
 net=RawConditionedFlow();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(2026150056);losses=[];counts=[];tick=time.monotonic()
 for epoch in epochs:
  rec,scene=s.epoch_data(fresh,epoch);order=rng.permutation(fit);total=0.;n=0
  for start in range(0,len(order),16):
   ids=order[start:start+16];x=core.tensor(rec[ids],'cpu');y=core.tensor(clean[ids],'cpu');t=torch.rand(len(ids));state=(1-t[:,None,None])*x+t[:,None,None]*y;v=net(state,x,t);den=(y.square().sum((1,2))).clamp_min(1e-8);loss=(((v-(y-x)).square().sum((1,2)))/den).mean()
   if not torch.isfinite(loss):raise FloatingPointError('Finite paired flow objective required')
   opt.zero_grad(set_to_none=True);loss.backward()
   if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or not net.output.weight.grad.abs().sum():raise ValueError('Finite nonzero flow output gradient required')
   torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();n+=len(ids);total+=float(loss.detach())*len(ids)
  losses.append(total/n);counts.append(n);print('RAW_FLOW_EPOCH',epoch,n,losses[-1],round(time.monotonic()-tick,1),flush=True)
 teacher,contract=core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);received,scene=s.epoch_data(cache,19);views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];r=np.random.default_rng(2026150057);chosen=[];groups=[]
 for name,group in [('RX4_6',validation),('RX8',heldout)]:
  for view in views:
   pool=group[scene[group]==view];amount=2 if a.smoke else 48
   if len(pool)<amount:raise ValueError('Source scene group too small')
   part=np.sort(r.choice(pool,size=amount,replace=False));chosen.extend(part.tolist());groups.extend([name]*amount)
 ids=np.asarray(chosen,dtype=np.int64)
 if len(np.unique(ids))!=len(ids) or np.intersect1d(ids,fit).size:raise ValueError('Unique disjoint source evaluation required')
 net.eval().requires_grad_(False);rows=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   ix=ids[start:start+8];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');out=net.restore(x,4);logit=teacher.classify_features(teacher.features(y));conf,pseudo=F.softmax(logit,1).max(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);vals=[conf,pseudo.float()]
   for signal in (x,out):
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);_,err=s.align(signal,y);vals.extend((pred.float(),err/den))
   rows.append(torch.stack(vals,1).cpu())
 values=np.from_dlpack(torch.cat(rows).contiguous())
 if values.shape!=(len(ids),6) or not np.isfinite(values).all():raise ValueError('Finite fixed source predictions required')
 a.out.mkdir(parents=True);checkpoint=a.out/'checkpoint.pt'
 with checkpoint.open('xb') as f:torch.save(dict(commit=a.commit,model=net.state_dict(),fit_ids=fit,losses=losses),f)
 path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],group=np.asarray(groups),values=values)
 result=dict(status='SOURCE_RAW_FLOW_SMOKE_PASS' if a.smoke else 'SOURCE_RAW_FLOW_TRAINING_COMPLETE',commit=a.commit,fit_ids=len(fit),counts=counts,losses=losses,evaluation_count=len(ids),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),F_frozen=True,F_training_forward_calls=0,DURA_loaded=False,blind_parent_loaded=False,L_s_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,steps=4,caveat='Previously explored source records; clean F pseudo targets are not identity truth; no ground test or real satellite claim.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()

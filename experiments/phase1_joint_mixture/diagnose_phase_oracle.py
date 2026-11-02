"""Source-only clean-paired phase/gain oracle; NOT a deployable restorer."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from validate_free_full import s,core,source_prior,VIEWS

def oracle(x,y):
 xi,xq=x[:,0],x[:,1];yi,yq=y[:,0],y[:,1];den=(xi.square()+xq.square()).sum(1).clamp_min(1e-12);ar=(xi*yi+xq*yq).sum(1)/den;ai=(xi*yq-xq*yi).sum(1)/den;norm=(ar.square()+ai.square()).sqrt().clamp_min(1e-12)
 def rotate(a,b):return torch.stack((a[:,None]*xi-b[:,None]*xq,b[:,None]*xi+a[:,None]*xq),1)
 return rotate(ar/norm,ai/norm),rotate(ar,ai)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);val,hold=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026130155);val=val[scene[val]!='clean'];hold=hold[scene[hold]!='clean'];ids=[np.sort(rng.choice(z,1024,replace=False)) for z in (val,hold)]
 teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);result={}
 for name,ixs in zip(('validation_RX4_6','heldout_RX8'),ids):
  rows=[]
  with torch.no_grad():
   for start in range(0,len(ixs),32):
    ix=ixs[start:start+32];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);phase,gain=oracle(x,y);waves=(x,phase,gain,dura(x));den=y.square().sum((1,2)).clamp_min(1e-12);cols=[target.float(),conf]
    for wave in waves:
     pred=teacher.classify_features(teacher.features(wave)).argmax(1);iq=(wave-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
    rows.append(torch.stack(cols,1).cpu())
  v=np.from_dlpack(torch.cat(rows).contiguous());trusted=v[:,1]>=.95;cell={'count':len(ixs),'valid_count':int(trusted.sum()),'scene_counts':{view:int(np.count_nonzero(scene[ixs]==view)) for view in VIEWS if view!='clean'}}
  for j,method in enumerate(('raw','oracle_phase','oracle_gain_phase','dura')):cell[method]={'pseudo_correct':int(np.count_nonzero(trusted&(v[:,2+2*j]==v[:,0]))),'direct_iq_nmse':float(v[:,3+2*j].mean())}
  result[name]=cell;print('PHASE_ORACLE_SOURCE',name,json.dumps(cell),flush=True)
 core.write(a.out/'result.json',dict(status='SOURCE_ORACLE_DIAGNOSTIC_COMPLETE',commit=a.commit,results=result,source_only=True,clean_pair_used_at_diagnostic=True,deployable=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Per-record clean-paired least-squares coefficients are unavailable at deployment. Pseudo targets are not actual accuracy; source views previously explored.'))
if __name__=='__main__':main()

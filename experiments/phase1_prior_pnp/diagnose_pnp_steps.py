"""Source-only decomposition of blind posterior, observation and prior updates."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from signal_prior import observation_step
from pipeline import load_fixed,evaluate_wave,s,core,action_matrix,posterior
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import source_prior

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(a.root,a.device);blind,prior,stats,gate,eye=models;teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);cache,clean,rx,training,pv,ph=s.source_contract(a.root);valid,held=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026126055)
 cohorts=[]
 for part,pool in ((1,valid),(2,held)):
  pool=pool[scene[pool]!='clean'];take=16 if a.smoke else 1024
  if len(pool)<take:raise ValueError('Complete source cohorts required')
  ids=np.sort(rng.choice(pool,take,replace=False));cohorts.append((part,ids))
 dura=source_prior.frozen_dura(a.root,a.device);indices=[];parts=[];values=[];tick=time.monotonic()
 with torch.no_grad():
  for part,ids in cohorts:
   for start in range(0,len(ids),8):
    ix=ids[start:start+8];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);active=gate(x).sigmoid()>.8;par=x.clone();ob=x.clone();de=x.clone();both=x.clone()
    if active.any():
     z=x[active];h,lv=blind(z);A=action_matrix(h);states,lp=posterior(z.reshape(len(z),512),A,eye[None]*lv.exp()[:,None,None],*stats);initial=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(z);o=initial.clone();d=initial.clone();b=initial.clone()
     for sig in (.16,.08,.03):
      sigma=initial.new_full((len(initial),),sig);o=observation_step(z,o,A);d=prior(d,sigma);b=prior(observation_step(z,b,A),sigma)
     par[active]=initial;ob[active]=o;de[active]=d;both[active]=b
    if start==0:
     refp,refb,refa=evaluate_wave(x,models);torch.testing.assert_close(par,refp,rtol=0,atol=0);torch.testing.assert_close(both,refb,rtol=0,atol=0);assert torch.equal(active,refa)
    waves=(x,dura(x),par,ob,de,both);den=y.square().sum((1,2)).clamp_min(1e-12);center=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);cols=[target.float(),conf,active.float()]
    for w in waves:
     prediction=teacher.classify_features(teacher.features(w)).argmax(1);direct=(w-y).square().sum((1,2))/den;_,err=s.align(w,y);cols.extend((prediction.float(),direct,err/center))
    e=par-y;lag=(e[:,:,:-1]*e[:,:,1:]).sum((1,2))/e.square().sum((1,2)).clamp_min(1e-12);iq=(e[:,0]*e[:,1]).sum(1)/(e[:,0].square().sum(1)*e[:,1].square().sum(1)).clamp_min(1e-20).sqrt();cols.extend((lag,iq));batch=torch.stack(cols,1).cpu()
    if not torch.isfinite(batch).all():raise ValueError('Finite source diagnostic required')
    indices.append(ix);parts.extend([part]*len(ix));values.append(batch)
   print('PNP_STEP_DIAGNOSTIC_PROGRESS',part,len(ids),'seconds',round(time.monotonic()-tick,1),flush=True)
 indices=np.concatenate(indices);values=np.from_dlpack(torch.cat(values).contiguous());parts=np.asarray(parts,dtype=np.int64);path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=indices,part=parts,rx=rx[indices],scene=scene[indices],values=values)
 core.write(a.out/'prediction_complete.json',dict(status='SOURCE_DIAGNOSTIC_FIXED',commit=a.commit,count=len(indices),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_only=True,DURA_comparator_only=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('PNP_STEP_DIAGNOSTIC_FIXED',len(indices),flush=True)
if __name__=='__main__':main()

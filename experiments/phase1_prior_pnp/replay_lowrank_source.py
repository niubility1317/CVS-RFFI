"""Replay fixed 128/256-rank independent unroll over held-out source IDs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pipeline_unroll as p
from lowrank_posterior import factors_from_covariance,lowrank_posterior

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=p.load_fixed(a.root,'cpu');teacher,_=p.core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);cache,clean,rx,training,pv,ph=p.s.source_contract(a.root);validation,heldout=p.s.new_splits(cache,rx,training,pv,ph);ids=np.r_[validation,heldout];fit=torch.load(a.root/'runs/20261102-phase1-observation-unroll-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)['fit_ids']
 if np.intersect1d(ids,fit).size or len(ids)!=11761:raise ValueError('Disjoint full source cohort required')
 old=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';old_m=p.core.read(old/'prediction_complete.json');old_path=old/'source_predictions.npz'
 if old_m.get('status')!='SOURCE_PREDICTIONS_FIXED' or hashlib.sha256(old_path.read_bytes()).hexdigest()!=old_m.get('sha256') or any(old_m.get(x)!=0 for x in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact prior source replay required')
 with np.load(old_path,allow_pickle=False) as f:old_ids=f['indices'].copy();old_values=f['values'].copy();old_scene=f['scene'].copy()
 received,scene=p.s.epoch_data(cache,19)
 if not np.array_equal(old_ids,ids) or not np.array_equal(old_scene,scene[ids]) or old_values.shape!=(len(ids),15):raise ValueError('Source cohort and prior replay must match')
 n=64 if a.smoke else len(ids);blind,prior,stats,gate,eye=models;means,cov,priors=stats;factors={k:factors_from_covariance(cov,k) for k in (128,256)};outputs=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,n,32):
   ix=ids[start:start+32];x=p.core.tensor(received[ix],'cpu');y=p.core.tensor(clean[ix],'cpu');raw=teacher.classify_features(teacher.features(x)).argmax(1);active=gate(x).sigmoid()>.8;raw_np=np.from_dlpack(raw.cpu().contiguous());active_np=np.from_dlpack(active.cpu().contiguous());ref=old_values[start:start+len(ix)]
   if not np.array_equal(raw_np,ref[:,5]) or not np.array_equal(active_np,ref[:,3]>.8):raise ValueError('CPU source raw F and gate must match saved GPU source replay')
   den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);batch=[]
   for rank in (128,256):
    wave=x.clone()
    if active.any():
     signal=x[active];h,lv=blind(signal);A=p.action_matrix(h);states,lp=lowrank_posterior(signal.reshape(len(signal),512),A,lv.exp(),means,factors[rank],priors);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);wave[active]=p.restore_observation_conditioned(prior,signal,z,A,lv)
    pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,err=p.s.align(wave,y);batch.extend((pred.float(),err/den))
   values=np.from_dlpack(torch.stack(batch,1).cpu().contiguous())
   if not np.isfinite(values).all():raise ValueError('Finite source output required')
   outputs.append(values)
   if (start+len(ix))%512==0:print('LOWRANK_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 result=np.concatenate(outputs)
 if a.smoke:
  p.core.write(a.out/'result.json',dict(status='PASS',count=n,source_only=True,raw_and_gate_match=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('LOWRANK_SOURCE_SMOKE PASS',flush=True);return
 path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],prior_values=old_values,lowrank_values=result)
 m=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=n,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prior_sha256=old_m['sha256'],source_only=True,raw_and_gate_match=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0);p.core.write(a.out/'prediction_complete.json',m);print('LOWRANK_SOURCE_REPLAY_COMPLETE',json.dumps(m),flush=True)
if __name__=='__main__':main()

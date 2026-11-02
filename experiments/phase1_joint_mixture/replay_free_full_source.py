"""Replay fixed independent direct mixture on the complete source cohort."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import phase1_free as p

def predict(models,received,ids,batch,threads):
 torch.set_num_threads(threads);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),batch):
   ix=ids[start:start+batch];raw,parent,candidate,active,choice=p.predict_batch(p.core.tensor(received[ix],'cpu'),models);out.append(torch.stack((raw,parent,candidate,active.long(),choice),1).cpu())
 return np.from_dlpack(torch.cat(out).contiguous())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source replay required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);cache,clean,rx,training,pv,ph=p.s.source_contract(a.root);validation,heldout=p.s.new_splits(cache,rx,training,pv,ph);ids=np.r_[validation,heldout];received,scene=p.s.epoch_data(cache,19);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=p.core.read(prior/'prediction_complete.json');path=prior/'source_predictions.npz';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 if len(ids)!=11761 or pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or sha(path)!=pm['sha256'] or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source reference required')
 with np.load(path,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx[ids],z['rx']) or not np.array_equal(scene[ids],z['scene']):raise ValueError('Exact source IDs and views required')
  old=z['values'].copy()
 fit_prior=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));pool=np.setdiff1d(training,fit_prior,assume_unique=True);fit=np.sort(np.random.default_rng(2026121855).choice(pool,size=4096,replace=False))
 if np.intersect1d(ids,np.r_[fit_prior,fit]).size:raise ValueError('Source cohort must be disjoint from physical prior and candidate fit IDs')
 ref=a.root/'runs/20261102-phase1-free-mixture-labeled-source-s2026092701-r01';rm=p.core.read(ref/'prediction_complete.json');rp=ref/'predictions.npz'
 if rm.get('status')!='SOURCE_PREDICTIONS_FIXED' or rm.get('count')!=1047 or rm.get('role')!='L_s' or rm.get('predictions_sha256')!='f63336f65fb396a090be84a4764cedc7671a0beed055d7919314650eeda75253' or sha(rp)!=rm['predictions_sha256'] or any(rm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed L_s source reference required')
 with np.load(rp,allow_pickle=False) as z:ref_ids=z['indices'].copy();ref_pred=z['predictions'][:,4].copy()
 order={int(x):i for i,x in enumerate(ids)};rows=np.asarray([order[int(x)] for x in ref_ids]);models=p.load_fixed(a.root,'cpu');checkpoint=a.root/'runs/20261102-phase1-free-mixture-pilot-s2026092701-r01/checkpoint.pt';model_sha=sha(checkpoint);target=ids[:64] if a.smoke else ids;chunks=[];tick=time.monotonic()
 if a.smoke:
  z8=predict(models,received,target,8,2);z32=predict(models,received,target,32,8)
  if not np.array_equal(z8,z32):raise ValueError('Source batch/thread decisions must match')
 for start in range(0,len(target),512):
  ix=target[start:start+512];z=predict(models,received,ix,32,8);reference=old[start:start+len(ix)]
  if not np.array_equal(z[:,0],reference[:,5]) or not np.array_equal(z[:,1],reference[:,7]) or not np.array_equal(z[:,3],reference[:,3]>.8):raise ValueError('Raw F, physical parent and source gate must match per record')
  chunks.append(z[:,2])
  if (start+len(ix))%2048==0:print('FREE_FULL_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 output=np.concatenate(chunks)
 if not a.smoke and not np.array_equal(output[rows],ref_pred):raise ValueError('Fixed L_s candidate decisions must match')
 if a.smoke:
  p.core.write(a.out/'result.json',dict(status='PASS',count=len(target),raw_parent_gate_match=True,batch_thread_decisions_match=True,source_only=True,DURA_in_candidate=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],prior_values=old,candidate_prediction=output)
 result=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=sha(path),prior_sha256=pm['sha256'],ls_reference_sha256=rm['predictions_sha256'],model_sha256=model_sha,raw_parent_gate_match=True,source_L_s_replay_match=True,source_only=True,DURA_in_candidate=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,caveat='Source views previously explored; no blind or ground test claim.')
 p.core.write(a.out/'prediction_complete.json',result);print('FREE_FULL_SOURCE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

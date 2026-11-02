"""Replay three fixed DURA-free complementary restorers on all source records."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import replay_complementary_source as base

NAMES=['attention','structured_prior','white_prior']
def predict(name,models,received,ids,batch,threads):
 torch.set_num_threads(threads);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),batch):
   ix=ids[start:start+batch];raw,pred,active=base.predict(name,base.core.tensor(received[ix],'cpu'),models);out.append(torch.stack((raw,pred,active.long()),1).cpu())
 return np.from_dlpack(torch.cat(out).contiguous())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source prediction output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);cache,clean,rx,training,pv,ph=base.s.source_contract(a.root);validation,heldout=base.s.new_splits(cache,rx,training,pv,ph);ids=np.r_[validation,heldout];received,scene=base.s.epoch_data(cache,19);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=base.core.read(prior/'prediction_complete.json');pp=prior/'source_predictions.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 if len(ids)!=11761 or pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('count')!=11761 or pm.get('sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or sha(pp)!=pm['sha256'] or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact source replay required')
 with np.load(pp,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx[ids],z['rx']) or not np.array_equal(scene[ids],z['scene']):raise ValueError('Same source IDs and views required')
  old=z['values'].copy()
 ref=a.root/'runs/20261102-phase1-complementary-source-replay-s2026092701-r02';rm=base.core.read(ref/'prediction_complete.json');rp=ref/'source_predictions.npz'
 if rm.get('status')!='SOURCE_PREDICTIONS_FIXED' or rm.get('count')!=1047 or rm.get('candidate_names')!=NAMES or rm.get('sha256')!='cf9d04b1ae7a4bcaa1114054b4c75fc71ee4fae861863173b3fa86285e60910c' or sha(rp)!=rm['sha256'] or any(rm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Exact source L_s reference predictions required')
 with np.load(rp,allow_pickle=False) as z:ref_ids=z['indices'].copy();ref_values=z['candidate_predictions'].copy()
 order={int(x):i for i,x in enumerate(ids)};rows=np.asarray([order[int(x)] for x in ref_ids]);fit_runs=['20261102-phase1-attention-waveform-pilot-s2026092701-r02','20261102-phase1-structured-iq-prior-s2026092701-r01','20261102-phase1-independent-iq-prior-s2026092701-r01'];hashes={}
 for name,run in zip(NAMES,fit_runs):
  path=a.root/'runs'/run/'checkpoint.pt';state=torch.load(path,map_location='cpu',weights_only=False)
  if np.intersect1d(ids,state['fit_ids']).size:raise ValueError('All replay source IDs must be disjoint from restoration fit: '+name)
  hashes[name]=sha(path)
 target=ids[:64] if a.smoke else ids;outputs=[];tick=time.monotonic()
 for col,name in enumerate(NAMES):
  torch.set_num_threads(2);models=base.load(name,a.root)
  if a.smoke:
   b8=predict(name,models,received,target,8,2);b32=predict(name,models,received,target,32,8)
   if not np.array_equal(b8,b32):raise ValueError('Batch/thread source decisions must match: '+name)
  chunks=[]
  for start in range(0,len(target),512):
   ix=target[start:start+512];r=predict(name,models,received,ix,32,8);reference=old[start:start+len(ix)]
   if not np.array_equal(r[:,0],reference[:,5]) or not np.array_equal(r[:,2],reference[:,3]>.8):raise ValueError('Common F and gate must match per record: '+name)
   chunks.append(r[:,1])
   if (start+len(ix))%2048==0:print('COMPLEMENT_FULL_SOURCE_PROGRESS',name,start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
  values=np.concatenate(chunks)
  if not a.smoke and not np.array_equal(values[rows],ref_values[:,col]):raise ValueError('Fixed L_s candidate replay mismatch: '+name)
  outputs.append(values);print('COMPLEMENT_FULL_SOURCE_DONE',name,len(values),'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  base.core.write(a.out/'result.json',dict(status='PASS',count=len(target),candidate_names=NAMES,raw_gate_match=True,batch_thread_decisions_match=True,source_only=True,DURA_in_candidate=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 out=a.out/'source_predictions.npz'
 with out.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],prior_values=old,candidate_predictions=np.stack(outputs,axis=1))
 result=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),candidate_names=NAMES,sha256=sha(out),prior_sha256=pm['sha256'],ls_reference_sha256=rm['sha256'],model_sha256=hashes,common_raw_gate_match=True,source_L_s_replay_match=True,source_only=True,DURA_in_candidate=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,caveat='Source records previously explored; clean F pseudo targets not identity truth; no ground test or blind claim.')
 base.core.write(a.out/'prediction_complete.json',result);print('COMPLEMENT_FULL_SOURCE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

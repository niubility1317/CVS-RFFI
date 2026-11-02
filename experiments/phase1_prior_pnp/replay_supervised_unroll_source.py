"""Replay fixed source-supervised unroll models on a common source cohort."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pipeline_unroll as p
from observation_unroll import ObservationCorrector,restore_observation_conditioned
MODELS=[('label','20261102-phase1-label-unroll-s2026092701-r01','a0e150d97e32d4298aeb19e35d035fced11bb28d'),('multiview','20261102-phase1-multiview-unroll-s2026092701-r01','edfe97998e32ddd07059d8b52a259fab7580b2e9')]

def load_models(root,ids):
 models=[];hashes={}
 for name,run,commit in MODELS:
  folder=root/'runs'/run;result=p.core.read(folder/'result.json');path=folder/'checkpoint.pt';state=torch.load(path,map_location='cpu',weights_only=False)
  if result.get('status')!='SOURCE_TRAINING_COMPLETE' or result.get('commit')!=commit or state.get('commit')!=commit or result.get('DURA_loaded')!=False or result.get('F_frozen')!=True or any(result.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')) or len(state['fit_ids'])!=8192 or np.intersect1d(ids,state['fit_ids']).size:raise ValueError('Exact independent trained source models and disjoint validation required: '+name)
  net=ObservationCorrector();net.load_state_dict(state['model'],strict=True);net.eval().requires_grad_(False);models.append(net);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
 return models,hashes

@torch.no_grad()
def compute(x,y,recovery,teacher,models):
 blind,unused,stats,gate,eye=recovery;raw=teacher.classify_features(teacher.features(x)).argmax(1);active=gate(x).sigmoid()>.8;outputs=[];signal=x[active]
 if active.any():
  h,lv=blind(signal);A=p.action_matrix(h);states,lp=p.posterior(signal.reshape(len(signal),512),A,eye[None]*lv.exp()[:,None,None],*stats);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal)
 den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12)
 for net in models:
  wave=x.clone()
  if active.any():wave[active]=restore_observation_conditioned(net,signal,z,A,lv)
  pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,err=p.s.align(wave,y);outputs.extend((pred.float(),err/den))
 out=torch.stack(outputs,1)
 if not torch.isfinite(out).all():raise ValueError('Finite source output required')
 return raw,active,out

def batches(received,clean,ids,recovery,teacher,models,batch):
 out=[];raw=[];active=[]
 for start in range(0,len(ids),batch):
  ix=ids[start:start+batch];a,b,c=compute(p.core.tensor(received[ix],'cpu'),p.core.tensor(clean[ix],'cpu'),recovery,teacher,models);raw.append(a.cpu());active.append(b.to(torch.uint8).cpu());out.append(c.cpu())
 return np.from_dlpack(torch.cat(raw).contiguous()),np.from_dlpack(torch.cat(active).contiguous()),np.from_dlpack(torch.cat(out).contiguous())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source replay output required')
 a.out.mkdir(parents=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cache,clean,rx,training,pv,ph=p.s.source_contract(a.root);validation,heldout=p.s.new_splits(cache,rx,training,pv,ph);ids=np.r_[validation,heldout]
 if len(ids)!=11761:raise ValueError('Complete source cohort required')
 prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=p.core.read(prior/'prediction_complete.json');path=prior/'source_predictions.npz'
 if pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or hashlib.sha256(path.read_bytes()).hexdigest()!=pm.get('sha256') or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed prior source replay required')
 with np.load(path,allow_pickle=False) as z:prior_ids=z['indices'].copy();old=z['values'].copy();old_scene=z['scene'].copy()
 received,scene=p.s.epoch_data(cache,19)
 if not np.array_equal(ids,prior_ids) or not np.array_equal(scene[ids],old_scene) or old.shape!=(11761,15):raise ValueError('Exact prior source IDs and views required')
 models,hashes=load_models(a.root,ids);recovery=p.load_fixed(a.root,'cpu');teacher,_=p.core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);n=64 if a.smoke else len(ids);chunks=[];tick=time.monotonic()
 for start in range(0,n,256):
  ix=ids[start:min(n,start+256)];raw,active,values=batches(received,clean,ix,recovery,teacher,models,32);ref=old[start:start+len(ix)]
  if not np.array_equal(raw,ref[:,5]) or not np.array_equal(active,ref[:,3]>.8):raise ValueError('Source raw F and input gate must match prior replay')
  if a.smoke:
   raw8,active8,values8=batches(received,clean,ix,recovery,teacher,models,8)
   if not np.array_equal(raw,raw8) or not np.array_equal(active,active8) or not np.array_equal(values[:,[0,2]],values8[:,[0,2]]):raise ValueError('Source batch prediction equivalence required')
  chunks.append(values)
  if (start+len(ix))%512==0:print('SUPERVISED_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  p.core.write(a.out/'result.json',dict(status='PASS',count=n,candidate_names=[x[0] for x in MODELS],raw_gate_match=True,batch_decisions_match=True,source_only=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 result=np.concatenate(chunks);out=a.out/'source_predictions.npz'
 with out.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],prior_values=old,candidate_values=result)
 manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),candidate_names=[x[0] for x in MODELS],sha256=hashlib.sha256(out.read_bytes()).hexdigest(),prior_sha256=pm['sha256'],model_sha256=hashes,source_only=True,raw_gate_match=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0);p.core.write(a.out/'prediction_complete.json',manifest);print('SUPERVISED_SOURCE_REPLAY_COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__':main()

"""Replay fixed identity flow over all previously established source IDs."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from raw_conditioned_flow import RawConditionedFlow
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core
TRAIN='20261102-phase1-identity-flow-source-s2026092701-r01'
PRIOR='20261102-phase1-unroll-source-replay-s2026092701-r01'
MODEL_SHA='4abd48af41c4552ff69f791d8582b74e692f2f6f5c8c4afb6bee9ab9544ffb73'
PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def batch(x,y,net,teacher):
 raw=teacher.classify_features(teacher.features(x)).argmax(1);wave=net.restore(x,steps=4);pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,error=s.align(wave,y);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);values=torch.stack((pred.float(),error/den),1)
 if not torch.isfinite(values).all():raise ValueError('Finite fixed predictions required')
 return raw,values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source replay output required')
 torch.set_num_threads(2);source=a.root/'runs'/TRAIN;m=core.read(source/'result.json');ck=source/'checkpoint.pt';initial=source/'source_predictions.npz';old=a.root/'runs'/PRIOR;pm=core.read(old/'prediction_complete.json');prior_path=old/'source_predictions.npz'
 if m.get('status')!='SOURCE_IDENTITY_FLOW_TRAINING_COMPLETE' or m.get('commit')!='4fa488be513e5a116ee1cb88411af2a82e17ccda' or m.get('evaluation_count')!=672 or m.get('checkpoint_sha256')!=MODEL_SHA or digest(ck)!=MODEL_SHA or m.get('prediction_sha256')!=digest(initial) or any(m.get(k)!=0 for k in ('L_s_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed identity flow required')
 if pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('sha256')!=PRIOR_SHA or digest(prior_path)!=PRIOR_SHA or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact prior source replay required')
 with np.load(initial,allow_pickle=False) as z:initial_ids=z['indices'].copy();initial_values=z['values'].copy();initial_group=z['group'].copy();initial_scene=z['scene'].copy()
 with np.load(prior_path,allow_pickle=False) as z:ids=z['indices'].copy();rx_saved=z['rx'].copy();scene_saved=z['scene'].copy();prior=z['values'].copy()
 if ids.shape!=(11761,) or len(np.unique(ids))!=11761 or prior.shape!=(11761,15) or len(np.unique(initial_ids))!=672 or initial_values.shape!=(672,6):raise ValueError('Complete unique source cohorts required')
 lookup={int(i):k for k,i in enumerate(ids)}
 if len(lookup)!=11761 or not all(int(i) in lookup for i in initial_ids):raise ValueError('Original source predictions must be in full cohort')
 pos=np.asarray([lookup[int(i)] for i in initial_ids]);saved_by_id={int(i):k for k,i in enumerate(initial_ids)}
 if not np.array_equal(scene_saved[pos],initial_scene) or not np.array_equal(prior[pos,0].astype(int),initial_values[:,1].astype(int)) or not np.array_equal(prior[pos,5].astype(int),initial_values[:,2].astype(int)) or not np.allclose(prior[pos,1],initial_values[:,0],rtol=1e-5,atol=1e-6):raise ValueError('Source F and original scene binding required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);state=torch.load(ck,map_location='cpu',weights_only=False)
 if state.get('commit')!=m['commit'] or len(state['fit_ids'])!=4096 or np.intersect1d(ids,state['fit_ids']).size or not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids]):raise ValueError('Disjoint fixed source cohort required')
 net=RawConditionedFlow();net.load_state_dict(state['model'],strict=True);net.eval().requires_grad_(False);teacher,_=core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False)
 if a.smoke:
  chosen=[]
  for group in ('RX4_6','RX8'):
   for view in core.VIEWS:
    q=np.flatnonzero((initial_group==group)&(initial_scene==view));chosen.extend(q[:4].tolist())
  if len(chosen)!=56:raise ValueError('Four source records per stratum required')
  work_ids=initial_ids[np.asarray(chosen)];work_pos=pos[np.asarray(chosen)]
 else:work_ids=ids;work_pos=np.arange(len(ids))
 chunks=[];checked=0;tick=time.monotonic()
 for start in range(0,len(work_ids),8):
  ix=work_ids[start:start+8];posi=work_pos[start:start+len(ix)];raw,values=batch(core.tensor(received[ix],'cpu'),core.tensor(clean[ix],'cpu'),net,teacher);raw=np.from_dlpack(raw.contiguous());values=np.from_dlpack(values.contiguous())
  if not np.array_equal(raw,prior[posi,5].astype(np.int64)):raise ValueError('Raw F baseline mismatch')
  for j,i in enumerate(ix):
   if int(i) in saved_by_id:
    prev=initial_values[saved_by_id[int(i)]]
    if int(values[j,0])!=int(prev[4]) or not np.isclose(values[j,1],prev[5],rtol=1e-5,atol=1e-6):raise ValueError('Original fixed source prediction mismatch')
    checked+=1
  chunks.append(values)
  if (start+len(ix))%512==0:print('IDENTITY_FULL_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 if checked!=(56 if a.smoke else 672):raise ValueError('Complete original source prediction replay required')
 a.out.mkdir(parents=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',count=len(work_ids),original_records_matched=checked,source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 values=np.concatenate(chunks);out=a.out/'source_predictions.npz'
 with out.open('xb') as f:np.savez(f,indices=ids,rx=rx_saved,scene=scene_saved,prior_values=prior,flow_values=values)
 manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=digest(out),prior_sha256=PRIOR_SHA,model_sha256=MODEL_SHA,initial_sha256=digest(initial),original_records_matched=checked,source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,caveat='Previously explored source IDs; no ground test.')
 core.write(a.out/'prediction_complete.json',manifest);print('IDENTITY_FULL_SOURCE_COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__':main()

"""Source-only three-split coefficient-agreement guard diagnosis."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from channel_agreement import agreement_metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;state=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);prior=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if state.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or state.get('epoch')!=10 or prior.get('source_ids')!=8192:raise ValueError('Exact independent source models required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);stats=(prior['means'].to(a.device),prior['covariances'].to(a.device),prior['priors'].to(a.device));eye=torch.eye(512,device=a.device)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);origin=a.root/'runs/20261102-phase1-independent-prior-pnp-step-diagnostic-s2026092701-r02';manifest=core.read(origin/'prediction_complete.json');fixed=origin/'source_predictions.npz'
 if manifest.get('sha256')!=hashlib.sha256(fixed.read_bytes()).hexdigest() or any(manifest.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact source cohort provenance required')
 with np.load(fixed,allow_pickle=False) as z:ids=z['indices'];split=z['part'];scene_orig=z['scene'];rx_orig=z['rx']
 if len(ids)!=2048 or not np.array_equal(rx[ids],rx_orig) or not np.array_equal(scene[ids],scene_orig) or not np.all(np.isin(ids[split==1],validation)) or not np.all(np.isin(ids[split==2],heldout)) or np.any(scene[ids]=='clean'):raise ValueError('Fixed disjoint impaired source records required')
 if a.smoke:
  chosen=np.r_[np.flatnonzero(split==1)[:16],np.flatnonzero(split==2)[:16]];ids=ids[chosen];split=split[chosen]
 outputs=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   ix=ids[start:start+8];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*stats);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);Ay=torch.bmm(A,y.reshape(len(y),512,1)).reshape_as(y);Az=torch.bmm(A,z.reshape(len(z),512,1)).reshape_as(z);as_complex=lambda q:torch.complex(q[:,0,:],q[:,1,:]);metric=agreement_metrics(as_complex(x),as_complex(Ay),as_complex(Az));outputs.append(metric.cpu())
   if (start+len(ix))%256==0:print('CHANNEL_AGREEMENT_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.from_dlpack(torch.cat(outputs).contiguous())
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',records=len(ids),source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('CHANNEL_AGREEMENT_SMOKE PASS',flush=True);return
 saved=a.out/'source_channel_agreement.npz'
 with saved.open('xb') as f:np.savez(f,indices=ids,split=split,rx=rx[ids],scene=scene[ids],values=values)
 core.write(a.out/'prediction_complete.json',dict(status='SOURCE_DIAGNOSTIC_FIXED',commit=a.commit,count=len(ids),sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,notes='Three disjoint interleaved time folds: blind fits use received and estimated parent, gate uses agreement of two fits, source paired clean IQ evaluates the third fold only. Parent uses all times, and the folds are correlated.'))
 print('CHANNEL_AGREEMENT_FIXED',len(ids),flush=True)
if __name__=='__main__':main()

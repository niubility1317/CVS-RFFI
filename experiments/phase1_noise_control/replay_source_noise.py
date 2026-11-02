"""Replay the fixed source blind noise estimator without identity labels."""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from noise_proxy import relative_logvar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,SOURCE

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def run(model,received,ids,batch):
 out=[]
 for start in range(0,len(ids),batch):
  x=core.tensor(received[ids[start:start+batch]],'cpu');h,lv=model(x);rel=relative_logvar(x,lv)
  if not torch.isfinite(h).all() or not torch.isfinite(rel).all():raise ValueError('Finite blind estimates required')
  out.append(torch.stack((lv,rel),1).cpu())
 return np.from_dlpack(torch.cat(out).contiguous())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source noise estimate required')
 torch.set_num_threads(1);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=core.read(prior/'prediction_complete.json');pp=prior/'source_predictions.npz';expected='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
 if pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('sha256')!=expected or sha(pp)!=expected or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed prior source predictions required')
 with np.load(pp,allow_pickle=False) as z:ids=z['indices'].copy();rx_prior=z['rx'].copy();scene_prior=z['scene'].copy()
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19)
 if ids.shape!=(11761,) or not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx[ids],rx_prior) or not np.array_equal(scene[ids],scene_prior):raise ValueError('Exact prior source cohort required')
 ck=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
 if state.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or state.get('epoch')!=10:raise ValueError('Fixed source blind estimator required')
 model=BlindActionNet();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False);n=128 if a.smoke else len(ids);chunks=[];tick=time.monotonic()
 for start in range(0,n,512):
  ix=ids[start:min(start+512,n)];values=run(model,received,ix,64)
  if a.smoke:
   alternate=run(model,received,ix,32)
   if not np.allclose(values,alternate,rtol=1e-5,atol=1e-6):raise ValueError('Stable source batch estimates required')
  chunks.append(values)
  if (start+len(ix))%2048==0:print('SOURCE_NOISE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.concatenate(chunks);a.out.mkdir(parents=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',count=n,batch_stable=True,DURA_loaded=False,F_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('SOURCE_NOISE_SMOKE_PASS',n,flush=True);return
 path=a.out/'source_noise.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],values=values)
 manifest=dict(status='SOURCE_NOISE_FIXED',commit=a.commit,count=n,sha256=sha(path),prior_sha256=expected,blind_checkpoint_sha256=sha(ck),source_only=True,DURA_loaded=False,F_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,metric='blind estimated log variance minus log observed IQ power; not measured SNR')
 core.write(a.out/'prediction_complete.json',manifest);print('SOURCE_NOISE_FIXED',sha(path),flush=True)
if __name__=='__main__':main()

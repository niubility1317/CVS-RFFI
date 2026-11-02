"""Replay one fixed spectral synthesis model on the complete source cohort."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from spectral_synthesis import SpectralSynthesis
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
import pipeline_unroll as p

@torch.no_grad()
def compute(x,y,recovery,teacher,net):
 blind,unused,stats,gate,eye=recovery;raw=teacher.classify_features(teacher.features(x)).argmax(1);active=gate(x).sigmoid()>.8;out=x.clone();parent=x.clone()
 if active.any():
  signal=x[active];h,lv=blind(signal);action=p.action_matrix(h);states,lp=p.posterior(signal.reshape(len(signal),512),action,eye[None]*lv.exp()[:,None,None],*stats);base=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);candidate=torch.cat((signal[:,None],states.reshape(-1,6,2,256)),1);parent[active]=base;out[active]=net(base,candidate,lv)
 den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);pred=teacher.classify_features(teacher.features(out)).argmax(1);pp=teacher.classify_features(teacher.features(parent)).argmax(1);_,err=p.s.align(out,y);_,pe=p.s.align(parent,y)
 return raw,active,pp,pe/den,torch.stack((pred.float(),err/den),1)

def batches(received,clean,ids,recovery,teacher,net,batch):
 chunks=[[] for _ in range(5)]
 for start in range(0,len(ids),batch):
  ix=ids[start:start+batch];vals=compute(p.core.tensor(received[ix],'cpu'),p.core.tensor(clean[ix],'cpu'),recovery,teacher,net)
  for j,t in enumerate(vals):chunks[j].append((t.to(torch.uint8) if j==1 else t).cpu())
 return [np.from_dlpack(torch.cat(v).contiguous()) for v in chunks]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source replay output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);cache,clean,rx,training,pv,ph=p.s.source_contract(a.root);validation,heldout=p.s.new_splits(cache,rx,training,pv,ph);ids=np.r_[validation,heldout]
 if len(ids)!=11761:raise ValueError('Complete source cohort required')
 source=a.root/'runs/20261102-phase1-spectral-synthesis-scaled-s2026092701-r01';sm=p.core.read(source/'prediction_complete.json');ck=source/'checkpoint.pt';state=torch.load(ck,map_location='cpu',weights_only=False);modelsha=hashlib.sha256(ck.read_bytes()).hexdigest()
 if sm.get('status')!='SOURCE_PREDICTIONS_FIXED' or sm.get('commit')!='d67348d3312a24917681c66d4e3074e3f3ff86c1' or state.get('commit')!=sm['commit'] or modelsha!='b859ba4f02932db384b4c8186a086c9c405ad606d952611b504b9596c2519f70' or len(state['fit_ids'])!=8192 or np.intersect1d(ids,state['fit_ids']).size:raise ValueError('Exact fixed source model and disjoint cohort required')
 prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=p.core.read(prior/'prediction_complete.json');path=prior/'source_predictions.npz'
 if pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or hashlib.sha256(path.read_bytes()).hexdigest()!=pm['sha256'] or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed prior source replay required')
 with np.load(path,allow_pickle=False) as z:prior_ids=z['indices'].copy();old=z['values'].copy();old_scene=z['scene'].copy()
 received,scene=p.s.epoch_data(cache,19)
 if not np.array_equal(ids,prior_ids) or not np.array_equal(scene[ids],old_scene) or old.shape!=(11761,15):raise ValueError('Exact same source IDs and views required')
 net=SpectralSynthesis();net.load_state_dict(state['model'],strict=True);net.eval().requires_grad_(False);recovery=p.load_fixed(a.root,'cpu');teacher,_=p.core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);n=64 if a.smoke else len(ids);chunks=[];tick=time.monotonic()
 for start in range(0,n,256):
  ix=ids[start:min(n,start+256)];raw,on,pp,pe,values=batches(received,clean,ix,recovery,teacher,net,8);ref=old[start:start+len(ix)]
  if not np.array_equal(raw,ref[:,5]) or not np.array_equal(on,ref[:,3]>.8) or not np.array_equal(pp,ref[:,7]) or not np.allclose(pe,ref[:,8],rtol=1e-5,atol=1e-7):raise ValueError('Raw F, gate and physical posterior must match previous source replay')
  if a.smoke:
   b=batches(received,clean,ix,recovery,teacher,net,32)
   if not np.array_equal(raw,b[0]) or not np.array_equal(on,b[1]) or not np.array_equal(pp,b[2]) or not np.array_equal(values[:,0],b[4][:,0]) or not np.allclose(values[:,1],b[4][:,1],rtol=1e-5,atol=1e-6):raise ValueError('Source batch prediction equivalence required')
  if not np.isfinite(values).all():raise ValueError('Finite fixed source output required')
  chunks.append(values)
  if (start+len(ix))%512==0:print('SPECTRAL_FULL_SOURCE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  p.core.write(a.out/'result.json',dict(status='PASS',count=n,raw_gate_physical_match=True,batch_decisions_match=True,source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 values=np.concatenate(chunks);out=a.out/'source_predictions.npz'
 with out.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],scene=scene[ids],prior_values=old,spectral_values=values)
 manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),prior_sha256=pm['sha256'],model_sha256=modelsha,source_only=True,raw_gate_physical_match=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,caveat='Source IDs were previously explored; no ground test.')
 p.core.write(a.out/'prediction_complete.json',manifest);print('SPECTRAL_FULL_SOURCE_COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__':main()

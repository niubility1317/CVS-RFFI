"""Stress fixed source physical recovery using guarded paired receiver filters."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from fit_sampling import select_fit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
import pipeline_unroll as p
from receiver_augmentation import receiver_filter
from train_multiview_unroll import allowed_fit_labels

@torch.no_grad()
def recover(x,recovery,teacher):
 blind,unused,stats,gate,eye=recovery;raw=teacher.classify_features(teacher.features(x)).argmax(1);active=gate(x).sigmoid()>.8;wave=x.clone()
 if active.any():
  z=x[active];h,lv=blind(z);A=p.action_matrix(h);states,lp=p.posterior(z.reshape(len(z),512),A,eye[None]*lv.exp()[:,None,None],*stats);wave[active]=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(z)
 pred=teacher.classify_features(teacher.features(wave)).argmax(1);return raw,pred,active,wave

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source stress diagnostic required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);cache,clean,rx,training,pv,ph=p.s.source_contract(a.root);validation,heldout=p.s.new_splits(cache,rx,training,pv,ph);prior_ids=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));fit=select_fit(training,prior_ids,8192);roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=p.core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);labels=allowed_fit_labels(a.root,cache,clean,roles,contract,fit,validation,heldout);known=fit[labels[fit]>=0]
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size or np.any(roles[known]!='L_s'):raise ValueError('Fit-only legal source roles required')
 correct=[]
 with torch.no_grad():
  for start in range(0,len(known),32):
   ix=known[start:start+32];v=teacher.classify_features(teacher.features(p.core.tensor(clean[ix],'cpu'))).argmax(1);correct.extend(np.from_dlpack(v.cpu().contiguous()).tolist())
 eligible=known[np.asarray(correct)==labels[known]];fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Exact source physical IDs required')
 waves=np.empty((len(eligible),2,256),dtype=np.float32);epochs=np.full(len(eligible),-1,dtype=np.int16);needed=2 if a.smoke else 48
 for epoch in (21,22,23,11,12,13,14,15,16,17,18,19,20):
  source=fresh if epoch>=21 else cache;info=p.core.read(source/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source view required')
  received,scene=p.s.epoch_data(source,epoch);found=np.flatnonzero((epochs<0)&(scene[eligible]=='practical_low_urban'));waves[found]=received[eligible[found]];epochs[found]=epoch
  if all(np.count_nonzero((epochs>=0)&(labels[eligible]==c))>=needed for c in range(6)):break
 if any(np.count_nonzero((epochs>=0)&(labels[eligible]==c))<needed for c in range(6)):raise ValueError('Insufficient class-balanced eligible low-city source views')
 rng=np.random.default_rng(2026138061);pick=np.concatenate([np.sort(rng.choice(np.flatnonzero((epochs>=0)&(labels[eligible]==c)),size=needed,replace=False)) for c in range(6)]);ids=eligible[pick];target=labels[ids];x=waves[pick];epoch_chosen=epochs[pick];coeff=rng.uniform(-1.,1.,size=(len(ids),4)).astype(np.float32);recovery=p.load_fixed(a.root,'cpu');bounds=[0.,.15,.3,.5];pieces=[];tick=time.monotonic()
 for bound in bounds:
  chunks=[]
  for start in range(0,len(ids),8):
   ix=slice(start,min(start+8,len(ids)));xx=p.core.tensor(x[ix],'cpu');yy=p.core.tensor(clean[ids[ix]],'cpu');co=p.core.tensor(coeff[ix]*bound,'cpu')
   with torch.no_grad():
    xa=receiver_filter(xx,co);ya=receiver_filter(yy,co);clean_after=teacher.classify_features(teacher.features(ya)).argmax(1);label=torch.tensor(target[ix].tolist(),dtype=torch.long);apply=(clean_after==label) if bound>0 else torch.zeros_like(label,dtype=torch.bool);xx=torch.where(apply[:,None,None],xa,xx);yy=torch.where(apply[:,None,None],ya,yy);raw,pred,active,wave=recover(xx,recovery,teacher);den=yy[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,err=p.s.align(wave,yy);out=torch.stack((raw.float(),pred.float(),clean_after.float(),apply.float(),active.float(),err/den),1)
   chunks.append(out.cpu())
  part=np.from_dlpack(torch.cat(chunks).contiguous());
  if not np.isfinite(part).all():raise ValueError('Finite source stress predictions required')
  pieces.append(part);print('RECEIVER_STRESS_SOURCE_BOUND',bound,'samples',len(ids),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=np.stack(pieces,axis=1);path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx[ids],epochs=epoch_chosen,labels=target,coefficients=coeff,values=values)
 results={}
 for j,bound in enumerate(bounds):
  results[str(bound)]={}
  for label in range(6):
   sel=target==label;v=values[sel,j];raw=v[:,0]==label;physical=v[:,1]==label;results[str(bound)][str(label)]=dict(count=int(sel.sum()),applied=int(np.count_nonzero(v[:,3])),active=int(np.count_nonzero(v[:,4])),raw_correct=int(raw.sum()),physical_correct=int(physical.sum()),repairs=int(np.count_nonzero(~raw&physical)),damages=int(np.count_nonzero(raw&~physical)),aligned_iq_error_mean=float(v[:,5].mean()))
 manifest=dict(status='SOURCE_STRESS_COMPLETE' if not a.smoke else 'PASS',commit=a.commit,count=len(ids),source_fit_ids=8192,L_s_metadata_labels_read=6300,L_s_scored=len(ids),bounds=bounds,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results,clean_guard='proposed clean F class equals already-correct true L_s class',F_frozen=True,blind_frozen=True,DURA_loaded=False,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Synthetic source-fit stress only; guard is F-based and does not prove fingerprint preservation; not ground or independent validation.')
 p.core.write(a.out/'result.json',manifest);print('RECEIVER_STRESS_COMPLETE',json.dumps({'status':manifest['status'],'count':len(ids),'class5':{k:v['5'] for k,v in results.items()}}),flush=True)
if __name__=='__main__':main()

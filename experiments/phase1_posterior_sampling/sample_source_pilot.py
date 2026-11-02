"""Source-only posterior draws and fixed classifier-based waveform selection."""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from posterior_sampling import conditional_gaussian_draws
from posterior_selection import choose_consensus,select_waveforms
from sampling_noise import source_noise,tensor_array
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def batch(x,y,indices,blind,stats,factors,gate,teacher,check_reference=False):
 means,cov,priors=stats;n=len(x);raw_logits=teacher.classify_features(teacher.features(x));raw_prob=F.softmax(raw_logits,dim=1);raw=raw_prob.argmax(1);active=gate(x).sigmoid()>.8;parent=x.clone();waves=x[:,None].expand(-1,6,-1,-1).clone();logweights=x.new_zeros((n,6));
 if active.any():
  signal=x[active];h,lv=blind(signal);A=action_matrix(h);var=lv.exp();selected=np.asarray(indices)[np.from_dlpack(active.to(torch.uint8).cpu().contiguous()).astype(bool)];z,e=source_noise(selected,6,512);draw,states,lp=conditional_gaussian_draws(signal.reshape(len(signal),512),A,var,means,factors,priors,z,e,covariance=cov)
  if check_reference:
   reference,rlp=posterior(signal.reshape(len(signal),512),A,torch.eye(512)[None]*var[:,None,None],*stats)
   if not torch.allclose(states,reference,rtol=1e-4,atol=1e-5) or not torch.allclose(lp.exp(),rlp.exp(),rtol=1e-3,atol=1e-4):raise ValueError('Conditional posterior means and mixture weights must match fixed posterior')
  parent[active]=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);waves[active]=draw.reshape(-1,6,2,256);logweights[active]=lp
 logits=teacher.classify_features(teacher.features(waves.reshape(-1,2,256))).reshape(n,6,6);prob=F.softmax(logits,dim=-1);sp=prob.argmax(-1);consensus,preferred=choose_consensus(prob,logweights);confidence=prob.max(-1).values.argmax(-1);cwave=select_waveforms(waves,consensus);fwave=select_waveforms(waves,confidence);cp=teacher.classify_features(teacher.features(cwave)).argmax(1);fp=teacher.classify_features(teacher.features(fwave)).argmax(1);pp=teacher.classify_features(teacher.features(parent)).argmax(1)
 if not torch.equal(cp,sp.gather(1,consensus[:,None]).squeeze(1)) or not torch.equal(fp,sp.gather(1,confidence[:,None]).squeeze(1)):raise ValueError('Selected IQ must reproduce saved candidate decisions')
 den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);errors=[]
 for wave in (x,parent,cwave,fwave):
  _,err=s.align(wave,y);errors.append(err/den)
 return (torch.stack((raw,pp,cp,fp),1),torch.stack(errors,1),prob,logweights,torch.stack((consensus,confidence),1),active,torch.stack((cwave,fwave),1))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source pilot required')
 torch.set_num_threads(2);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';pm=core.read(prior/'prediction_complete.json');ppath=prior/'source_predictions.npz';expected='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
 if pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('sha256')!=expected or sha(ppath)!=expected or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source prior required')
 ref=a.root/'runs/20261102-phase1-spectral-synthesis-scaled-s2026092701-r01';rm=core.read(ref/'prediction_complete.json');rpath=ref/'source_predictions.npz'
 if rm.get('count')!=672 or rm.get('prediction_sha256')!='f9ac6f7dbfb1553c368ac63c1d38c965d179a8490393f3bd5c22da8b7c6a917d' or sha(rpath)!=rm['prediction_sha256']:raise ValueError('Fixed 672 source IDs required')
 with np.load(rpath,allow_pickle=False) as z:ids=z['indices'].copy();rx=z['rx'].copy();scene=z['scene'].copy();split=z['split'].copy()
 with np.load(ppath,allow_pickle=False) as z:
  order=np.argsort(z['indices']);keys=z['indices'][order];pos=np.searchsorted(keys,ids)
  if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],ids) or not np.array_equal(z['rx'][order[pos]],rx) or not np.array_equal(z['scene'][order[pos]],scene):raise ValueError('Source physical IDs must match')
  old=z['values'][order[pos]].copy()
 cache,clean,allrx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,allrx,training,pv,ph);received,allscene=s.epoch_data(cache,19)
 if len(set(ids.tolist()))!=672 or np.intersect1d(ids,training).size or not np.array_equal(allrx[ids],rx) or not np.array_equal(allscene[ids],scene):raise ValueError('Unique fixed source evaluation records required')
 b=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if b.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or b.get('epoch')!=10 or p.get('source_ids')!=8192:raise ValueError('Fixed source blind estimator and prior required')
 blind=BlindActionNet();blind.load_state_dict(b['model'],strict=True);blind.eval().requires_grad_(False);stats=(p['means'],p['covariances'],p['priors']);cov=(stats[1]+stats[1].transpose(-1,-2))*.5;factors,info=torch.linalg.cholesky_ex(cov)
 if (info!=0).any() or float((factors@factors.transpose(-1,-2)-cov).norm()/cov.norm())>1e-5:raise ValueError('Accurate source covariance factors required')
 fixed=torch.load(a.root/'runs'/s.GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
 if fixed.get('commit')!=s.GATE_COMMIT or fixed.get('selected',{}).get('kind')!='gated_direct_iq':raise ValueError('Fixed source gate required')
 gate=s.BlindGate();gate.load_state_dict(fixed['gate'],strict=True);gate.eval().requires_grad_(False);teacher,_=core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False)
 if a.smoke:
  chosen=[]
  for group in ('RX4_6','RX8'):
   for view in core.VIEWS:
    ix=np.flatnonzero((split==group)&(scene==view))
    if len(ix)<2:raise ValueError('Two records per source stratum required')
    chosen.extend(ix[:2].tolist())
  select=np.array(chosen,dtype=np.int64)
 else:select=np.arange(len(ids))
 chunks=[[] for _ in range(7)];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(select),8):
   pos=select[start:start+8];idx=ids[pos];x=core.tensor(received[idx],'cpu');y=core.tensor(clean[idx],'cpu');vals=batch(x,y,idx,blind,stats,factors,gate,teacher,a.smoke);raw,physical,cons,conf=vals[0].unbind(1)
   if not np.array_equal(tensor_array(raw),old[pos,5]) or not np.array_equal(tensor_array(physical),old[pos,7]) or not np.array_equal(tensor_array(vals[5]),old[pos,3]>.8):raise ValueError('Source raw, physical and fixed gate must match')
   for j,v in enumerate(vals):chunks[j].append(v.cpu())
   if (start+len(pos))%128==0:print('POSTERIOR_SAMPLE_SOURCE_PROGRESS',start+len(pos),'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  a.out.mkdir(parents=True);core.write(a.out/'result.json',dict(status='PASS',records=len(select),source_only=True,posterior_mean_and_weights_match=True,selected_waveform_predictions_match=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('POSTERIOR_SAMPLE_SOURCE_SMOKE_PASS',len(select),flush=True);return
 arr=[tensor_array(torch.cat(v)) for v in chunks];path=a.out/'source_predictions.npz';a.out.mkdir(parents=True)
 with path.open('xb') as f:np.savez(f,indices=ids,rx=rx,scene=scene,split=split,prior_values=old,predictions=arr[0],aligned_errors=arr[1],probabilities=arr[2],logweights=arr[3],choices=arr[4],active=arr[5],selected_waveforms=arr[6])
 m=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=sha(path),prior_sha256=expected,reference_sha256=rm['prediction_sha256'],blind_checkpoint_sha256=sha(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt'),source_prior_sha256=sha(a.root/'runs'/PRIOR_RUN/'source_prior.pt'),sampling_seed_base=2026165054,draws_per_component=1,selection=['weighted_consensus','max_confidence'],source_only=True,DURA_loaded=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0,caveat='Approximate posterior under fixed blind channel and Gaussian source prior; source records previously explored; no ground claim.')
 core.write(a.out/'prediction_complete.json',m);print('POSTERIOR_SAMPLE_SOURCE_FIXED',sha(path),flush=True)
if __name__=='__main__':main()

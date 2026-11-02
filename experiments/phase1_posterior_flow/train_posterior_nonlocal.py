"""Source-only nonlocal synthesis trained from physical posterior hypotheses."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from posterior_nonlocal_synthesis import PosteriorNonlocalSynthesis
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_spectral_synthesis'))
from fit_sampling import select_fit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN
from train_multiview_unroll import allowed_fit_labels

@torch.no_grad()
def hypotheses(x,blind,prior,eye):
 h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*prior);base=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);candidates=torch.cat((x[:,None],states.reshape(-1,6,2,256)),1);return base,candidates,lv

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--fit-size',type=int,choices=(1024,8192),default=1024);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(2026135054);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);prior_ids=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));fit=select_fit(training,prior_ids,a.fit_size);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Independent source fit IDs required')
 teacher,contract=core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);labels=allowed_fit_labels(a.root,cache,clean,roles,contract,fit,validation,heldout)
 b=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if b.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or b.get('epoch')!=10 or p.get('source_ids')!=8192:raise ValueError('Exact source physical models required')
 blind=BlindActionNet();blind.load_state_dict(b['model']);blind.eval().requires_grad_(False);prior=(p['means'],p['covariances'],p['priors']);eye=torch.eye(512);net=PosteriorNonlocalSynthesis();fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Matching source physical IDs required')
 selected=fit if not a.smoke else np.sort(np.r_[fit[roles[fit]=='L_s'][:16],fit[roles[fit]=='U_s'][:16]])
 if a.smoke and len(selected)!=32:raise ValueError('Mixed source roles required')
 chunks=[];scene_counts={};tick=time.monotonic()
 for epoch in (21,22,23):
  info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
  received,scene=s.epoch_data(fresh,epoch);ids=selected[scene[selected]!='clean']
  for v in np.unique(scene[ids]):scene_counts[str(v)]=scene_counts.get(str(v),0)+int(np.count_nonzero(scene[ids]==v))
  for start in range(0,len(ids),8):
   ix=ids[start:start+8];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');base,candidate,lv=hypotheses(x,blind,prior,eye)
   with torch.no_grad():yf=teacher.features(y);soft=F.softmax(teacher.classify_features(yf)/2,dim=1)
   chunks.append((base,candidate,lv,y,yf,soft,torch.tensor(labels[ix].tolist()),torch.tensor(rx[ix].tolist())))
  print('POSTERIOR_NONLOCAL_SOURCE_PREP',epoch,'samples',sum(len(c[0]) for c in chunks),'seconds',round(time.monotonic()-tick,1),flush=True)
 if not chunks:raise ValueError('Impaired source examples required')
 base,candidate,lv,y,yf,soft,lab,groups=(torch.cat(q) for q in zip(*chunks));del chunks;fit_exposures=len(base);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);rng=torch.Generator().manual_seed(2026135056);losses=[]
 for epoch in range(1,(2 if a.smoke else 6)):
  order=torch.randperm(len(base),generator=rng);total=0.
  for start in range(0,len(order),16):
   ix=order[start:start+16];out=net(base[ix],candidate[ix],lv[ix]);feat=teacher.features(out);logits=teacher.classify_features(feat);den=y[ix].flatten(1).square().sum(1).clamp_min(1e-8);wave=(out-y[ix]).flatten(1).square().sum(1)/den;semantic=1-F.cosine_similarity(feat,yf[ix],dim=1,eps=1e-8);kl=F.kl_div(F.log_softmax(logits/2,dim=1),soft[ix],reduction='none').sum(1)*4;per=wave+.2*semantic+.1*kl;loss=torch.stack([per[groups[ix]==g].mean() for g in groups[ix].unique()]).mean();known=lab[ix]>=0
   if known.any():loss=loss+.2*F.cross_entropy(logits[known],lab[ix][known].long())
   opt.zero_grad(set_to_none=True);loss.backward();grad=net.coefficients.weight.grad
   if grad is None or not torch.isfinite(grad).all() or (epoch==1 and start==0 and grad.abs().sum()==0) or not torch.isfinite(loss):raise ValueError('Finite nonzero synthesis gradient required')
   if any(p.requires_grad or p.grad is not None for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Frozen teacher and blind estimator required')
   torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
  losses.append(total/len(base));print('POSTERIOR_NONLOCAL_EPOCH',epoch,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 net.eval().requires_grad_(False);received,scene=s.epoch_data(cache,19);views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];rng=np.random.default_rng(2026135057);groups_eval=[('RX4_6',validation),('RX8',heldout)];ids=[];splits=[]
 for group,rows in groups_eval:
  for view in views:
   eligible=rows[scene[rows]==view];n=2 if a.smoke else 48
   if len(eligible)<n:raise ValueError('Source evaluation stratum too small')
   chosen=np.sort(rng.choice(eligible,size=n,replace=False));ids.extend(chosen.tolist());splits.extend([group]*n)
 ids=np.array(ids,dtype=np.int64)
 if len(np.unique(ids))!=len(ids) or np.intersect1d(ids,fit).size:raise ValueError('Unique disjoint source evaluation required')
 values=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   ix=ids[start:start+8];x=core.tensor(received[ix],'cpu');target=core.tensor(clean[ix],'cpu');base,candidates,lv=hypotheses(x,blind,prior,eye);restored=net(base,candidates,lv);clean_logits=teacher.classify_features(teacher.features(target));conf,pseudo=F.softmax(clean_logits,1).max(1);den=target[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);row=[conf,pseudo.float()]
   for sig in (x,base,restored):
    pred=teacher.classify_features(teacher.features(sig)).argmax(1);_,err=s.align(sig,target);row.extend((pred.float(),err/den))
   values.append(torch.stack(row,1).cpu())
 arr=np.from_dlpack(torch.cat(values).contiguous())
 if arr.shape!=(len(ids),8) or not np.isfinite(arr).all():raise ValueError('Finite fixed source predictions required')
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',source_fit_ids=len(selected),fit_exposures=fit_exposures,source_eval_ids=len(ids),F_frozen=True,blind_frozen=True,DURA_loaded=False,L_s_metadata_labels_read=6300,L_s_fit_labels_used=int(np.count_nonzero(labels[selected]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0));return
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=net.state_dict(),fit_ids=fit,losses=losses),f)
 path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,split=np.asarray(splits),rx=rx[ids],scene=scene[ids],values=arr)
 sha=hashlib.sha256(path.read_bytes()).hexdigest();model_sha=hashlib.sha256((a.out/'checkpoint.pt').read_bytes()).hexdigest();core.write(a.out/'prediction_complete.json',dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),prediction_sha256=sha,checkpoint_sha256=model_sha,source_fit_ids=len(fit),fit_exposures=fit_exposures,scene_counts=scene_counts,losses=losses,F_frozen=True,blind_frozen=True,DURA_loaded=False,L_s_metadata_labels_read=6300,L_s_fit_labels_used=int(np.count_nonzero(labels[fit]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates_after_prediction=0,caveat='Source F clean pseudo targets are not identity truth; source IDs previously explored.'))
 print('POSTERIOR_NONLOCAL_SOURCE_FIXED',sha,'count',len(ids),flush=True)
if __name__=='__main__':main()

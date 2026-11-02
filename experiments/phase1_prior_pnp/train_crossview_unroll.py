"""Train independent recovery with paired-view class-aware feature contrast."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from observation_unroll import ObservationCorrector,restore_observation_conditioned
from crossview_contrast import paired_class_contrast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def allowed_fit_labels(root,cache,clean,roles,contract,fit,validation,heldout):
 complete=core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json');source=Path(complete['config']['source_cache'])
 with np.load(source/'L_s_metadata.npz',allow_pickle=False) as data:ids=data['ids'].copy();truth=data['y'].copy()
 physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);positions=np.flatnonzero(roles=='L_s');physical_l=physical[positions]
 if len(ids)!=6300 or truth.shape!=(6300,) or np.any((truth<0)|(truth>=6)) or list(ids)!=contract['role_ids']['L_s'] or len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(physical_l.tolist()):raise ValueError('Exact L_s IDs and labels required')
 lookup={str(v):int(i) for i,v in zip(positions,physical_l)};ordered=np.array([lookup[str(v)] for v in ids]);path=(source/'L_s_x.npy').resolve();index=path.parent/'L_s_index.npz'
 if not index.is_file():index=path.parent/'L_s_metadata.npz'
 with np.load(index,allow_pickle=False) as data:stored=data['ids'].copy()
 if len(set(stored.tolist()))!=len(ids) or set(stored.tolist())!=set(ids.tolist()):raise ValueError('Exact L_s IQ binding required')
 row={str(v):i for i,v in enumerate(stored)};x=np.load(path,mmap_mode='r',allow_pickle=False);source_order=np.array([row[str(v)] for v in ids]);
 if x.shape!=(6300,2,256) or not np.allclose(x[source_order],clean[ordered],rtol=1e-5,atol=1e-6):raise ValueError('L_s source IQ binding differs')
 labels=np.full(len(roles),-1,dtype=np.int64);allowed=np.isin(ordered,fit);labels[ordered[allowed]]=truth[allowed]
 if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1) or np.any(labels[fit[roles[fit]=='L_s']]<0):raise ValueError('Only fit L_s labels may affect training')
 return labels

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026127054)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);prior_ids=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);fit=np.sort(np.random.default_rng(2026125055).choice(pool,size=8192,replace=False))
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Independent source fit and heldout IDs required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);labels=allowed_fit_labels(a.root,cache,clean,roles,contract,fit,validation,heldout);teacher.eval().requires_grad_(False);model=ObservationCorrector().to(a.device)
 blindstate=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p0=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if blindstate.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or blindstate.get('epoch')!=10 or p0.get('source_ids')!=8192:raise ValueError('Exact independent source models required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);stats=(p0['means'].to(a.device),p0['covariances'].to(a.device),p0['priors'].to(a.device));eye=torch.eye(512,device=a.device)
 selected=fit if not a.smoke else np.sort(np.r_[fit[roles[fit]=='L_s'][:32],fit[roles[fit]=='U_s'][:32]])
 if a.smoke and len(selected)!=64:raise ValueError('Mixed source roles required')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Matching source IDs required')
 received_iq=np.empty((3,len(fit),2,256),dtype=np.float32);assignment=np.full((3,len(fit)),-1,dtype=np.int16);wanted=np.isin(fit,selected);tick=time.monotonic();scene_counts={}
 for epoch in (21,22,23,11,12,13,14,15,16,17,18,19,20):
  source=fresh if epoch>=21 else cache;info=core.read(source/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
  received,scene=s.epoch_data(source,epoch);positions=np.flatnonzero(wanted&np.any(assignment<0,axis=0)&(scene[fit]!='clean'));slots=np.argmin(assignment[:,positions]>=0,axis=0);received_iq[slots,positions]=received[fit[positions]];assignment[slots,positions]=epoch
  for value in np.unique(scene[fit[positions]]):scene_counts[str(value)]=scene_counts.get(str(value),0)+int(np.count_nonzero(scene[fit[positions]]==value))
  if np.all(assignment[:,wanted]>0):break
 if np.any(assignment[:,wanted]<0) or not np.isfinite(received_iq[:,wanted]).all() or np.any(np.diff(np.sort(assignment[:,wanted],axis=0),axis=0)==0):raise ValueError('Three distinct finite impaired source views required')
 lookup=np.full(len(rx),-1,dtype=np.int64);lookup[fit]=np.arange(len(fit));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(2026127055);counts={'exposures':0,'L_s':0,'U_s':0};losses=[]
 for epoch in range(1,(2 if a.smoke else 9)):
  order=selected if a.smoke else rng.permutation(fit);total=0.
  for start in range(0,len(order),8):
   ix=order[start:start+8];slots=(epoch-1+lookup[ix])%3;x1=core.tensor(received_iq[slots,lookup[ix]],a.device);x2=core.tensor(received_iq[(slots+1)%3,lookup[ix]],a.device);x=torch.cat((x1,x2),0);clean_iq=core.tensor(clean[ix],a.device);y=torch.cat((clean_iq,clean_iq),0)
   with torch.no_grad():
    h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*stats);initial=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);yf=teacher.features(y);clean_logits=teacher.classify_features(yf);soft=F.softmax(clean_logits/2,1);confidence,pseudo=F.softmax(clean_logits[:len(ix)],1).max(1)
   out=restore_observation_conditioned(model,x,initial,A,lv);feature=teacher.features(out);logits=teacher.classify_features(feature);den=y.flatten(1).square().sum(1).clamp_min(1e-8);wave=(out-y).flatten(1).square().sum(1)/den;semantic=1-F.cosine_similarity(feature,yf,dim=1,eps=1e-8);kl=F.kl_div(F.log_softmax(logits/2,1),soft,reduction='none').sum(1)*4;per=wave+.2*semantic+.1*kl;groups=np.tile(rx[ix],2);loss=torch.stack([per[torch.tensor((groups==g).tolist(),device=a.device)].mean() for g in np.unique(groups)]).mean()
   target=torch.tensor(labels[ix].tolist(),dtype=torch.long,device=a.device);repeated_target=target.repeat(2);known=repeated_target>=0
   if known.any():loss=loss+.1*F.cross_entropy(logits[known],repeated_target[known])
   class_target=torch.where(target>=0,target,torch.where(confidence>=.95,pseudo,torch.full_like(pseudo,-1)));contrast=paired_class_contrast(feature[:len(ix)],feature[len(ix):],class_target,temperature=.2);loss=loss+.05*contrast
   if a.smoke and epoch==1 and start==0:
    gradient=torch.autograd.grad(contrast,model.output.weight,retain_graph=True)[0]
    if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise ValueError('Finite nonzero cross-view gradient required')
   opt.zero_grad(set_to_none=True);loss.backward()
   if model.output.weight.grad is None or not torch.isfinite(model.output.weight.grad).all() or (epoch==1 and start==0 and model.output.weight.grad.abs().sum()==0):raise ValueError('Finite nonzero correction gradient required')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   if not torch.isfinite(loss) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()) or any(p.grad is not None or p.requires_grad for p in blind.parameters()):raise ValueError('Finite loss and frozen F and blind models required')
   opt.step();total+=float(loss.detach())*len(ix)
   if epoch==1:counts['exposures']+=len(ix);counts['L_s']+=int(np.count_nonzero(roles[ix]=='L_s'));counts['U_s']+=int(np.count_nonzero(roles[ix]=='U_s'))
  losses.append(total/len(order));print('CROSSVIEW_UNROLL_EPOCH',epoch,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',counts=counts,scene_counts=scene_counts,views_per_source_per_epoch=2,contrast_weight=.05,contrast_temperature=.2,F_frozen=True,blind_frozen=True,DURA_loaded=False,L_s_metadata_labels_read=6300,L_s_fit_labels_used=int(np.count_nonzero(labels[selected]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0));print('CROSSVIEW_UNROLL_SMOKE PASS',flush=True);return
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=model.state_dict(),counts=counts,losses=losses,fit_ids=fit,source_view_assignment=assignment),f)
 core.write(a.out/'result.json',dict(status='SOURCE_TRAINING_COMPLETE',commit=a.commit,counts=counts,losses=losses,scene_counts=scene_counts,assignment_counts={str(int(i)):int(np.count_nonzero(assignment==i)) for i in np.unique(assignment)},views_per_source_per_epoch=2,contrast_weight=.05,contrast_temperature=.2,F_frozen=True,blind_frozen=True,DURA_loaded=False,L_s_metadata_labels_read=6300,L_s_fit_labels_used=int(np.count_nonzero(labels[selected]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0,fit_ids=len(fit)))
 print('CROSSVIEW_UNROLL_TRAINING_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()

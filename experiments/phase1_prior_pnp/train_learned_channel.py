"""Train source-only blind channel correction with source paired IQ."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from learned_channel import ChannelRegressor,channel_features,oracle_target
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026128054)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);prior_ids=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);fit=np.sort(np.random.default_rng(2026125055).choice(pool,size=8192,replace=False))
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Independent source fit and heldout IDs required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);state=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p0=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if state.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or state.get('epoch')!=10 or p0.get('source_ids')!=8192:raise ValueError('Exact independent source models required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);stats=(p0['means'].to(a.device),p0['covariances'].to(a.device),p0['priors'].to(a.device));eye=torch.eye(512,device=a.device);model=ChannelRegressor().to(a.device)
 selected=fit if not a.smoke else np.sort(np.r_[fit[roles[fit]=='L_s'][:32],fit[roles[fit]=='U_s'][:32]])
 if a.smoke and len(selected)!=64:raise ValueError('Mixed source roles required')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Matching source IDs required')
 features=np.empty((len(fit),62),dtype=np.float32);targets=np.empty((len(fit),5),dtype=np.complex64);assignment=np.full(len(fit),-1,dtype=np.int16);wanted=np.isin(fit,selected);tick=time.monotonic();scene_counts={}
 for epoch in (21,22,23,11,12,13,14,15,16,17,18,19,20):
  source=fresh if epoch>=21 else cache;info=core.read(source/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
  received,scene=s.epoch_data(source,epoch);positions=np.flatnonzero(wanted&(assignment<0)&(scene[fit]!='clean'))
  with torch.no_grad():
   for start in range(0,len(positions),8):
    loc=positions[start:start+8];ix=fit[loc];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*stats);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);Ay=torch.bmm(A,y.reshape(len(y),512,1)).reshape_as(y);Az=torch.bmm(A,z.reshape(len(z),512,1)).reshape_as(z);f=channel_features(x,Az,lv);target=oracle_target(x,Ay);features[loc]=f.cpu().numpy();targets[loc]=target.cpu().numpy();assignment[loc]=epoch
  for value in np.unique(scene[fit[positions]]):scene_counts[str(value)]=scene_counts.get(str(value),0)+int(np.count_nonzero(scene[fit[positions]]==value))
  print('CHANNEL_LEARNING_SOURCE',epoch,'count',len(positions),'remaining',int(np.count_nonzero(wanted&(assignment<0))),'seconds',round(time.monotonic()-tick,1),flush=True)
  if np.all(assignment[wanted]>0):break
 if np.any(assignment[wanted]<0) or not np.isfinite(features[wanted]).all() or not np.isfinite(targets[wanted]).all():raise ValueError('Complete finite source targets required')
 x=core.tensor(features[wanted],a.device);target_array=np.array(targets[wanted],dtype=np.complex64,copy=True,order='C');y=torch.frombuffer(memoryview(target_array),dtype=torch.complex64).reshape(target_array.shape).to(a.device);groups=rx[fit[wanted]];model.mean.copy_(x.mean(0));model.std.copy_(x.std(0,unbiased=False).clamp_min(1e-4));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);rng=np.random.default_rng(2026128055);losses=[]
 for epoch in range(1,(2 if a.smoke else 31)):
  order=rng.permutation(len(x));total=0.
  for start in range(0,len(order),256):
   ix=order[start:start+256];prediction=model.forward_features(x[ix]);per=(prediction-y[ix]).abs().square().mean(1);g=groups[ix];loss=torch.stack([per[torch.tensor((g==r).tolist(),device=a.device)].mean() for r in np.unique(g)]).mean();opt.zero_grad(set_to_none=True);loss.backward()
   if model.net[-1].weight.grad is None or not torch.isfinite(model.net[-1].weight.grad).all() or (epoch==1 and start==0 and model.net[-1].weight.grad.abs().sum()==0):raise ValueError('Finite nonzero source regressor gradient required')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   if not torch.isfinite(loss) or any(p.grad is not None or p.requires_grad for p in blind.parameters()):raise ValueError('Finite loss and frozen action required')
   opt.step();total+=float(loss.detach())*len(ix)
  losses.append(total/len(order));print('CHANNEL_LEARNING_EPOCH',epoch,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 counts={'exposures':int(wanted.sum()),'L_s':int(np.count_nonzero(roles[fit[wanted]]=='L_s')),'U_s':int(np.count_nonzero(roles[fit[wanted]]=='U_s'))}
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',counts=counts,scene_counts=scene_counts,blind_frozen=True,DURA_loaded=False,F_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('CHANNEL_LEARNING_SMOKE PASS',flush=True);return
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=model.state_dict(),counts=counts,losses=losses,fit_ids=fit,source_view_assignment=assignment),f)
 core.write(a.out/'result.json',dict(status='SOURCE_TRAINING_COMPLETE',commit=a.commit,counts=counts,losses=losses,scene_counts=scene_counts,assignment_counts={str(int(i)):int(np.count_nonzero(assignment==i)) for i in np.unique(assignment)},blind_frozen=True,DURA_loaded=False,F_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,fit_ids=len(fit)))
 print('CHANNEL_LEARNING_TRAINING_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()

"""Train an independent waveform attention network with source-only frozen DURA teacher."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from attention_restorer import AttentionRestorer
from source_pilot import s,core,labels_source,source_prior,SCENES

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026123054)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph)
 rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);fit=np.sort(np.random.default_rng(2026123055).choice(pool,size=8192,replace=False))
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint fit and source validation IDs required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=truth[fit];del truth
 if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only source L_s training labels allowed')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Matching source physical IDs required')
 views=[]
 for epoch in ((21,) if a.smoke else (21,22,23)):
  info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Fixed source views required')
  received,scene=s.epoch_data(fresh,epoch);views.append((received,scene))
 pairs=np.concatenate([np.column_stack((np.full(len(fit),j,dtype=np.int64),fit)) for j in range(len(views))]);pairs=np.array([x for x in pairs if views[x[0]][1][x[1]]!='clean'])
 if a.smoke:
  known=pairs[roles[pairs[:,1]]=='L_s'][:32];unknown=pairs[roles[pairs[:,1]]=='U_s'][:32];pairs=np.concatenate((known,unknown))
  if len(known)!=32 or len(unknown)!=32:raise ValueError('Mixed role smoke required')
 initial_rng=torch.get_rng_state();dura=source_prior.frozen_dura(a.root,a.device);teacher_waves=[]
 with torch.no_grad():
  for start in range(0,len(pairs),32):
   part=pairs[start:start+32];xx=np.stack([views[j][0][i] for j,i in part]);teacher_waves.append(dura(core.tensor(xx,a.device)).detach().cpu())
 teacher_waves=torch.cat(teacher_waves);del dura;torch.set_rng_state(initial_rng)
 if len(teacher_waves)!=len(pairs) or not torch.isfinite(teacher_waves).all():raise ValueError('Finite fixed source teacher waves required')
 net=AttentionRestorer().to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001);generator=np.random.default_rng(2026123056);losses=[];counts=dict(exposures=0,labeled=0,eligible=0);tick=time.monotonic()
 for epoch in range(1,(2 if a.smoke else 9)):
  total=0.;order=generator.permutation(len(pairs))
  for start in range(0,len(order),32):
   take=order[start:start+32];part=pairs[take];target_wave=teacher_waves[take.tolist()].to(a.device);ids=part[:,1];xx=np.stack([views[j][0][i] for j,i in part]);sc=[str(views[j][1][i]) for j,i in part];x=core.tensor(xx,a.device);y=core.tensor(clean[ids],a.device)
   with torch.no_grad():
    fy=teacher.features(y);logits=teacher.classify_features(fy);conf,pseudo=F.softmax(logits,1).max(1);lab=torch.tensor(labels[ids].tolist(),device=a.device,dtype=torch.long);known=lab>=0;target=torch.where(known,lab,pseudo);eligible=known|(conf>=.95)
   out=net(x);feature=teacher.features(out);pred=teacher.classify_features(feature);ce=F.cross_entropy(pred,target,reduction='none')*eligible.float();den=y.flatten(1).square().sum(1).clamp_min(1e-8);wave=(out-target_wave).flatten(1).square().sum(1)/den;semantic=(F.normalize(feature,dim=1)-F.normalize(fy,dim=1)).square().sum(1);clean_output=net(y);identity=(clean_output-y).flatten(1).square().sum(1)/den
   per=ce+wave+0.5*semantic+0.2*identity;groups=[(int(rx[i]),v) for i,v in zip(ids,sc)];unique=sorted(set(groups));loss=torch.stack([per[torch.as_tensor([g==u for g in groups],device=a.device)].mean() for u in unique]).mean()
   opt.zero_grad(set_to_none=True);loss.backward()
   if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or (epoch==1 and start==0 and net.output.weight.grad.abs().sum()==0):raise ValueError('Nonzero finite restoration gradient required')
   torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
   if not torch.isfinite(loss) or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('Finite loss and frozen F required')
   opt.step();total+=float(loss.detach())*len(ids)
   if epoch==1:counts['exposures']+=len(ids);counts['labeled']+=int(known.sum());counts['eligible']+=int(eligible.sum())
  losses.append(total/len(order));print('DISTILLED_ATTENTION_EPOCH',epoch,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',counts=counts,F_frozen=True,DURA_used_for_source_teacher=True,DURA_used_in_network=False,U_s_truth_reads=0,V_reads=0,test_reads=0));print('DISTILLED_ATTENTION_SOURCE_SMOKE PASS',flush=True);return
 with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=net.state_dict(),counts=counts,losses=losses,fit_ids=fit),f)
 core.write(a.out/'result.json',dict(status='TRAINING_COMPLETE',commit=a.commit,counts=counts,losses=losses,F_frozen=True,DURA_used_for_source_teacher=True,DURA_used_in_network=False,U_s_truth_reads=0,V_reads=0,test_reads=0,fit_ids=len(fit)))
 print('DISTILLED_ATTENTION_SOURCE_TRAINING_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()

"""Source-only posterior residual corrector with legal L_s supervision."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bayes_residual_model import ResidualCorrector,backproject,hybrid_loss

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-blind-bayes-source-validation-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='bcc0c56686ad57d0c101d7b0a7fc7822dd6e5ccb':raise ValueError('Exact source validation dependency required')
sys.path.insert(0,str(DEP/'experiments/phase1_blind_bayes_validation'))
from blind_bayes_source_validation import BlindActionNet,action_matrix,s,core,posterior,source_prior,SOURCE,SHA
from conditional_bayes_oracle import SCENES
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source
PRIOR_RUN='20261102-phase1-blind-bayes-source-validation-s2026092701-r01'
PRIOR_SHA='5e35516d54d3e3d1d1749933861f290c1f87c8ffdb4b863d2f9819914135ceb9'

@torch.no_grad()
def prior_output(model,x,means,covs,priors,eye):
 h,lv=model(x);A=action_matrix(h);N=eye[None,:,:]*lv.exp()[:,None,None];states,lp=posterior(x.reshape(len(x),512),A,N,means,covs,priors);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);return z,backproject(x,z,A)

def objective(net,teacher,z,b,y,labels):
 with torch.no_grad():yf=teacher.features(y);soft=F.softmax(teacher.classify_features(yf)/2,1)
 out=net(z,b);features=teacher.features(out);logits=teacher.classify_features(features);den=y.square().sum((1,2)).clamp_min(1e-8);iq=((out-y).square().sum((1,2))/den).mean();finger=(1-F.cosine_similarity(features,yf,dim=1,eps=1e-8)).mean();return .7*iq+.2*finger+hybrid_loss(logits,soft,labels)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026120854)
 checkpoint=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_path.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed blind model and prior required')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False)
 if saved['commit']!='94693a342bb2daa684c0f21415d7b949f9fb2107' or saved['epoch']!=10 or prior['source_ids']!=8192:raise ValueError('Exact source provenance required')
 means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);r_prior=np.random.default_rng(2026120355);prior_ids=np.sort(r_prior.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);rng=np.random.default_rng(2026120855);fit=pool;received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(r_prior.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(np.r_[fit,prior_ids],np.r_[validation,heldout]).size or np.intersect1d(fit,prior_ids).size:raise ValueError('Disjoint source roles required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);model=BlindActionNet().to(a.device);model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);source_truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=source_truth[fit];del source_truth
 if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only L_s train labels may enter optimization')
 train_labeled=int(np.count_nonzero(labels[fit]>=0));fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Exact original source physical IDs required')
 for epoch in (21,22,23):
  info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Exact fresh source view metadata required')
 rec,scene20=s.epoch_data(fresh,21);net=ResidualCorrector().to(a.device);labeled=fit[(roles[fit]=='L_s')&(scene20[fit]!='clean')][:4];unlabeled=fit[(roles[fit]=='U_s')&(scene20[fit]!='clean')][:4];index=np.r_[labeled,unlabeled]
 if len(labeled)!=4 or len(unlabeled)!=4:raise ValueError('Four L_s and four U_s source examples required')
 x=core.tensor(rec[index],a.device);y=core.tensor(clean[index],a.device);z,b=prior_output(model,x,means,covs,priors,eye);target=torch.tensor(labels[index].tolist(),device=a.device,dtype=torch.long);loss=objective(net,teacher,z,b,y,target);zero=torch.autograd.grad(loss*0,list(net.parameters()),retain_graph=True,allow_unused=True)
 if not torch.isfinite(loss) or any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise ValueError('Zero gradient source smoke failed')
 loss.backward()
 if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or not net.output.weight.grad.abs().sum()>0:raise ValueError('Finite nonzero source gradient required')
 if any(p.requires_grad or p.grad is not None for p in list(model.parameters())+list(teacher.parameters())):raise ValueError('Fixed prior model and F required')
 net.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=8,L_s_records=4,U_s_records=4,zero_gradient=True,finite_nonzero_gradient=True,F_frozen=True,blind_parent_frozen=True,DURA_loaded=False,L_s_metadata_labels=6300,train_L_s_labels=train_labeled,U_s_truth_reads=0,V_reads=0,test_reads=0);print('BAYES_RESIDUAL_SMOKE',json.dumps(smoke),flush=True)
 if a.smoke:
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
  return
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);total=0.;used=0;used_labeled=0;updates=0;tick=time.monotonic();epoch_losses=[];epoch_counts=[];net.train()
 for epoch in (21,22,23):
  rec,scene20=s.epoch_data(fresh,epoch);order=rng.permutation(fit);epoch_total=0.;epoch_used=0
  for start in range(0,len(order),16):
   idx=order[start:start+16];idx=idx[scene20[idx]!='clean']
   if not len(idx):continue
   x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);z,b=prior_output(model,x,means,covs,priors,eye);target=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);l=objective(net,teacher,z,b,y,target)
   if not torch.isfinite(l):raise FloatingPointError('Nonfinite source residual objective')
   opt.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();used+=len(idx);epoch_used+=len(idx);used_labeled+=int((target>=0).sum().item());total+=float(l.detach())*len(idx);epoch_total+=float(l.detach())*len(idx);updates+=1
   if updates%256==0:print('BAYES_RESIDUAL_FULL_PROGRESS',epoch,updates,'active',used,'labeled',used_labeled,'loss',total/used,'seconds',round(time.monotonic()-tick,1),flush=True)
  if epoch_used<30000:raise ValueError('Insufficient source impaired records')
  epoch_counts.append(epoch_used);epoch_losses.append(epoch_total/epoch_used);print('BAYES_RESIDUAL_FULL_EPOCH',epoch,'active',epoch_used,'loss',epoch_losses[-1],flush=True)
 if updates<6000 or used_labeled<3000 or any(p.grad is not None for p in list(model.parameters())+list(teacher.parameters())):raise ValueError('Training source/freeze contract violated')
 with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=net.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
 net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);z,b=prior_output(model,x,means,covs,priors,eye);variants={'raw':x,'dura':dura(x),'parent':z,'corrected':net(z,b)};den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
   for name,signal in variants.items():
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   out.append(torch.stack(cols,1).cpu())
   if (start+len(idx))%64==0:print('BAYES_RESIDUAL_EVALUATED',start+len(idx),flush=True)
 values=torch.cat(out);results={}
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()))
  for i,name in enumerate(('raw','dura','parent','corrected')):cell[name]=dict(correct=int((mask&(rows[:,2+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,3+2*i].mean().item()))
  results[view]=cell
 result=dict(status='COMPLETE_SOURCE_ONLY_RESIDUAL_FULL',commit=a.commit,scenes=results,fit_ids=len(fit),train_L_s_labels=train_labeled,used=used,used_labeled=used_labeled,updates=updates,mean_loss=total/used,epoch_losses=epoch_losses,epoch_counts=epoch_counts,probe_ids=len(ids),probe_receivers=[8],parent_sha256=SHA,prior_sha256=PRIOR_SHA,source_only=True,F_frozen=True,blind_parent_frozen=True,DURA_loaded_after_training=True,L_s_metadata_labels=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Only labels from L_s fit IDs entered loss; L_s metadata file contained 6300 labels; source IDs and views previously explored, pseudo targets not true test accuracy.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('BAYES_RESIDUAL_FULL_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

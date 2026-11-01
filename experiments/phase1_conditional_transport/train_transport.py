"""Source-only conditional flow matching in frozen blind posterior coordinates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transport_model import ConditionalTransport,backproject,integrate
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,labels_source,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES
from bayes_residual_model import hybrid_loss

@torch.no_grad()
def initial(blind,x,means,covs,priors,eye):
 h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],means,covs,priors);parent=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);return parent,A,lv

def objective(net,teacher,x,y,parent,A,lv,labels):
 with torch.no_grad():yf=teacher.features(y);soft=F.softmax(teacher.classify_features(yf)/2,1);t=torch.rand(len(x),device=x.device);state=parent+t[:,None,None]*(y-parent);residual=backproject(x,state,A)
 velocity=net(state,parent,residual,t,lv);target=y-parent;den=y.square().sum((1,2)).clamp_min(1e-8);matching=((velocity-target).square().sum((1,2))/den).mean();endpoint=state+(1-t[:,None,None])*velocity;features=teacher.features(endpoint);logits=teacher.classify_features(features);feature=(1-F.cosine_similarity(features,yf,dim=1,eps=1e-8)).mean();loss=matching+.2*feature+hybrid_loss(logits,soft,labels);return loss,matching.detach()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121454)
 ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_file.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact blind checkpoint and source prior required')
 saved=torch.load(ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False)
 if saved['commit']!='94693a342bb2daa684c0f21415d7b949f9fb2107' or saved['epoch']!=10 or prior['source_ids']!=8192:raise ValueError('Fixed source provenance required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(saved['model'],strict=True);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);prng=np.random.default_rng(2026120355);prior_ids=np.sort(prng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);rng=np.random.default_rng(2026121455);fit=np.sort(rng.choice(pool,size=4096,replace=False));received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(prng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES])
 if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source IDs required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=truth[fit];del truth
 if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only L_s fit truth may enter optimization')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Exact source physical IDs required')
 for epoch in (21,22,23):
  info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
  if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Exact source view provenance required')
 rec,scene21=s.epoch_data(fresh,21);labeled=fit[(roles[fit]=='L_s')&(scene21[fit]!='clean')][:4];unlabeled=fit[(roles[fit]=='U_s')&(scene21[fit]!='clean')][:4];idx=np.r_[labeled,unlabeled]
 if len(labeled)!=4 or len(unlabeled)!=4:raise ValueError('Four L_s and four U_s source records required')
 net=ConditionalTransport().to(a.device);x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);parent,A,lv=initial(blind,x,means,covs,priors,eye);lab=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);loss,matching=objective(net,teacher,x,y,parent,A,lv,lab);zero=torch.autograd.grad(loss*0,list(net.parameters()),retain_graph=True,allow_unused=True)
 if not torch.isfinite(loss) or any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise ValueError('Zero gradient source preflight required')
 loss.backward()
 if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or net.output.weight.grad.abs().sum()==0 or any(p.grad is not None or p.requires_grad for p in list(blind.parameters())+list(teacher.parameters())):raise ValueError('Finite flow gradient and frozen models required')
 net.zero_grad(set_to_none=True);smoke=dict(status='PASS',source_fit_ids=len(fit),L_s_metadata_labels=6300,fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),mixed_L_s=4,mixed_U_s=4,zero_gradient=True,finite_nonzero_gradient=True,F_frozen=True,blind_parent_frozen=True,DURA_loaded=False,U_s_truth_reads=0,V_reads=0,test_reads=0);print('TRANSPORT_SOURCE_SMOKE',json.dumps(smoke),flush=True)
 if a.smoke:
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
  return
 opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001);losses=[];useds=[];updates=0;labeled_used=0;tick=time.monotonic()
 for epoch in (21,22,23):
  rec,sc=s.epoch_data(fresh,epoch);order=rng.permutation(fit);total=0.;used=0
  for start in range(0,len(order),16):
   idx=order[start:start+16];idx=idx[sc[idx]!='clean']
   if not len(idx):continue
   x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);parent,A,lv=initial(blind,x,means,covs,priors,eye);lab=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);loss,matching=objective(net,teacher,x,y,parent,A,lv,lab)
   if not torch.isfinite(loss):raise FloatingPointError('Finite conditional transport loss required')
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();used+=len(idx);labeled_used+=int((lab>=0).sum().item());updates+=1;total+=float(loss.detach())*len(idx)
  if used<3000:raise ValueError('Expected impaired source training exposure')
  losses.append(total/used);useds.append(used);print('TRANSPORT_EPOCH',epoch,'active',used,'loss',losses[-1],'updates',updates,'seconds',round(time.monotonic()-tick,1),flush=True)
 if any(p.grad is not None for p in list(blind.parameters())+list(teacher.parameters())):raise ValueError('Ground F and blind parent must remain frozen')
 with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,epoch=3,model=net.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
 net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);parent,A,lv=initial(blind,x,means,covs,priors,eye);variants=[x,dura(x),parent]+[integrate(net,x,parent,A,lv,k) for k in (1,4,8)];den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
   for signal in variants:
    pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
   batch=torch.stack(cols,1).cpu()
   if not torch.isfinite(batch).all():raise ValueError('Finite source predictions required')
   out.append(batch)
 values=torch.cat(out);results={};names=('raw','dura','parent','flow_1','flow_4','flow_8')
 for j,view in enumerate(SCENES):
  rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum().item()))
  for i,name in enumerate(names):cell[name]=dict(correct=int((valid&(rows[:,2+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,3+2*i].mean().item()))
  results[view]=cell
 result=dict(status='COMPLETE_SOURCE_ONLY_TRANSPORT_PILOT',commit=a.commit,fit_ids=len(fit),fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),L_s_metadata_labels=6300,active_views=useds,used_labeled=labeled_used,updates=updates,epoch_losses=losses,scenes=results,probe_ids=len(ids),F_frozen=True,blind_parent_frozen=True,DURA_loaded_after_training=True,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source L_s fit labels and U_s clean-F soft predictions only; no U_s truth. Observed source IDs and pseudo targets do not establish test accuracy or physical validity.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('TRANSPORT_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

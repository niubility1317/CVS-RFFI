"""Phase1 source-only learned self-reference, frozen-ground selection."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase1_self_reference_model import SelfReferenceNet
from self_reference_selection import choose,VIEWS

DEPENDENCY=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-phase-shift-student-s2026092701-r02-release')
if (DEPENDENCY/'release_commit.txt').read_text().strip()!='284317bd819df974395d3388b1920a41300937cb':raise ValueError('Exact Phase1 phase/ground dependency release required')
sys.path.insert(0,str(DEPENDENCY/'experiments/phase1_phase_shift_restorer'))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as source_prior

ALPHAS=(.5,1.);THRESHOLDS=(.8,.99);EPOCHS=(1,2,3)

def objective(net,teacher,anchor,clean):
 with torch.no_grad():
  features=teacher.features(clean);logits=teacher.classify_features(features);soft=F.softmax(logits/2,1);aligned,_=s.align(clean,anchor)
 out=net(anchor);feat=teacher.features(out);logits=teacher.classify_features(feat);kl=F.kl_div(F.log_softmax(logits/2,1),soft,reduction='none').sum(1).mean()*4;finger=(1-F.cosine_similarity(feat,features,dim=1,eps=1e-8)).mean();den=aligned.square().mean((1,2)).clamp_min(1e-8);iq=((out-aligned).square().mean((1,2))/den).mean();inv=s.complex_gain_invariant_error(out,aligned).mean();return kl+.5*finger+.3*iq+.2*inv


def smoke(net,phase,gate,teacher,clean,received,scene,ids,device):
 index=ids[scene[ids]!='clean'][:32]
 if len(index)!=32:raise ValueError('32 impaired source records required')
 x=core.tensor(received[index],device);y=core.tensor(clean[index],device)
 with torch.no_grad():active=gate(x).sigmoid()>.8;anchor=phase.anchor(x[active])
 if int(active.sum())<1:raise ValueError('Active source records required')
 loss=objective(net,teacher,anchor,y[active]);params=list(net.parameters());zero=torch.autograd.grad(loss*0,params,retain_graph=True,allow_unused=True)
 if not torch.isfinite(loss) or any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise ValueError('Source zero-gradient check failed')
 loss.backward()
 if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or not net.output.weight.grad.abs().sum()>0:raise ValueError('Finite nonzero self-reference gradient required')
 if any(p.requires_grad or p.grad is not None for p in list(teacher.parameters())+list(phase.parameters())+list(gate.parameters())):raise ValueError('Frozen Phase1 components required')
 net.zero_grad(set_to_none=True);return dict(status='PASS',records=len(index),active=int(active.sum()),zero_gradient=True,finite_nonzero_gate_gradient=True,F_frozen=True,phase_frozen=True,gate_frozen=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0)


def evaluate(net,phase,gate,dura,teacher,clean,received,scene,rx,validation,heldout,device):
 ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];rows=[]
 with torch.no_grad():
  for start in range(0,len(ids),32):
   idx=ids[start:start+32];x=core.tensor(received[idx],device);y=core.tensor(clean[idx],device);a=phase.anchor(x);full=net(a);p=F.softmax(teacher.classify_features(teacher.features(y)),1);target=p.argmax(1);confidence=p.max(1).values;prob=gate(x).sigmoid();d=dura(x);dp=teacher.classify_features(teacher.features(d)).argmax(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);cols=[target.float(),confidence,dp.float(),prob,de/den,raw.float(),re/den]
   for alpha in ALPHAS:
    z=a+alpha*(full-a);zp=teacher.classify_features(teacher.features(z)).argmax(1);_,ze=s.align(z,y);cols.extend((zp.float(),ze/den))
   rows.append(torch.stack(cols,1).cpu())
 v=np.from_dlpack(torch.cat(rows).contiguous());result={}
 for ai,alpha in enumerate(ALPHAS):
  for threshold in THRESHOLDS:
   cells={}
   for part,receivers in ((1,(4,6)),(2,(8,))):
    for receiver in receivers:
     for view in VIEWS:
      mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);a=v[mask]
      if not len(a):raise ValueError('All source receiver/scene cells required')
      trusted=a[:,1]>=.95;active=a[:,3]>threshold
      if not trusted.any():raise ValueError('Confident clean-F pseudo targets required')
      pred=np.where(active,a[:,7+2*ai],a[:,5]);iq=np.where(active,a[:,8+2*ai],a[:,6]);cells[f'{part}:{receiver}:{view}']=dict(count=len(a),valid_count=int(trusted.sum()),active=int(active.sum()),baseline_correct=int((trusted&(a[:,2]==a[:,0])).sum()),candidate_correct=int((trusted&(pred==a[:,0])).sum()),baseline_iq=float(a[:,4].mean()),candidate_iq=float(iq.mean()))
   result[f'{alpha}:{threshold}']=cells
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output root exists')
 if a.device!='cpu':raise ValueError('This run is predeclared CPU only')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(2026120254);rng=np.random.default_rng(2026120255)
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);pool=training
 if np.intersect1d(pool,np.r_[validation,heldout]).size:raise ValueError('Disjoint source physical IDs required')
 phase,gate=s.initial_models(a.root,a.device);phase.eval().requires_grad_(False);teacher,_=core.teacher(a.root,a.device);net=SelfReferenceNet().to(a.device);received,scene=s.epoch_data(cache,11)
 if a.smoke:
  result=smoke(net,phase,gate,teacher,clean,received,scene,pool,a.device)
  with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
  print('SELF_REFERENCE_SOURCE_SMOKE',json.dumps(result),flush=True);return
 opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);states={};losses=[];counts=[]
 for epoch in EPOCHS:
  received,scene=s.epoch_data(cache,epoch+10);order=rng.permutation(pool);net.train();count=0;total=0.;start_time=time.monotonic()
  for start in range(0,len(order),32):
   idx=order[start:start+32];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device)
   with torch.no_grad():active=gate(x).sigmoid()>.8
   if not active.any():continue
   x=x[active];y=y[active]
   with torch.no_grad():anchor=phase.anchor(x)
   loss=objective(net,teacher,anchor,y)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source objective')
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();count+=len(x);total+=float(loss.detach())*len(x)
   if (start//32+1)%128==0:print('SELF_REFERENCE_PROGRESS',epoch,'batches',start//32+1,'active',count,'loss',total/count,'elapsed',round(time.monotonic()-start_time,1),flush=True)
  if count<1000 or any(p.grad is not None for p in list(phase.parameters())+list(gate.parameters())+list(teacher.parameters())):raise ValueError('Frozen source components or active count violated')
  losses.append(total/count);counts.append(count);states[epoch]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  with (a.out/('checkpoint_epoch%02d.pt'%epoch)).open('xb') as f:torch.save(dict(commit=a.commit,epoch=epoch,model=states[epoch]),f)
  print('SELF_REFERENCE_EPOCH',epoch,'active',count,'loss',losses[-1],flush=True)
 # Frozen DURA is loaded only for source validation, after all candidate weights are fixed.
 received,scene=s.epoch_data(cache,19);dura=source_prior.frozen_dura(a.root,a.device);results={}
 for epoch in EPOCHS:
  net.load_state_dict(states[epoch]);net.eval().requires_grad_(False);results[epoch]=evaluate(net,phase,gate,dura,teacher,clean,received,scene,rx,validation,heldout,a.device);print('SELF_REFERENCE_EVALUATED',epoch,flush=True)
 selected=choose(results)
 if selected['kind']!='dura':
  with (a.out/'selected.pt').open('xb') as f:torch.save(dict(commit=a.commit,selected=selected,model=states[selected['epoch']],phase=phase.phase.state_dict(),gate=gate.state_dict()),f)
 result=dict(selected=selected,validation=results,training_ids=len(pool),validation_ids=len(validation),heldout_ids=len(heldout),losses=losses,active_counts=counts,source_cache_epoch=19,source_only=True,F_frozen=True,phase_frozen=True,gate_frozen=True,DURA_loaded_only_after_training=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='source physical IDs previously observed in other simulated views; pseudo targets are clean-F predictions, not true labels')
 with (a.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('SELF_REFERENCE_SELECTION',json.dumps(selected),flush=True)
if __name__=='__main__':main()

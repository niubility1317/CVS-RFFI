"""Focused default-head joint updates on real CVS, synthetic source IQ only."""
import copy,random
import torch
from . import design as d,model as api
from .iq import Runtime,AugmentorProxy
from experiments.cvs_additive_as20 import model as original
from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from SSDG import train_ssdg as native

def run():
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1;return
  raise AssertionError('Unauthorized config accepted')
 for k,v in [('epochs',50),('model_seed',2026092702),('checkpoint_sources',['old']),('target_access',True),('separate_head_training',True)]:
  c=copy.deepcopy(d.config(d.rows()[0]));c[k]=v;reject(lambda:d.validate(c))
 for k,v in [('scope','U'),('mode','random'),('seed',0),('second','stronger'),('NN','newhead')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c['view_refresh'][k]=v;reject(lambda:d.validate(c))
 g=torch.Generator().manual_seed(17834);x=torch.randn(128,2,256,generator=g);saved=x.clone();labels=torch.arange(128)%6;rx=[d.FIT[(i//3)%4] for i in range(128)];day=[i%3+1 for i in range(128)];domains=torch.tensor([3+d.FIT.index(r)*3+dd-1 for r,dd in zip(rx,day)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(rx),day_i=torch.tensor(day),sig_i=torch.arange(128),base_index=torch.arange(128))
 initial=[];loss=[];grads=[];ends=[];reports=[]
 for arm in ('ORIGINAL','RAW','REPLAY','REFRESH'):
  c=d.config(next(r for r in d.rows() if r['arm']==('RAW' if arm=='ORIGINAL' else arm)));a=d.make_args(c,'cpu');cfg=native.build_aug_base_cfg(a);base=native.make_augmentor(cfg);r=Runtime('RAW' if arm=='ORIGINAL' else arm);r.calls=r.consumed=1110;r.set_L(meta);aug=base if arm=='ORIGINAL' else AugmentorProxy(base,r);native.configure_augmentor_for_epoch(aug,cfg,6,a);base.p_phase_rot=1.
  torch.manual_seed(17835);random.seed(17835);ordinary=aug(x,labels=labels,no_pa=False);augstate=(torch.get_rng_state().clone(),random.getstate())
  torch.manual_seed(d.SEED);m=(original.build(d.template(),'cpu') if arm=='ORIGINAL' else api.build(c,'cpu')).train();initial.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  if arm!='ORIGINAL':
   m._refresh_runtime=r;teacher=copy.deepcopy(m);assert teacher._refresh_runtime is None
  selected=x if arm in ('ORIGINAL','RAW') else r.first.clone() if arm=='REPLAY' else r.fresh.clone()
  seen=[];features=m.id_backbone.encoder._view_features
  def trace(value):seen.append(value.detach().clone());return features(value)
  m.id_backbone.encoder._view_features=trace;torch.set_rng_state(augstate[0]);random.setstate(augstate[1])
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01);corr,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+corr
  loss.append(total.detach());grads.append(torch.autograd.grad(total,tuple(m.parameters()),allow_unused=True,retain_graph=True));ends.append((torch.get_rng_state().clone(),random.getstate()))
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(t,x) for t in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(seen[6],selected) and torch.equal(saved,x) and m._joint_L_packets==1024 and m._joint_As_packets==0 and m._supcon_applied==1 and not m._supcon_handles and m._supcon_pending is None
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W))
  m.id_backbone.encoder._view_features=features;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,export(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=arm,L_forwards=8,same_ordinary_IQ_twice=True,raw_four_forwards_unchanged=True,existing_SupCon_only=True,default_head_joint_update=True,export160_exact=True,new_parameters=0,synthetic_loss=float(total.detach()),runtime_calls=0 if arm=='ORIGINAL' else r.calls-1110))
 assert all(torch.equal(v,s[k]) for s in initial[1:] for k,v in initial[0].items()) and torch.equal(loss[0],loss[1]) and all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(grads[0],grads[1])) and torch.equal(ends[0][0],ends[1][0]) and ends[0][1]==ends[1][1]
 changed=sum(a is not None and b is not None and not torch.equal(a,b) for a,b in zip(grads[2],grads[3]));assert changed>0
 return dict(status='PASS',protocol_negatives=negative,rows=reports,all_initial206states_exact=True,RAW8_native_loss_all_gradients_RNG_bitwise=True,REFRESH_changed_parameter_gradients=changed,synthetic_NN_packets=4120,synthetic_optimizer_updates=4,synthetic_model_forced_native_phase_gate=True,real_IQ=0,model_checkpoint_loads=0,target_access=False,baseline_network_and_head_unchanged=True)

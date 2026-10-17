"""Full fixed CVS synthetic graph, matched four policies, no real IQ/checkpoint."""
import copy,json,random
import torch
from . import design as d,model as api
from SSDG import train_ssdg as native
from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def run():
 torch.set_num_threads(2);x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(202610172010));saved=x.clone();y=torch.arange(128)%6;rx=[d.FIT[(i//3)%4] for i in range(128)];days=[i%3+1 for i in range(128)];domains=torch.tensor([d.RX.index(r)*3+dd-1 for r,dd in zip(rx,days)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(rx),day_i=torch.tensor(days),sig_i=torch.arange(128),base_index=torch.arange(128));state=None;models={};base_losses=[];ends=[];rows=[]
 for row in d.rows():
  torch.manual_seed(d.SEED);m=api.build(d.config(row),'cpu').train()
  if state is None:state={k:v.detach().clone() for k,v in m.state_dict().items()}
  else:assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  assert len(state)==206 and sum(p.numel() for p in m.parameters())==900722 and not m._additive_bn_names;teacher=copy.deepcopy(m);assert teacher._group_steps==teacher._group_generated==0;models[row['arm']]=m
 for arm,m in models.items():
  torch.manual_seed(202610172011);random.seed(202610172011);c=d.config(next(r for r in d.rows() if r['arm']==arm));a=d.make_args(c,'cpu');cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a);ordinary=aug(x,labels=y,no_pa=False);old=m.id_backbone.encoder._view_features;seen=[]
  def trace(v):seen.append(v.detach().clone());return old(v)
  m.id_backbone.encoder._view_features=trace;m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);corr,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+corr;r=m._joint_details[-1]
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[1],ordinary) and all(torch.equal(v,x) for v in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved)
  assert (torch.equal(seen[6],x))==(arm=='RAW') and m._group_steps==1 and m._group_generated==128 and sum(m._group_coverage.values())==128 and m._joint_As_packets==(0 if arm=='RAW' else 128) and r['group_shadow_generated']==128 and m._supcon_pending is None and not m._supcon_handles and m._joint_L_packets==1024 and m._additive_L_packets==256 and m._additive_calls==1
  assert abs(r['total_L']-r['base_total_L']-.25*r['extra_pair_loss'])<2e-6;base_losses.append(r['base_total_L']);ends.append((torch.get_rng_state().clone(),random.getstate()));m.id_backbone.encoder._view_features=old;W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert any(p.grad is not None and bool((p.grad!=0).any()) for n,p in m.named_parameters() if n!=W);clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(x[:2]);e=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,e(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  rows.append(dict(arm=arm,base_loss=r['base_total_L'],extra_input=r['extra_input'],same_IQ_pair=True,As=r['extra_As_packets'],generated=r['group_shadow_generated'],groups=r['group_session_count'],default_head_joint_update=True))
 assert all(v==base_losses[0] for v in base_losses) and all(torch.equal(ends[0][0],v[0]) and ends[0][1]==v[1] for v in ends[1:])
 sg=models['SCENE_GROUP']._joint_details[-1];pg=models['PROP_GROUP']._joint_details[-1]
 for k in ('group_session','group_scene','group_environment_seed','group_packet_seed','group_noise_stream_seed','group_packet_residual_hz','group_packet_coarse_error_hz','group_packet_phase','group_packet_tracking_z'):assert sg[k]==pg[k]
 neg=0
 for key,value in [('epochs',50),('model_seed',2026092702),('target_access',True),('baseline_reuse',True)]:
  bad=copy.deepcopy(d.config(d.rows()[0]));bad[key]=value
  try:d.validate(bad)
  except ValueError:neg+=1
  else:raise AssertionError('Protected config override accepted')
 return dict(status='PASS',synthetic_NN=4120,optimizer_updates=4,synthetic_channel_processes=512,native_initializations=1024,source_IQ=0,CP_reads=0,rows=rows,all206_initial_states_equal=True,allbase_losses_exact=True,CPU_Python_rng_endstates_equal=True,unchanged_default_joint160=True,extra_U_views=0,negative=neg)
if __name__=='__main__':print(json.dumps(run()))

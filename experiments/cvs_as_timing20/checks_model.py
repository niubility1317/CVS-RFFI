"""Real fixed CVS graph,synthetic IQ only;timing inputs and joint defaulthead updates."""
import copy,json,random
import torch
from . import design as d,model as api
from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from SSDG import train_ssdg as native
def run():
 torch.set_num_threads(2);g=torch.Generator().manual_seed(20261017991);x=torch.randn(128,2,256,generator=g);y=torch.arange(128)%6;rx=[d.FIT[(i//3)%4] for i in range(128)];days=[i%3+1 for i in range(128)];domains=torch.tensor([3+d.FIT.index(r)*3+dd-1 for r,dd in zip(rx,days)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(rx),day_i=torch.tensor(days),sig_i=torch.arange(128),base_index=torch.arange(128));saved=x.clone();models={};state=None;observations=[];count=0
 for row in d.rows():
  torch.manual_seed(d.SEED);m=api.build(d.config(row),'cpu').train()
  if state is None:state={k:v.detach().clone() for k,v in m.state_dict().items()}
  else:assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  assert len(m.state_dict())==206 and not m._additive_bn_names and sum(p.numel() for p in m.parameters())==900722
  teacher=copy.deepcopy(m);assert teacher._timing_calls==0 and teacher._timing_generated==0
  models[row['arm']]=m
 for epoch in (6,14):
  actual={};shadow={};losses={}
  for arm,m in models.items():
   torch.manual_seed(20261017992+epoch);random.seed(20261017992+epoch);a=d.make_args(d.config(next(r for r in d.rows() if r['arm']==arm)),'cpu');cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,epoch,a);ordinary=aug(x,labels=y,no_pa=False)
   old=m.id_backbone.encoder._view_features;seen=[]
   def trace(v):seen.append(v.detach().clone());return old(v)
   m.id_backbone.encoder._view_features=trace;m._feature_l_steps=m._joint_calls=m._legacy_calls=(epoch-1)*222
   with m._phase1_labeled_feature_context():z=m(ordinary,return_aux=True)
   ce=torch.nn.functional.cross_entropy(z['tx_logits'],y,label_smoothing=.01);corr,_=m._source_labeled_feature_hook(model=m,labeled_output=z,labels=y,extra=(domains,meta),domains=domains,optimizer_step=(epoch-1)*222+1,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+corr;record=m._joint_details[-1];active=d.active(arm,epoch)
   assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(v,x) for v in seen[2:6]) and torch.equal(seen[6],seen[7]) and (not torch.equal(seen[6],x) if active else torch.equal(seen[6],x)) and torch.equal(saved,x) and record['extra_As_packets']==128*active and record['As_shadow_generated']==128 and record['As_active']==active and m._supcon_pending is None and not m._supcon_handles
   assert abs(record['total_L']-record['base_total_L']-.25*record['extra_pair_loss'])<2e-6 and record['extra_view_once'] and record['extra_same_IQ_pair']
   actual[arm]=seen[-1];shadow[arm]={k:record[k] for k in ('shadow_raw_MSE','shadow_scene_counts','shadow_state_counts','shadow_lock_counts')};losses[arm]=total.detach();m.id_backbone.encoder._view_features=old;count+=1024
   if epoch==14:
    W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W))
    m.eval()
    with torch.no_grad(),fresh_parameter_caches(m):
     f=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,export(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
    count+=6
  assert all(v==shadow['RAW'] for v in shadow.values());aa='EARLY7' if epoch==6 else 'LATE7';bb='LATE7' if epoch==6 else 'EARLY7';assert torch.equal(actual['ALL'],actual[aa]) and torch.equal(actual['RAW'],actual[bb]) and torch.equal(losses['ALL'],losses[aa]) and torch.equal(losses['RAW'],losses[bb]);observations.append(dict(epoch=epoch,active_pair=aa,inactive_pair=bb,all_shared_physics_statistics_exact=True))
 return dict(status='PASS',model_states=206,new_parameters=0,default_head_joint_updates=4,synthetic_NN=count,synthetic_native_physical_calls=1024,real_IQ=0,CP=0,target_access=False,observations=observations)
if __name__=='__main__':print(json.dumps(run()))

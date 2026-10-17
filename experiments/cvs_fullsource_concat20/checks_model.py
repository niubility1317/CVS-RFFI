"""Actual F/default head and optimizer; synthetic IQ only, all five source RX."""
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from . import design as d,model as api,supcon

def run():
 from .checks_protocol import run as protocol
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);proof=protocol();reports=[];initial=None;negative=0;NN=0;physics=0
 for row in d.rows():
  torch.manual_seed(20261017701);c=d.config(row);m=api.build(c,'cpu');state={k:v.detach().clone() for k,v in m.state_dict().items()}
  if initial is None:initial=state
  else:assert set(initial)==set(state) and all(torch.equal(initial[k],v) for k,v in state.items())
  args=d.make_args(c,'cpu');assert getattr(args,'source_fold_held_rx',None) is None and not getattr(args,'source_fold_training_receivers',None)
  replica=deepcopy(m).eval();assert replica._full_channel is None or replica._full_channel is not m._full_channel;del replica
  m.train();m._feature_l_steps=m._full_calls=m._legacy_calls=1110
  torch.manual_seed(20261017702);raw=torch.randn(128,2,256);ordinary=raw*.9;labels=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
  meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
  assert set(rx.tolist())==set(d.FIT)
  seen=[];original=m.id_backbone.encoder._full_features
  def traced(x):seen.append(x.detach().clone());return original(x)
  m.id_backbone.encoder._full_features=traced
  opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);W=m.get_parameter('id_backbone.encoder.core.id_backbone.cls_head.weight');before=W.detach().clone()
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
  correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
  total=ce+correction;n=2 if row['arm']=='NATIVE' else 4;assert len(seen)==n and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1])
  if n==4:assert torch.equal(seen[2],seen[3]) and (torch.equal(seen[2],ordinary)==(row['arm']=='REPEAT'))
  record=m._full_records[0];expected=record['common_RD'] if n==2 else .5*(record['common_RD']+record['extra_RD']);expected+=record['SupCon_weight']*record['SupCon'];assert abs(float(total.detach())-expected)<2e-6
  assert m._supcon_applied==1 and len(record['SupCon_details'])==4 and not m._supcon_handles and m._supcon_pending is None and m._legacy_actual_L_input is None
  total.backward();assert W.grad is not None and torch.isfinite(W.grad).all();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,W)
  m.id_backbone.encoder._full_features=original;m.eval();calls=m._full_calls
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(raw[:2]);export=api.export(m).eval();ef=export(raw[:2]);assert f.shape==(2,160) and torch.equal(f,ef)
  assert m._full_calls==calls;NN+=n*128+4
  if m._full_channel is not None:
   physics+=128;ch=m._full_channel;sm={k:v[:15] for k,v in meta.items()};cpu=torch.get_rng_state();py=random.getstate();np0=np.random.get_state()
   a,ra=ch.apply(raw[:15],sm,epoch=6,step=1112);b,rb=ch.apply(raw[:15],sm,epoch=6,step=1112);physics+=30
   assert torch.equal(a,b) and ra==rb and torch.equal(cpu,torch.get_rng_state()) and py==random.getstate();np1=np.random.get_state();assert np0[0]==np1[0] and np.array_equal(np0[1],np1[1]) and np0[2:]==np1[2:]
   assert {k[0] for k in ch.coverage}==set(d.FIT)
   for key,val in (('rx_i',torch.full((15,),2)),('day_i',torch.zeros(15,dtype=torch.long)),('eq_i',torch.zeros(15,dtype=torch.long))):
    bad=dict(sm);bad[key]=val
    try:ch.apply(raw[:15],bad,epoch=6,step=1112)
    except ValueError:negative+=1
    else:raise AssertionError('Forbidden source role accepted')
   audit=ch.audit();assert all(v['processing_route']=='residual' and v['mode']=='post_sync' and not v['equalization_enabled'] for v in audit['configurations'].values())
  reports.append(dict(arm=row['arm'],L_forwards=n,parameters=sum(p.numel() for p in m.parameters()),default_head_joint_update=True,export160_exact=True,full_source_RX=list(d.FIT),SupCon_layer_forward_terms=4,same_IQ_pairs=True))
 from . import evaluate as ev,proof as pv
 original_read,original_ids,original_source=d.read,ev.ids,pv.source
 try:
  pv.source=lambda c:None;ev.ids=lambda:np.arange(ev.COUNT,dtype=np.int64)
  valid=dict(status='PREDICTIONS_COMPLETE',config=d.config(d.rows()[0]),count=ev.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,bad in (('status','PENDING'),('count',5),('views',['clean']),('truth_read',True),('query_fit',True),('head_fit_updates',1)):
   value=dict(valid);value[key]=bad
   def fake_read(path):
    if Path(path).name=='reference_recipe.json':return original_read(path)
    if Path(path).name=='complete.json' and Path(path).parent.name=='prediction':return value
    raise AssertionError('Truth or unregistered data opened before all views fixed')
   d.read=fake_read
   try:ev.score(d.rows()[0]['row_id'])
   except ValueError:negative+=1
   else:raise AssertionError('Incomplete prediction reached scorer')
 finally:d.read,ev.ids,pv.source=original_read,original_ids,original_source
 from .source import RecordingLoader
 meta=dict(rx_i=torch.tensor([1,3,4,6,8]),day_i=torch.ones(5,dtype=torch.long),eq_i=torch.ones(5,dtype=torch.long),sig_i=torch.arange(5),base_index=torch.arange(5));x=torch.ones(5,2,256);domain=torch.arange(5)
 import io
 assert len(list(RecordingLoader([(x,torch.full((5,),-1),domain,meta)],'U',io.StringIO())))==1
 try:list(RecordingLoader([(x,torch.zeros(5,dtype=torch.long),domain,meta)],'U',io.StringIO()))
 except ValueError:negative+=1
 else:raise AssertionError('U TX truth accepted by trace loader')
 return dict(status='PASS',protocol=proof,role_negative_cases=negative,rows=reports,synthetic_NN=NN,synthetic_physics_packets=physics,synthetic_optimizer_steps=3,real_IQ=0,CP=0,target_access=False)

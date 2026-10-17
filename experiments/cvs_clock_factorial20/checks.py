"""Synthetic protocol and exact B00-equivalence checks; zero real IQ/target."""
from copy import deepcopy
from pathlib import Path
import io,random
import numpy as np
import torch
from . import design as d,model as api
from .clock import resample
from experiments.cvs_fullsource_concat20 import model as old,supcon

def protocol():
 count=0
 for row in d.rows():
  c=d.config(row);d.validate(c)
  mutations=[('epochs',21),('label_epochs',12),('baseline_reuse',not c['baseline_reuse']),('head_fit_updates',1),('classifier','linear'),('new_Ag',not c['new_Ag']),('As',not c['As'])]
  for key,value in mutations:
   bad=deepcopy(c);bad[key]=value
   try:d.validate(bad)
   except ValueError:count+=1
   else:raise AssertionError('Protocol mutation accepted '+key)
  for area,key,value in (('clock','max_ppm',81.),('data_contract','RX',[1,3,4,6]),('data_contract','L',5580),('factorial','composition_view',True),('factorial','common_pair_weight',.4)):
   bad=deepcopy(c);bad[area][key]=value
   try:d.validate(bad)
   except ValueError:count+=1
   else:raise AssertionError('Nested protocol mutation accepted')
 return count

def run():
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=protocol();reports=[];NN=0;physics=0;clock=0;initial=None
 torch.manual_seed(202610175412);x=torch.randn(4,2,256)
 assert torch.equal(resample(x,torch.zeros(4,dtype=torch.float64)),x)
 a=resample(x,torch.tensor([-80.,-40.,40.,80.],dtype=torch.float64));assert torch.isfinite(a).all() and not torch.equal(a,x)
 for p in (torch.tensor([81.]*4),torch.tensor([float('nan')]*4)):
  try:resample(x,p)
  except ValueError:negative+=1
  else:raise AssertionError('Clock range accepted')
 for row in d.rows():
  c=d.config(row);torch.manual_seed(202610175413);m=api.build(c,'cpu');state={k:v.detach().clone() for k,v in m.state_dict().items()}
  if initial is None:initial=state
  else:assert set(initial)==set(state) and all(torch.equal(initial[k],v) for k,v in state.items())
  replica=deepcopy(m).eval();assert replica._clock is None or replica._clock is not m._clock;assert replica._full_channel is None or replica._full_channel is not m._full_channel;del replica
  m.train();m._feature_l_steps=m._full_calls=m._legacy_calls=1110
  torch.manual_seed(202610175414);raw=torch.randn(128,2,256);ordinary=raw*.9;labels=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
  meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
  seen=[];original=m.id_backbone.encoder._full_features
  def traced(x):seen.append(x.detach().clone());return original(x)
  m.id_backbone.encoder._full_features=traced
  opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);W=m.get_parameter('id_backbone.encoder.core.id_backbone.cls_head.weight');before=W.detach().clone();rng=torch.get_rng_state()
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
  corr,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
  total=ce+corr;assert len(seen)==4 and torch.equal(seen[0],seen[1]) and torch.equal(seen[2],seen[3])
  gm=(torch.arange(128)+1111)%2==0;sm=~gm
  assert torch.equal(seen[2][gm],ordinary[gm])==(not c['new_Ag'])
  assert torch.equal(seen[2][sm],ordinary[sm])==(not c['As'])
  rec=m._full_records[0];assert rec['Ag_packets']==(64 if c['new_Ag'] else 0) and rec['As_packets']==(64 if c['As'] else 0)
  expected=.5*(rec['common_RD']+rec['extra_RD'])+rec['SupCon_weight']*rec['SupCon'];assert abs(float(total.detach())-expected)<2e-6
  assert len(rec['SupCon_details'])==4 and m._supcon_pending is None and not m._supcon_handles
  total.backward();assert W.grad is not None and torch.isfinite(W.grad).all()
  if row['arm']=='B00':
   ref=old.build(d.reference_config(),'cpu');ref.load_state_dict(state);ref.train();ref._feature_l_steps=ref._full_calls=ref._legacy_calls=1110;torch.set_rng_state(rng)
   with ref._phase1_labeled_feature_context():ro=ref(ordinary,return_aux=True)
   rce=torch.nn.functional.cross_entropy(ro['tx_logits'],labels,label_smoothing=.01)
   rr,_=ref._source_labeled_feature_hook(model=ref,labeled_output=ro,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=rce,labeled_count=128,labeled_inputs=raw)
   rt=rce+rr;rt.backward();assert torch.equal(total.detach(),rt.detach())
   for (name,p),(rn,rp) in zip(m.named_parameters(),ref.named_parameters()):
    assert name==rn and ((p.grad is None and rp.grad is None) or (p.grad is not None and rp.grad is not None and torch.equal(p.grad,rp.grad)))
   for k,v in m.state_dict().items():assert torch.equal(v,ref.state_dict()[k])
   NN+=512;del ref
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,W)
  m.id_backbone.encoder._full_features=original;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(raw[:2]);ef=api.export(m).eval()(raw[:2]);assert f.shape==(2,160) and torch.equal(f,ef)
  NN+=516;physics+=64 if c['As'] else 0;clock+=64 if c['new_Ag'] else 0
  reports.append(dict(arm=row['arm'],parameters=sum(p.numel() for p in m.parameters()),L_forwards=4,Ag_packets=rec['Ag_packets'],As_packets=rec['As_packets'],default_head_joint_update=True,export160_exact=True,same_IQ_pairs=True,B00_equivalent_to_existing_REP=row['arm']=='B00'))
 from .source import RecordingLoader
 md={k:v[:5] for k,v in meta.items()}
 assert len(list(RecordingLoader([(raw[:5],torch.full((5,),-1),domains[:5],md)],'U',io.StringIO())))==1
 try:list(RecordingLoader([(raw[:5],labels[:5],domains[:5],md)],'U',io.StringIO()))
 except ValueError:negative+=1
 else:raise AssertionError('U truth accepted')
 from . import evaluate as ev,proof as pv
 original_read,original_ids,original_source=d.read,ev.ids,pv.source
 try:
  pv.source=lambda c:None;ev.ids=lambda:np.arange(ev.COUNT)
  valid=dict(status='PREDICTIONS_COMPLETE',config=d.config(d.train_rows()[0]),count=ev.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,bad in (('status','PENDING'),('count',5),('views',['clean']),('truth_read',True),('query_fit',True),('head_fit_updates',1)):
   value=dict(valid);value[key]=bad
   def fake_read(path):
    if Path(path).name=='complete.json' and Path(path).parent.name=='prediction':return value
    raise AssertionError('Truth opened before prediction closure')
   d.read=fake_read
   try:ev.score(d.train_rows()[0]['row_id'])
   except ValueError:negative+=1
   else:raise AssertionError('Invalid prediction accepted')
 finally:d.read,ev.ids,pv.source=original_read,original_ids,original_source
 return dict(status='PASS',negative_cases=negative,rows=reports,synthetic_NN=NN,synthetic_physics_packets=physics,synthetic_clock_packets=clock+4,synthetic_optimizer_steps=4,real_IQ=0,CP=0,target_access=False)

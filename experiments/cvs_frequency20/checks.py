"""Actual six-forward F objectives, unchanged control and protocol negatives."""
import copy,math,subprocess,sys
from pathlib import Path
from types import MethodType
import torch
def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from .frequency import apply
 from experiments.cvs_crossrx_satmix20.model import hook as old_hook
 from experiments.cvs_crossrx_satmix20 import design as old_design,supcon
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0
 direct=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_frequency20/runner.py'),'--unregistered'],capture_output=True,text=True,timeout=10)
 assert direct.returncode!=0 and 'No frequency training overrides allowed' in direct.stderr
 for key,value in (('epochs',50),('model_initialization','pretrained'),('test_views',['clean']),('model_seed',2026092702)):
  c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered budget/data/seed accepted')
 for key,value in (('As',True),('raw_weight',.25),('U_rule','changed')):
  c=copy.deepcopy(d.config(d.rows()[1]));c['training_mechanism'][key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Scientific boundary changed')
 for key,value in (('max_cycles_per_sample',d.FREQUENCY_MAX*2),('NoExplicitCFO',False)):
  c=copy.deepcopy(d.config(d.rows()[1]));c['frequency'][key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Fixed frequency or identity input policy changed')
 raw=torch.randn(128,2,256);before=raw.clone();rng=torch.get_rng_state();x,angles,error=apply(raw,1111)
 assert torch.equal(rng,torch.get_rng_state()) and torch.equal(raw,before) and max(angles)<=d.FREQUENCY_MAX and min(angles)>=-d.FREQUENCY_MAX and error<=1e-6
 phase=2*math.pi*torch.tensor(angles)[:,None]*torch.arange(256);canonical=torch.complex(raw[:,0],raw[:,1])*torch.complex(phase.cos(),phase.sin())
 assert torch.equal(x[:,:,0],raw[:,:,0]) and torch.allclose(x,torch.stack((canonical.real,canonical.imag),1),rtol=0,atol=1e-6)
 again,a2,_=apply(raw,1111);different,a3,_=apply(raw,1112)
 assert torch.equal(x,again) and angles==a2 and not torch.equal(x,different) and angles!=a3
 for value,step in ((raw,0),(raw,4441),(raw[:,:,:128],1),(raw.fill_(float('nan')),1)):
  try:apply(value,step)
  except ValueError:negative+=1
  else:raise AssertionError('Bad frequency input/step accepted')
 reports=[];initials=[]
 for row in d.rows():
  torch.manual_seed(20261016209);c=d.config(row);m=api.build(c,'cpu');m.train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()})
  m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  torch.manual_seed(20261016210);raw=torch.randn(128,2,256);ordinary=raw.roll(2,-1)*.9;labels=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
  meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
  xx,yy,extra=native.move_batch((raw,labels,domains,meta),torch.device('cpu'));assert torch.equal(xx,raw) and torch.equal(yy,labels) and torch.equal(extra[1]['rx_i'],rx)
  args=d.make_args(c,'cpu');assert args.use_aug and isinstance(native.build_aug_base_cfg(args),dict) and not args.baseline_ckpt and not args.teacher_ckpt
  seen=[];original=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return original(v)
  def objective(model):
   torch.manual_seed(20261016211)
   with model._phase1_labeled_feature_context():out=model(ordinary,return_aux=True)
   ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
   correction,_=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
   return ce+correction
  legacy=None
  if row['arm']=='B00':
   legacy=copy.deepcopy(m);legacy._source_labeled_feature_hook=MethodType(old_hook,legacy);legacy._joint_config=old_design.config(old_design.rows()[0]);legacy_loss=objective(legacy);legacy_loss.backward()
  m.id_backbone.encoder._view_features=trace
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';start=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m)
  assert len(seen)==6 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and torch.equal(seen[2],raw) and torch.equal(seen[2],seen[3]) and torch.equal(seen[4],seen[5]) and (torch.equal(seen[4],raw)==(row['arm']=='B00'))
  record=m._joint_details[0];expected=128 if row['arm']=='B10' else 0
  assert record['frequency_packets']==m._frequency_packets==expected and len(record['frequency_values'])==expected and record['As_packets']==m._joint_As_packets==0 and not m._joint_coverage
  assert m._supcon_applied==1 and len(record['supcon_details'])==4 and not m._supcon_handles and m._supcon_pending is None and math.isclose(record['supcon_weight'],.01,abs_tol=1e-15)
  total.backward();grad=m.get_parameter(W).grad
  assert grad is not None and torch.isfinite(grad).all()
  if legacy is not None:
   assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(grad,legacy.get_parameter(W).grad)
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(start,m.get_parameter(W))
  m.id_backbone.encoder._view_features=original;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   features=m.id_backbone.encoder.features(raw[:2]).clone();export=api.export(m).eval();assert features.shape==(2,160) and torch.equal(features,export(raw[:2]))
  reports.append(dict(arm=row['arm'],L_forwards=6,same_IQ_pairs=True,raw_weight=.5,ordinary_weight=.25,slot_weight=.25,frequency_packets=expected,As_packets=0,supcon_terms=4,frequency_and_raw_have_no_supcon=True,private_frequency_generator_preserves_native_RNG=True,default_head_joint_update=True,export160_exact=True,total=float(total.detach())))
 assert all(torch.equal(v,initials[1][k]) for k,v in initials[0].items())
 a,b=[d.make_args(d.config(r),'cpu') for r in d.rows()]
 differences={k for k,v in vars(a).items() if v!=vars(b).get(k) and not (isinstance(v,float) and isinstance(vars(b).get(k),float) and math.isnan(v) and math.isnan(vars(b)[k]))}
 assert differences=={'candidate_id','output_dir'}
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_scratch_initialization=True,control_loss_and_head_gradient_exactly_match_previous_B00=True,frequency_same_step_reproducible=True,frequency_different_step_differs=True,source_IQ_reads=0,target_access=False,synthetic_optimizer_steps=2,synthetic_L_objectives=3,synthetic_NN_packets=2312,direct_script_startup_regression_PASS=True,canonical_native_CFO_operator_matches=True,head_fit_updates=0)
if __name__=='__main__':
 import json
 value=run()
 from . import design as d
 d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print('LOCAL_RESULT '+json.dumps({k:v for k,v in value.items() if k!='rows'}))

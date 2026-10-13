import json,copy
import torch
from . import design as d
from .model import build,from_phase1,install
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply

def checks():
 torch.set_num_threads(2);torch.manual_seed(54017)
 configs=[d.config(r) for r in d.training_rows()[:2]]
 models=[build(c).train() for c in configs]
 if any(c['model_seed']!=2026092701 for c in configs):raise ValueError('Matched seed test drift')
 p,q=[m.id_backbone.encoder for m in models]
 for key,value in p.state_dict().items():
  if not torch.equal(value,q.state_dict()[key]):raise ValueError('Matched power/complex parameter initialization differs: '+key)
 if sum(v.numel() for v in p.spectral.parameters())!=45344:raise ValueError('Relation capacity differs')
 x=torch.randn(6,2,256);y=torch.arange(6);report=[]
 for c,m in zip(configs,models):
  e=m.id_backbone.encoder;head=e.core.id_backbone.cls_head
  m.eval()
  with torch.no_grad():z=e.features(x);f=from_phase1(m);value=f(x)
  if not torch.equal(value,z):raise ValueError('Export changed actual joint160')
  if sum(v.numel() for v in f.parameters())!=221569:raise ValueError('Feature parameter count differs')
  if any(n.endswith('cls_head.weight') for n,_ in f.named_parameters()):raise ValueError('Old class rows retained')
  changed=copy.deepcopy(m);changed.id_backbone.encoder.core.id_backbone.cls_head.weight.data.normal_()
  if not torch.equal(from_phase1(changed)(x),value):raise ValueError('Export depends on old classifier')
  clone=copy.deepcopy(e);clone.spectral_gain.data.zero_()
  with torch.no_grad():base=e.core.features(x)+e.response_gain.tanh()*e.response_projection(e.response(x))
  if not torch.equal(clone.features(x),base):raise ValueError('Disabled residual changed baseline features')
  xx=x.clone().requires_grad_(True);m.train();m.zero_grad(set_to_none=True)
  logits=m.forward_identity_only(xx)['tx_logits'];loss=torch.nn.functional.cross_entropy(logits,y);loss.backward()
  spectral=[v.grad for v in e.spectral.parameters()]
  if any(g is None or not torch.isfinite(g).all() for g in spectral):raise ValueError('Relation gradients invalid')
  gn=sum(float(g.square().sum()) for g in spectral)**.5
  if gn<=0 or head.weight.grad is None or e.spectral_gain.grad is None:raise ValueError('Real CE bypassed feature/classifier path')
  if not torch.isfinite(xx.grad).all():raise ValueError('Input gradient invalid')
  m.eval()
  with torch.no_grad():one=f(x[:1]);batch=f(x[:4]);zero=f(torch.zeros(2,2,256))
  if not torch.allclose(one,batch[:1],atol=2e-5,rtol=2e-5) or not torch.isfinite(zero).all():raise ValueError('Packet-independent export differs')
  report.append(dict(arm=c['arm'],classifier_type=type(head).__name__,classifier_shape=list(head.weight.shape),feature_parameters=221569,extra_feature_parameters=45504,feature_export_max_difference=0,old_W_discarded=True,CE_relation_gradient_l2=gn,feature_gain_gradient_l2=float(e.spectral_gain.grad.norm()),classifier_joint_CE_retained=True,zero_input_finite=True,per_packet_independent=True))
 negatives=0
 for c in [dict(configs[0],epochs=51),dict(configs[0],epochs=20),dict(configs[0],source_episode={}),dict(configs[0],checkpoint_sources=['unknown']),dict(configs[0],target_access=True)]:
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered budget/provenance/role accepted')
 return dict(status='PASS',rows=report,matched_initialization=True,zero_gain_exact_baseline=True,negative_config_cases=negatives,real_IQ_or_target_read=False)
if __name__=='__main__':print(json.dumps(checks()))

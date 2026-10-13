import copy,torch
from .trace import objects,capture,derivatives,parameter_record,group,W
from . import design as d

def smoke_model(model):
 model.eval().requires_grad_(True);saved={k:v.clone() for k,v in model.state_dict().items()};params,groups=objects(model)
 out,activations=capture(model,torch.zeros(2,2,256));loss=torch.nn.functional.cross_entropy(out['tx_logits'],torch.tensor([0,1]),label_smoothing=.01)
 pg,ag=derivatives(loss,params,activations)
 if not torch.isfinite(loss) or out['tx_logits'].shape!=(2,6) or set(ag)!=set(activations):raise ValueError('Actual no-query derivative smoke failed')
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Smoke mutated model or accumulated gradients')
 return dict(status='PASS',zero_IQ=True,query_read=False,class_W_gradient_object=False,feature_parameters=176065,real_source_IQ_read=False,head_fit=False,optimizer_steps=0)

def checks():
 from experiments.cvs_neighbor_pair50.model import build
 from experiments.cvs_source_gate_audit.gates import physical_metadata
 torch.set_num_threads(2);torch.manual_seed(d.SEED);model=build(d.source_config(d.rows()[1])).eval().requires_grad_(True)
 negatives=0
 for name in (W,'dom_backbone.weight','id_backbone.encoder.core.id_backbone.unregistered.weight'):
  try:group(name)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid gradient object accepted')
 try:objects(model.train())
 except ValueError:negatives+=1
 else:raise AssertionError('Train-mode feature observer accepted')
 model.eval()
 for role in ('U_s','target'):
  try:physical_metadata({},role)
  except ValueError:negatives+=1
  else:raise AssertionError('Unauthorized gradient-source role accepted')
 result=smoke_model(model);params,groups=objects(model);fake=[torch.ones_like(p) for n,p in params];data=parameter_record(groups,fake,[v*.5 for v in fake],[v*.25 for v in fake])
 assert sum(v['elements'] for v in data)==176065 and all(abs(v['combined_U_over_L']-.75)<1e-12 and abs(v['L_U_cosine']-1)<1e-12 for v in data)
 return dict(status='PASS',synthetic_only=True,protocol_negative_checks=negatives,feature_object_boundary=True,no_accumulated_gradients_or_state_updates=True,source_or_query_read=False,smoke=result)

if __name__=='__main__':
 import json
 print(json.dumps(checks()))

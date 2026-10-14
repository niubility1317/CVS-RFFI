import json,torch
from . import design as d
from .objective import capture,views,descriptors,relation_indices,losses
from experiments.cvs_source_layer_order.checks import checks as base_checks

def checks():
 torch.set_num_threads(2);base=base_checks();s=torch.zeros(90,4,28);s[:,:,6]=1;s[:,:,0]=torch.linspace(-1,1,90)[:,None];dynamic,clock=descriptors(s.flatten(1));assert dynamic.shape==(90,108) and torch.equal(clock,torch.ones(90,4)) and torch.equal(dynamic.reshape(90,4,27)[:,:,0],s[:,:,0]);negatives=0
 for bad in (s.flatten(1)[:,:111],s.flatten(1)*float('nan'),s.flatten(1)*2):
  try:descriptors(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid constant-time descriptor accepted')
 from experiments.cvs_local_order50 import model as pm,design as pd
 m=pm.build(d.source_config(),'cpu',design=pd).eval();state={k:v.clone() for k,v in m.state_dict().items()};y=torch.tensor([t for t in range(6) for dd in d.DAYS for r in d.RX]);rx=torch.tensor([r for t in range(6) for dd in d.DAYS for r in d.RX]);day=torch.tensor([dd for t in range(6) for dd in d.DAYS for r in d.RX]);q,groups=relation_indices(y,rx,day,list(range(90)));rng=torch.get_rng_state().clone()
 with capture(m) as maps:out=m.forward_identity_only(torch.randn(90,2,256,generator=torch.Generator().manual_seed(1911)))
 z=views(maps,out);params=[dict(m.named_parameters())[n] for n in d.PARAMETERS];head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;records=[]
 for name in d.LEVELS:
  loss,sc,values=losses(z[name],q,groups);gs=torch.autograd.grad(loss,params+[head],allow_unused=True,retain_graph=True);assert gs[-1] is None and all(g is None or torch.isfinite(g).all() for g in gs)
  if name!='T2_WINDOW':assert gs[2] is not None and gs[2].norm()>0
  else:assert gs[2] is None
  records.append(dict(view=name,shape=list(z[name].shape),gradient_norms=[float(g.norm()) if g is not None else 0 for g in gs[:-1]],class_W_dependency=False))
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()) and torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in m.parameters())
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec');assert 'experiments/cvs_local_order50' in t.PATHS
 return dict(status='PASS',protocol_negative_tests=base['protocol_negative_tests']+negatives,actual_native_views=records,clock_coordinates_removed=4,network_and_deployed_features_modified=False,state_RNG_grad_unchanged=True,optimizer_updates=0,head_fit_updates=0,query_read=False,synthetic_IQ_only=True)
if __name__=='__main__':print(json.dumps(checks()))

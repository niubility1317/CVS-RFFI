import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import design as d
from .operators import coarsen,descriptor,identities,from_coordinates
from .probe import Probe
from .metrics import score,decision
from experiments.cvs_F_nontrue20 import model as base
from experiments.cvs_reference_identity.model import ReferenceCVS
def run():
 torch.set_num_threads(2);torch.manual_seed(20261019941);u=torch.randn(8,8,6);e=coarsen(u,'E');i=coarsen(u,'I');both=coarsen(u,'EI');ids=identities(u)
 assert max(ids.values())<2e-6 and torch.equal(e[:,2:6],u[:,2:6]) and torch.equal(i[:,:2],u[:,:2]) and torch.equal(i[:,6:],u[:,6:]) and torch.equal(e[:,:2],u[:,:2].mean(1,keepdim=True).expand(-1,2,-1)) and torch.equal(e[:,6:],u[:,6:].mean(1,keepdim=True).expand(-1,2,-1))
 assert torch.equal(both,coarsen(e,'I')) and torch.allclose(descriptor(i,'NATIVE')[:,:6],descriptor(u,'NATIVE')[:,:6],atol=2e-6) and torch.equal(coarsen(u,'NATIVE'),u)
 for mode in d.MODES:
  assert torch.equal(coarsen(u,mode),torch.cat([coarsen(v[None],mode) for v in u])) and torch.equal(coarsen(torch.ones_like(u),mode),torch.ones_like(u))
 m=base.build(d.joint.parent_config(),'cpu').eval().requires_grad_(False);state={k:v.clone() for k,v in m.state_dict().items()};x=torch.randn(2,2,256);rng=torch.get_rng_state().clone();original=ReferenceCVS.features;NN=0
 with torch.no_grad():
  actual=m.forward_identity_only(x);NN+=2
  with Probe(m) as p:
   replay=m.forward_identity_only(x);NN+=2;r=p.take(replay)
  assert p.closed and ReferenceCVS.features is original and torch.equal(actual['tx_logits'],replay['tx_logits']) and torch.equal(actual['id_feat_joint'],replay['id_feat_joint'])
  pool=m.id_backbone.encoder.sequence_pool
  for mode in d.MODES[1:]:
   def replace(module,args,out):
    coords=[torch.tanh(view.local_projection(args[0])).transpose(1,2) for view in module.views];return from_coordinates(module,coords,mode)
   h=pool.register_forward_hook(replace)
   try:v=m.forward_identity_only(x);NN+=2
   finally:h.remove()
   assert torch.equal(v['tx_logits'],r['logits'][mode])
 assert NN==10 and torch.equal(torch.get_rng_state(),rng) and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
 y=np.tile(np.arange(6),3);rx=np.repeat([1,3,4],6);day=np.ones(18,int);logits=np.eye(6)[y]*4;records=score(logits,y,rx,day);assert all(v['accuracy']==1 and v['macro_f1']==(1 if v['dimension']!='transmitter' else 1/6) for v in records)
 def sample(n,e,i,raw):return dict(views={v:dict(conditions={mode:[dict(accuracy=(raw if v=='raw' else e) if mode=='E' else n if mode=='NATIVE' else i)] for mode in d.MODES}) for v in d.VIEWS})
 assert decision({'FULL_As_JOINT_FINAL30':sample(.7,.72,.71,.7)})['supports_boundary_hypothesis']
 assert not decision({'FULL_As_JOINT_FINAL30':sample(.7,.704,.71,.7)})['supports_boundary_hypothesis']
 assert not decision({'FULL_As_JOINT_FINAL30':sample(.7,.72,.71,.697)})['supports_boundary_hypothesis']
 negative=0
 for fn in (lambda:coarsen(u,'OTHER'),lambda:coarsen(u[:,:7],'E'),lambda:coarsen(u*float('nan'),'E'),lambda:d.validate(dict(d.config(),V_used=True)),lambda:d.validate(dict(d.config(),target_access=True))):
  try:fn()
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered operator/role accepted')
 return dict(status='PASS',synthetic_whole_model_NN=NN,real_checkpoint=0,real_IQ=0,fit_updates=0,target_access=False,negative_tests=negative,E_I_matched_regions=True,no_cross_packet_mix=True,endpoint_and_timearea_identities=True,native_base_pool_joint_logits_bit_equal=True,alternative_direct_whole_forward_bit_equal=True,model_and_RNG_unchanged=True,method_and_hooks_restored=True,source_metrics_and_fixed_decision=True)
if __name__=='__main__':print(json.dumps(run()))

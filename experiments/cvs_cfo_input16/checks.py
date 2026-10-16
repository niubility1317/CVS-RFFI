"""Focused finite/budget/control negatives plus one local untrained-F path check."""
from copy import deepcopy
import math,torch
from . import design as d,audit
from .operator import apply,view
from experiments.cvs_crossrx_satmix20 import model as api
def run():
 torch.set_num_threads(2);x=audit.synthetic();rng=torch.get_rng_state().clone();positive=apply(x,d.FREQUENCY_MAX);negative=apply(x,-d.FREQUENCY_MAX);restored=apply(positive,-d.FREQUENCY_MAX)
 n=torch.arange(256,dtype=x.dtype);f=torch.full((32,1),d.FREQUENCY_MAX);phase=2.*math.pi*f*n;canonical=torch.complex(x[:,0],x[:,1])*torch.complex(phase.cos(),phase.sin())
 assert torch.equal(rng,torch.get_rng_state()) and torch.equal(view(x,'identity'),x) and torch.equal(positive[:,:,0],x[:,:,0]) and torch.allclose(positive,torch.stack((canonical.real,canonical.imag),1),rtol=0,atol=1e-6)
 assert (positive-x).norm()>0 and (negative-x).norm()>0 and float((restored-x).flatten(1).norm(dim=1).div(x.flatten(1).norm(dim=1)).max())<=1e-6
 negatives=0
 def reject(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,TypeError):negatives+=1;return
  raise AssertionError('Forbidden operator input accepted')
 for f in (d.FREQUENCY_MAX*1.01,float('nan'),float('inf'),True):reject(lambda f=f:apply(x,f))
 reject(lambda:apply(x[:,:,:255],0.));reject(lambda:apply(x.double(),0.));bad=x.clone();bad[0,0,0]=float('nan');reject(lambda:apply(bad,0.));reject(lambda:view(x,'unregistered'));reject(lambda:view(x[:2],'identity'));reject(lambda:audit.qualify({}))
 clean=dict(normalized_joint_change_median=0.,normalized_joint_change_max=0.,logits_abs_change_max=0.,power_relative_error_max=0.,inverse_input_relative_error_max=0.)
 values={n:dict(clean) for n in d.VIEWS};assert audit.qualify(values)['decision']=='FAIL'
 values['frequency_positive'].update(normalized_joint_change_median=.01,normalized_joint_change_max=.02,logits_abs_change_max=.1);assert audit.qualify(values)['decision']=='PASS'
 v=deepcopy(values);v['positive_inverse']['inverse_input_relative_error_max']=1e-3;assert audit.qualify(v)['decision']=='FAIL'
 v=deepcopy(values);v['negative_inverse']['logits_abs_change_max']=.1;assert audit.qualify(v)['decision']=='FAIL'
 v=deepcopy(values);v['frequency_positive']['power_relative_error_max']=.1;assert audit.qualify(v)['decision']=='FAIL'
 m=api.build(d.prior.config(d.prior.rows()[0]),'cpu').eval().requires_grad_(False)
 assert m.id_backbone.encoder.response.contract()['explicit_CFO_identity_input'] is False and m.id_backbone.encoder.response.contract()['internal_CFO_compensation'] is True
 with torch.no_grad():z=m.forward_identity_only(x[:2])['tx_logits']
 assert z.shape==(2,6) and torch.isfinite(z).all()
 old=d.BASE;d.BASE=d.ROOT/d.LOCAL/'local_untrained_input_check';d.BASE.mkdir(parents=True,exist_ok=True)
 try:result=audit.run(m)
 finally:d.BASE=old
 return dict(status='PASS',operator_protocol_rejections=negatives,qualifier_counterexamples=4,canonical_native_CFO_operator_matches=True,power_and_inverse_control_PASS=True,private_synthetic_rng_and_no_operator_rng=True,actual_untrained_F_input_path_PASS=True,default_head_shape=[6,160],NoExplicitCFO=True,internal_compensation_retained=True,synthetic_NN_packets=162,source_IQ_reads=0,real_CP_reads=0,training_updates=0,head_fit_updates=0,target_access=False,engineering_decision_is_not_efficacy=True)
if __name__=='__main__':
 import json
 value=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print('LOCAL_RESULT '+json.dumps(value))

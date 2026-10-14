import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .metrics import observe_gradients,vector_stats,summary

def checks():
 torch.set_num_threads(2);torch.manual_seed(2741)
 # Analytic source-error decomposition on the same graph and accepted denominator.
 p=torch.randn(5,6,requires_grad=True);q=torch.randn(4,5);logits=q@p
 pseudo=torch.tensor([0,1,2,3]);truth=torch.tensor([0,2,2,4]);keep=torch.tensor([True,True,False,True])
 result=observe_gradients(logits,pseudo,truth,keep,[p],'L_s');assert result['accepted']==3 and result['accepted_wrong']==2
 expected=torch.autograd.grad(.16*F.cross_entropy(logits,truth,reduction='none')[keep].mean(),p,retain_graph=True)[0].flatten()
 assert torch.allclose(expected,result['vectors']['matched_true'][0]) and torch.allclose(result['vectors']['total'][0],result['vectors']['correct'][0]+result['vectors']['wrong'][0],atol=1e-6)
 empty=observe_gradients(logits,pseudo,truth,~torch.ones_like(keep),[p],'L_s');assert empty['accepted']==0 and all(not v.any() for values in empty['vectors'].values() for v in values)
 exact=observe_gradients(logits,pseudo,pseudo,keep,[p],'L_s');assert torch.equal(exact['vectors']['total'][0],exact['vectors']['matched_true'][0]) and not exact['vectors']['wrong'][0].any()
 negative=0
 for role in ('V','U_s'):
  try:observe_gradients(logits,pseudo,truth,keep,[p],role)
  except ValueError:negative+=1
  else:raise AssertionError('Forbidden role gradient accepted')
 for change in [dict(roles=['U_s']),dict(query_read=True),dict(head_updates=1),dict(optimizer_steps=1),dict(gradient_role='V')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered source action accepted')
 stats=summary(dict(cos=torch.tensor([.2,.4,.8])),torch.tensor([True,True,True]),torch.ones(3,dtype=torch.bool));assert stats['wrong']==0 and stats['groups']['wrong']['cos']['mean'] is None
 assert vector_stats(torch.tensor([1.]),torch.tensor([1.]),torch.tensor([2.]))['wrong_over_total_norm']==2.
 from experiments.cvs_sigtime_pair50.model import build
 c=d.source_config(d.rows()[0]);model=build(c,'cpu').train().requires_grad_(False)
 params=[dict(model.named_parameters())[name] for name in d.PARAMETERS]
 for value in params:value.requires_grad_(True)
 before={key:value.clone() for key,value in model.state_dict().items()};x=torch.randn(4,2,256);a=model.forward_identity_only(x)
 assert a['id_feat_joint'].shape==(4,160) and a['tx_logits'].shape==(4,6)
 r=observe_gradients(a['tx_logits'],pseudo,truth,keep,params,'L_s');assert all(v.norm()>0 for v in r['vectors']['total'])
 assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items()) and all(value.grad is None for value in model.parameters())
 return dict(status='PASS',synthetic_only=True,no_real_IQ_or_checkpoint_or_truth_read=True,negative_checks=negative,common_denominator_decomposition=True,empty_acceptance_safe=True,actual_SIG_TIME_joint160_and_three_parameter_gradients=True,unchanged_parameters_buffers=True,no_accumulated_grad=True,no_head_fit=True)
if __name__=='__main__':print(json.dumps(checks()))

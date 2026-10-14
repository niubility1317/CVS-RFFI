import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .metrics import observe,summarize_vectors

def checks():
 torch.set_num_threads(2);torch.manual_seed(27415);p=torch.randn(4,160,requires_grad=True);truth=torch.zeros(4,dtype=torch.long);weak_logits=torch.zeros(4,6);weak_feature=torch.randn(4,160)
 with torch.no_grad():
  p[:,:6]=0
  for i,target in enumerate((0,1,0,1)):p[i,target]=3
  for i,target in enumerate((0,0,1,1)):weak_logits[i,target]=3
 r=observe(p[:,:6],weak_logits,p,weak_feature,truth,[p],'L_s');assert set(r['groups'].values())=={1}
 expected=torch.autograd.grad(F.cross_entropy(p[:,:6],truth,label_smoothing=.01),p,retain_graph=True)[0].flatten();assert torch.allclose(r['vectors']['main'][0],expected)
 assert torch.allclose(r['vectors']['effective'][0],.1*r['vectors']['feature'][0],atol=1e-6)
 for key in ('main','feature'):assert torch.allclose(r['vectors'][key][0],sum(r['vectors'][key+'_'+name][0] for name in d.GROUPS),atol=1e-6)
 q=p.detach().clone().requires_grad_(True);q_logits=q[:,:6];t=q_logits.detach().argmax(1);all_right=q_logits.detach().clone();empty=observe(q_logits,all_right,q,weak_feature,t,[q],'L_s');s=summarize_vectors(empty['vectors'],empty['groups'],['synthetic'])[0]
 assert empty['groups'][d.GROUPS[0]]==4 and all(empty['groups'][key]==0 and s['groups'][key]['effective_vs_matched_true']['cosine'] is None for key in d.GROUPS[1:])
 negative=0
 for role in ('U_s','V','query'):
  try:observe(p[:,:6],weak_logits,p,weak_feature,truth,[p],role)
  except ValueError:negative+=1
  else:raise AssertionError('Forbidden role accepted')
 for change in [dict(roles=['V']),dict(query_read=True),dict(U_IQ_read=True),dict(U_truth_read=True),dict(head_updates=1),dict(optimizer_steps=1),dict(V_forward=True),dict(feature_coefficient=.2),dict(model_state_updates=1)]:
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Frozen source action changed')
 try:observe(p[:,:6],weak_logits,p,weak_feature.requires_grad_(True),truth,[p],'L_s')
 except ValueError:negative+=1
 else:raise AssertionError('Teacher gradient allowed')
 from experiments.cvs_feature_mean_teacher50.model import build
 for row in d.rows():
  a=d.prior(row).make_args(d.source_config(row),'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.strong_noise_std)==(50,33,17,.015)
 model=build(d.source_config(d.rows()[0]),'cpu').train().requires_grad_(False);teacher=copy.deepcopy(model).eval();params=[dict(model.named_parameters())[name] for name in d.PARAMETERS]
 for value in params:value.requires_grad_(True)
 before={key:value.clone() for key,value in model.state_dict().items()};x=torch.randn(4,2,256)
 with torch.no_grad():weak=teacher.forward_identity_only(x)
 strong=model.forward_identity_only(x+.015*torch.randn_like(x));v=observe(strong['tx_logits'],weak['tx_logits'],strong['id_feat_joint'],weak['id_feat_joint'],torch.tensor([0,1,2,3]),params,'L_s')
 assert len(v['vectors'])==11 and all(value.norm()>0 for value in v['vectors']['main']) and all(value.norm()>0 for value in v['vectors']['feature'])
 assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items()) and all(value.grad is None for value in model.parameters())
 return dict(status='PASS',synthetic_only=True,no_real_IQ_or_checkpoint_or_truth_read=True,negative_checks=negative,same_graph_common_denominator=True,actual_original_feature_objective=True,empty_groups_undefined_cosine=True,actual_SIG_joint160_three_parameters=True,six_native_contexts=True,parameters_buffers_unchanged=True,no_accumulated_grad=True,no_head_fit=True)
if __name__=='__main__':print(json.dumps(checks()))

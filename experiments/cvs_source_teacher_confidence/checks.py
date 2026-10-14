import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .metrics import policy,observe

def checks():
 from experiments.cvs_source_feature_ce_gradient.checks import checks as previous
 parent=previous();torch.set_num_threads(2);torch.manual_seed(27451);p=torch.randn(4,160,requires_grad=True);truth=torch.zeros(4,dtype=torch.long);weak_logits=torch.zeros(4,6);weak_feature=torch.randn(4,160)
 with torch.no_grad():
  p[:,:6]=0
  for i,target in enumerate((0,1,0,1)):p[i,target]=3
  for i,(target,value) in enumerate(zip((0,0,1,1),(2,5,2,3))):weak_logits[i,target]=value
 rule=policy(weak_logits,p[:,:6]);assert rule['keep'].tolist()==[False,True,False,True] and not rule['keep'].requires_grad
 r=observe(p[:,:6],weak_logits,p,weak_feature,truth,[p],'L_s');per=.5*(F.normalize(p,dim=1,eps=1e-8)-F.normalize(weak_feature,dim=1,eps=1e-8)).square().sum(1)
 expected=torch.autograd.grad(.1*(per*rule['keep']).sum()/4,p,retain_graph=True)[0].flatten();assert torch.allclose(r['vectors']['gated'][0],expected,atol=1e-6)
 assert len(r['vectors'])==16 and sum(r['retained_groups'].values())==2 and torch.allclose(r['vectors']['gated'][0],sum(r['vectors']['gated_'+key][0] for key in d.GROUPS),atol=1e-6)
 all_logits=torch.zeros(4,6);all_logits[:,0]=30
 all_kept=observe(p[:,:6],all_logits,p,weak_feature,truth,[p],'L_s');assert all_kept['policy']['retained']==4 and torch.allclose(all_kept['vectors']['gated'][0],all_kept['vectors']['effective'][0],atol=1e-6)
 empty=observe(p[:,:6],torch.zeros(4,6),p,weak_feature,truth,[p],'L_s');assert empty['policy']['retained']==0 and all(not x.any() for x in empty['vectors']['gated'])
 negative=0
 for role in ('U_s','V','query'):
  try:observe(p[:,:6],weak_logits,p,weak_feature,truth,[p],role)
  except ValueError:negative+=1
  else:raise AssertionError('Forbidden gradient role accepted')
 for change in [dict(gap_threshold=.01),dict(V_gradient=True),dict(query_read=True),dict(U_truth_read=True),dict(head_updates=1),dict(optimizer_steps=1),dict(loss_denominator='selected count')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Fixed source rule changed')
 from .audit import qualification
 q=[dict(row_id=r['row_id'],groups={'WEAK_RIGHT_STRONG_WRONG':dict(count=10,retained=8,retention=.8),'WEAK_WRONG_STRONG_RIGHT':dict(count=10,retained=2,retention=.2)}) for r in d.rows()];assert qualification(q)['status']=='PASS';q[-1]['groups']['WEAK_WRONG_STRONG_RIGHT']['retention']=.9;assert qualification(q)['status']=='FAIL'
 from experiments.cvs_feature_mean_teacher50.model import build
 model=build(d.source_config(d.rows()[0]),'cpu').train().requires_grad_(False);teacher=copy.deepcopy(model).eval();params=[dict(model.named_parameters())[name] for name in d.PARAMETERS]
 for x in params:x.requires_grad_(True)
 before={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(4,2,256)
 with torch.no_grad():weak=teacher.forward_identity_only(x)
 strong=model.forward_identity_only(x+.015*torch.randn_like(x));v=observe(strong['tx_logits'],weak['tx_logits'],strong['id_feat_joint'],weak['id_feat_joint'],torch.tensor([0,1,2,3]),params,'L_s');assert len(v['vectors'])==16 and all(x.shape==(size,) for x,size in zip(v['vectors']['main'],(3456,6912,576)))
 with torch.no_grad():val=model.forward_identity_only(x)
 assert not val['tx_logits'].requires_grad and not val['id_feat_joint'].requires_grad and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in model.parameters())
 return dict(status='PASS',synthetic_only=True,no_real_IQ_or_checkpoint_or_truth_read=True,parent_checks=parent['status'],new_negative_checks=negative,FP32_confidence_ties_kept=True,full_batch_denominator=True,empty_and_all_retained_safe=True,actual_SIG_joint160_sixteen_derivative_sets=True,V_no_grad=True,all_six_qualification_required=True,no_parameter_buffer_or_head_update=True)
if __name__=='__main__':print(json.dumps(checks()))

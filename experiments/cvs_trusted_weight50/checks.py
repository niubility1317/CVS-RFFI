"""Focused local actual-model loss/gradient/update checks, no real IQ or query."""
import copy,json,tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .observer import Observer,records
from experiments.cvs_source_weight import checks as weight_checks
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_phase1_overlay.model import native_modules
def main():
 torch.set_num_threads(2);configs=[d.config(r) for r in d.training_rows()];native=native_modules();old=native._trusted_pseudo_loss_hook;results=[]
 for c in configs:
  a=d.make_args(c,'cpu');assert a.epochs==50 and a.label_epochs==33 and a.pseudo_epochs==17 and a.lambda_u==.16 and not a.freeze_backbone and not a.baseline_ckpt and not a.teacher_ckpt
  with tempfile.TemporaryDirectory() as temp,numerical_context(d.FULL_FP32_POLICY):
   m=api.build(c,'cpu',design=d).train();obs=Observer();out=Path(temp);obs.initialize(m,c,out);obs.steps=33*222
   torch.manual_seed(2026101414);x=torch.zeros(2,2,256);x[1]=1.;l=m.forward_identity_only(x)['tx_logits'];s=m.forward_identity_only(x)['tx_logits'];prob=torch.tensor([.995,.97]);desired=torch.logsumexp(s[:,1:].detach(),1)+torch.log(prob/(1-prob));boost=torch.zeros_like(s);boost[:,0]=desired-s[:,0].detach();s=s+boost
   pseudo=torch.zeros(2,dtype=torch.long);mask=torch.ones(2,dtype=torch.bool);L=F.cross_entropy(l,torch.tensor([0,1]),label_smoothing=.01);U=F.cross_entropy(s[mask],pseudo[mask]);p=s.softmax(1);ent=-(p*p.clamp_min(1e-8).log()).sum(1).mean();rng=torch.get_rng_state().clone();before={k:v.clone() for k,v in m.state_dict().items()}
   with obs.native_scope(native):
    v=native._trusted_pseudo_loss_hook(model=m,args=a,epoch=34,batch_idx=1,loss_l=L,loss_u=U,loss_ent=ent,active=True,logits=s,pseudo=pseudo,mask=mask)
    if c['arm']=='SIG_TIME':assert v is U
    else:assert not torch.allclose(v,U)
    assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in m.parameters()) and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    loss=native._compose_unlabeled_closed_loss(L,args=a,muse_state=None,muse_total=v,identity=v,entropy=ent,domain=U*0,adv=U*0,satellite=U*0);loss.backward()
    assert m.id_backbone.encoder.core.id_backbone.cls_head.weight.grad is not None and m.id_backbone.encoder.core.id_backbone.cls_head.weight.grad.norm()>0
    pool=[(n,p) for n,p in m.named_parameters() if '.sequence_pool.' in n];assert len(pool)==5 and all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in pool)
    optimizer=torch.optim.SGD(m.parameters(),lr=.001);optimizer.step();assert all(not torch.equal(before[n],p) for n,p in pool)
   assert native._trusted_pseudo_loss_hook is old and len(obs.grad_records)==1 and len(obs.grad_records[0]['records'])==3
   results.append(dict(arm=c['arm'],default_head_joint_gradient=True,feature_gradient_records=3,pool_tensors_updated=5,observer_parameter_or_rng_mutation=False,actual_loss=float(v.detach()),lambda_ent=a.lambda_ent))
 # The weighting changes only loss, so initial default body/head/feature functions must match.
 with numerical_context(d.FULL_FP32_POLICY):
  aa=api.build(configs[0],'cpu',design=d).eval();bb=api.build(configs[1],'cpu',design=d).eval();assert all(torch.equal(v,bb.state_dict()[k]) for k,v in aa.state_dict().items())
  with torch.no_grad():assert torch.equal(aa.forward_identity_only(torch.zeros(2,2,256))['id_feat_joint'],bb.forward_identity_only(torch.zeros(2,2,256))['id_feat_joint'])
 negatives=0
 for edit in [dict(epochs=51),dict(model_seed=99),dict(query_count=1),dict(output_root='bad')]:
  bad=dict(configs[1],**edit)
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed config accepted')
 for key,value in [('hardness_scale',.2),('raw_bounds',[.5,1.5]),('no_U_truth',False),('default_joint_head',False),('gradient_epochs',[50])]:
  bad=copy.deepcopy(configs[1]);bad['trusted_weight'][key]=value
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed weight/permissions accepted')
 print(json.dumps(dict(status='PASS',query_read=False,real_IQ_read=False,head_fit=False,actual_models=results,initial_parameters_and_features_identical=True,negative_configs=negatives,scope='Two synthetic actual native models; scoped pre-backward parameter derivatives plus normal joint optimizer update. Not a real source training run.')))
if __name__=='__main__':main()

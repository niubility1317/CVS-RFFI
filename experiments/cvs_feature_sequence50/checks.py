"""Focused mathematical and actual-default-head checks; no dataset access."""
import json
import torch
from .features import distribution_descriptor,path_signature,moment_descriptor,SequencePool

def check_math():
 torch.set_num_threads(2)
 omega=torch.zeros(14,6);omega[:,0]=torch.linspace(.5,2,14)
 p=torch.zeros(1,4,6);q=p.clone();p[0,:,0]=torch.tensor([-1.,1.,-1.,1.]);q[0,:,0]=torch.tensor([-2**.5,0.,0.,2**.5])
 for mode in ('MEAN','MEAN_VAR','NONLINEAR_MEAN'):
  assert torch.allclose(distribution_descriptor(p,mode,omega),distribution_descriptor(q,mode,omega),atol=1e-6)
 assert not torch.allclose(distribution_descriptor(p,'RFF',omega),distribution_descriptor(q,'RFF',omega))
 order=torch.tensor([1,3,0,2]);assert torch.allclose(distribution_descriptor(p,'RFF',omega),distribution_descriptor(p[:,order],'RFF',omega),atol=1e-7)
 loop=torch.zeros(1,5,6);loop[0,:,:2]=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
 s=path_signature(loop,False);assert s.shape==(1,21) and torch.equal(s[:,:6],torch.zeros(1,6)) and abs(float(s[0,6])-1)<1e-7
 assert torch.allclose(path_signature(loop.flip(1),False),-s,atol=1e-7)
 assert not torch.allclose(path_signature(loop[:,[0,2,1,3,4]],False),s)
 assert path_signature(loop,True).shape==(1,28) and moment_descriptor(loop).shape==(1,27)
 assert torch.allclose(moment_descriptor(loop),moment_descriptor(loop[:,[2,0,4,1,3]]),atol=1e-7)
 for arm in ('MEAN','MEAN_VAR','NONLINEAR_MEAN','RFF','COV','SIG','SIG_TIME'):
  m=SequencePool(96,arm,seed=2026092701);h=torch.randn(3,96,32,requires_grad=True);z=m(h)
  assert z.shape==(3,160) and torch.isfinite(z).all();z.square().mean().backward()
  assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad.norm()>0
  for name,pv in m.named_parameters():assert pv.grad is not None and torch.isfinite(pv.grad).all() and pv.grad.norm()>0,(arm,name)
  assert m.omega.requires_grad is False
 try:SequencePool(96,'UNREGISTERED',seed=1)
 except ValueError:pass
 else:raise AssertionError('Unregistered mechanism accepted')
 return dict(status='PASS',equal_moments_different_distribution=True,known_signed_area=True,RFF_permutation_invariant=True,signature_order_sensitive=True,all_feature_parameters_live=True,classifier_training=False,target_data_read=False)



def check_model():
 from copy import deepcopy
 from . import design as d
 from .model import build,classifier,groups
 torch.set_num_threads(2);x=torch.randn(8,2,256);y=torch.arange(8)%6
 base=build(d.config(d.rows()[0]),'cpu');reference=deepcopy(classifier(base).state_dict());base.eval();basez=base.forward_identity_only(x)['id_feat_joint'].detach();records=[];parameters=[]
 for row in d.rows()[:8]:
  c=d.config(row);m=build(c,'cpu');h=classifier(m)
  assert h.weight.shape==(6,160) and h.scale==30 and all(torch.equal(v,reference[k]) for k,v in h.state_dict().items())
  m.eval();before=m.forward_identity_only(x);z=before['id_feat_joint'].detach()
  assert z.shape==(8,160) and torch.allclose(before['tx_logits'],h.classify(z),atol=1e-5)
  if row['arm']!='BASE_REF':assert not torch.allclose(z,basez)
  W=h.weight.detach().clone()
  with torch.no_grad():h.weight.copy_(torch.randn_like(h.weight))
  assert torch.equal(m.forward_identity_only(x)['id_feat_joint'],z)
  with torch.no_grad():h.weight.copy_(W)
  clone=build(c,'cpu');clone.load_state_dict(m.state_dict(),strict=True);clone.eval()
  assert torch.equal(clone.forward_identity_only(x)['id_feat_joint'],z)
  teacher=deepcopy(m).eval();assert torch.equal(teacher.forward_identity_only(x)['id_feat_joint'],z)
  m.train()
  with m._phase1_labeled_feature_context():out=m.forward_identity_only(x)
  torch.nn.functional.cross_entropy(out['tx_logits'],y).backward()
  assert m._feature_l_steps==1 and h.weight.grad is not None and h.weight.grad.norm()>0
  pg,_=groups(m,c);optimized={id(p) for g in pg for p in g['params']}
  for name,p in m.named_parameters():
   if '.sequence_pool.' in name:assert id(p) in optimized and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,(row['arm'],name)
  assert all(not mod._forward_hooks for mod in m.modules())
  extra=m._feature_learning_report['extra_parameters']
  if row['arm']!='BASE_REF':parameters.append(extra)
  records.append(dict(arm=row['arm'],extra_parameters=extra,default_head_initial_unchanged=True,default_head_joint_gradient=True,features_independent_of_class_weights=True,actual_z_changed=row['arm']!='BASE_REF',teacher_and_checkpoint_roundtrip=True))
 assert len(set(parameters))==1
 invalid=d.config(d.rows()[1]);invalid['separate_head_training']=True
 try:d.validate(invalid)
 except ValueError:pass
 else:raise AssertionError('Separate head training authorized')
 return dict(status='PASS',rows=records,head_fit_updates=0,target_data_read=False)

if __name__=='__main__':print(json.dumps(dict(math=check_math(),model=check_model())))

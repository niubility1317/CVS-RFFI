"""Focused descriptor, real adapter and default-head checks; synthetic inputs only."""
import copy,json,math,torch
from torch import nn
from . import design as d,model as api
from .features import CompositePool,export
from .descriptors import distribution_descriptor,moment_descriptor,path_signature,ARMS

def checks():
 torch.set_num_threads(2);seed=d.SEEDS[0];models={}
 for arm in d.ARMS[1:]:models[arm]=api.build(d.config(next(r for r in d.rows() if r['arm']==arm)),'cpu').eval()
 pools={a:m.id_backbone.encoder.sequence_pool for a,m in models.items()}
 assert [sum(p.numel() for p in pools[a].parameters()) for a in d.ARMS[1:]]==[18976,55968,55968]
 original={a:{k:v for k,v in m.state_dict().items() if '.sequence_pool.' not in k} for a,m in models.items()}
 assert all(set(v)==set(original['SIG_TIME']) and all(torch.equal(t,original['SIG_TIME'][k]) for k,t in v.items()) for v in original.values())
 a,b=pools['SIG_TIME_X3'].state_dict(),pools[d.PRIMARY].state_dict()
 assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items() if k!='views.2.descriptor_lift')
 # Fixed mean/variance and covariance descriptors are window permutation invariant.
 u=torch.randn(3,8,6);omega=pools[d.PRIMARY].views[1].omega;order=torch.tensor([6,0,4,2,7,3,5,1])
 assert torch.allclose(distribution_descriptor(u,'MEAN_VAR',omega),distribution_descriptor(u[:,order],'MEAN_VAR',omega),atol=1e-6)
 assert torch.allclose(moment_descriptor(u),moment_descriptor(u[:,order]),atol=1e-6)
 assert not torch.allclose(path_signature(u,True),path_signature(u[:,order],True))
 # Directional mean/variance follows from the same-path full moments; avoid an independent-information claim.
 mu=u.mean(1);center=u-mu[:,None];cov=center.transpose(1,2)@center/u.shape[1]
 assert torch.allclose(distribution_descriptor(u,'MEAN_VAR',omega),torch.cat((mu@omega.T,torch.einsum('id,bdk,ik->bi',omega,cov,omega)),1),atol=1e-6)
 x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3]);boundary=[]
 for arm,m in models.items():
  out=m.forward_identity_only(x);z=out['id_feat_joint'].detach().clone();head=m.id_backbone.encoder.core.id_backbone.cls_head
  assert type(head).__name__=='ResidualPhysicalHead' and head.weight.shape==(6,160) and head.scale==30 and out['tx_logits'].shape==(4,6) and z.shape==(4,160)
  with torch.no_grad():saved=head.weight.clone();head.weight.zero_();other=m.forward_identity_only(x)['id_feat_joint'];head.weight.copy_(saved)
  assert torch.equal(z,other)
  exported=export(m).eval().requires_grad_(False)
  assert torch.equal(z,exported.forward_features(x)) and not any(n.endswith('cls_head.weight') for n,p in exported.named_parameters())
  assert torch.allclose(z,torch.cat([m.forward_identity_only(t[None])['id_feat_joint'] for t in x]),atol=1e-4,rtol=1e-5)
  m.train();before={n:p.detach().clone() for n,p in m.id_backbone.encoder.sequence_pool.named_parameters()};optimizer=torch.optim.AdamW(m.parameters(),lr=2e-4)
  loss=torch.nn.functional.cross_entropy(m.forward_identity_only(x)['tx_logits'],y,label_smoothing=.01);loss.backward()
  for name,p in m.id_backbone.encoder.sequence_pool.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
  assert head.weight.grad is not None and head.weight.grad.norm()>0
  optimizer.step();optimizer.zero_grad(set_to_none=True)
  assert all(not torch.equal(v,dict(m.id_backbone.encoder.sequence_pool.named_parameters())[n]) for n,v in before.items())
  m.eval();teacher=copy.deepcopy(m);assert torch.equal(m.forward_identity_only(x)['id_feat_joint'],teacher.forward_identity_only(x)['id_feat_joint'])
  assert all(not v._forward_hooks and not v._forward_pre_hooks for v in m.modules())
  boundary.append(dict(arm=arm,feature_parameters=sum(p.numel() for n,p in m.named_parameters() if n.startswith('id_backbone.') and not n.endswith('core.id_backbone.cls_head.weight')),pool_parameters=sum(p.numel() for p in m.id_backbone.encoder.sequence_pool.parameters()),W_zeroing_leaves_actual_joint160_identical=True,actual_synthetic_joint_gradient_update=True,EMA_deepcopy_exact=True))
 negatives=0
 for c in (dict(d.config(d.rows()[1]),separate_head_training=True),dict(d.config(d.rows()[1]),readouts=['FITTED_HEAD']),dict(d.config(d.rows()[1]),model_seed=123)):
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Foreign seed/head config accepted')
 from experiments.cvs_feature_sequence50.source import native_default_head_only
 from experiments.cvs_phase1_overlay.model import native_modules
 native=native_modules();old=native._evaluate_zid_leakage_probes
 with native_default_head_only(native) as evidence:
  assert native._evaluate_zid_leakage_probes(None,None,None,None)['head_fit_updates']==0
 assert native._evaluate_zid_leakage_probes is old and evidence['disabled_calls']==1
 return dict(status='PASS',synthetic_only=True,boundary=boundary,protocol_negatives=negatives,default_joint_head_only=True,source_query_IQ_read=False,native_fitted_probe_disabled=True,independent_views_not_same_path_information_claim=True)
if __name__=='__main__':print(json.dumps(checks()))

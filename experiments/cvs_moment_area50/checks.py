"""Synthetic algebra plus actual native gradient/update/export; no data access."""
import copy,json,torch
from . import design as d,model as api
from .features import shape_descriptor
from experiments.cvs_signature_stats50.features import export
from experiments.cvs_signature_stats50.descriptors import path_signature

def checks():
 torch.set_num_threads(2);torch.manual_seed(711);u=torch.randn(100,8,6,dtype=torch.float64)
 cov=shape_descriptor(u,'COV_SHAPE');area=shape_descriptor(u,'AREA_SHAPE');assert cov.shape==area.shape==(100,27) and torch.equal(cov[:,:12],area[:,:12])
 center=u-u.mean(1,keepdim=True);sigma=center.transpose(1,2)@center/8;ij=torch.triu_indices(6,6,offset=1)
 assert torch.equal(cov[:,:6],u.mean(1)) and torch.equal(cov[:,6:12],sigma.diagonal(dim1=-2,dim2=-1)) and torch.equal(cov[:,12:],sigma[:,ij[0],ij[1]])
 order=torch.tensor([0,5,3,1,6,4,2,7]);assert torch.allclose(cov,shape_descriptor(u[:,order],'COV_SHAPE'),atol=1e-12,rtol=1e-12)
 reverse=shape_descriptor(u.flip(1),'AREA_SHAPE');assert torch.allclose(area[:,:12],reverse[:,:12],atol=1e-12,rtol=1e-12) and torch.allclose(area[:,12:],-reverse[:,12:],atol=1e-12,rtol=1e-12)
 assert not torch.allclose(area[:,12:],shape_descriptor(u[:,order],'AREA_SHAPE')[:,12:])
 timed=path_signature(u,True);indices=torch.triu_indices(7,7,offset=1);time_positions=7+torch.nonzero(indices[1]==6).flatten();formula=(u[:,1:]+u[:,:-1]).sum(1)/14-(u[:,0]+u[:,-1])/2
 assert torch.allclose(timed[:,time_positions],formula,atol=1e-12,rtol=1e-12) and torch.allclose(path_signature(u[:,order],True)[:,time_positions],timed[:,time_positions],atol=1e-12,rtol=1e-12)
 models={r['arm']:api.build(d.config(r),'cpu').eval() for r in d.training_rows()};params={k:dict(m.named_parameters()) for k,m in models.items()}
 assert all(set(v)==set(params['SIG_TIME']) and all(torch.equal(p,params['SIG_TIME'][n]) for n,p in v.items()) for v in params.values())
 lifts=[models[k].id_backbone.encoder.sequence_pool.views[0].descriptor_lift for k in ('COV_SHAPE','AREA_SHAPE')];assert torch.equal(*lifts) and lifts[0].shape==(27,28) and torch.linalg.matrix_rank(lifts[0])==27
 x=torch.randn(6,2,256);y=torch.arange(6);rows=[]
 for arm,m in models.items():
  pool=m.id_backbone.encoder.sequence_pool;assert sum(p.numel() for p in pool.parameters())==18976
  out=m.forward_identity_only(x);z=out['id_feat_joint'].detach().clone();head=m.id_backbone.encoder.core.id_backbone.cls_head
  assert type(head).__name__=='ResidualPhysicalHead' and head.weight.shape==(6,160) and head.scale==30
  with torch.no_grad():old=head.weight.clone();head.weight.zero_();other=m.forward_identity_only(x)['id_feat_joint'];head.weight.copy_(old)
  assert torch.equal(z,other);e=export(m).eval().requires_grad_(False);assert torch.equal(z,e.forward_features(x)) and not any(n.endswith('cls_head.weight') for n,p in e.named_parameters())
  m.train();before={n:p.detach().clone() for n,p in pool.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
  loss=torch.nn.functional.cross_entropy(m.forward_identity_only(x)['tx_logits'],y,label_smoothing=.01);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in pool.parameters()) and head.weight.grad is not None and head.weight.grad.norm()>0
  opt.step();opt.zero_grad(set_to_none=True);assert all(not torch.equal(v,dict(pool.named_parameters())[n]) for n,v in before.items())
  m.eval();teacher=copy.deepcopy(m);assert torch.equal(m.forward_identity_only(x)['id_feat_joint'],teacher.forward_identity_only(x)['id_feat_joint']) and all(not v._forward_hooks for v in m.modules())
  rows.append(dict(arm=arm,pool_parameters=18976,actual_feature_export_exact=True,W_independent=True,default_head_joint_gradient=True,new_parameters_numerically_updated=True,teacher_deepcopy_exact=True))
 for bad in [dict(d.config(d.training_rows()[0]),readouts=['REFITTED_HEAD']),dict(d.config(d.training_rows()[0]),model_seed=123),dict(d.config(d.training_rows()[0]),separate_head_training=True)]:
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Foreign/head configuration accepted')
 return dict(status='PASS',synthetic_only=True,strict_report_route1_formula=True,same_mean_variance_27slots=True,only15_relations_differ=True,covariance_permutation_invariant=True,area_reversal_antisymmetric=True,area_internal_order_sensitive=True,SIG_TIME_time_area_identity_checked=True,same_parameter_initialization=True,same_fixed_fullrank_lift=True,actual_network_rows=rows,negative_configs_rejected=True,source_or_query_IQ_read=False)
if __name__=='__main__':print(json.dumps(checks()))

"""Focused hierarchy algebra and real native f_theta checks, without IQ data."""
import copy,json,torch
from . import design as d,model as api
from .features import continuous
from experiments.cvs_signature_stats50.features import export
from experiments.cvs_source_hierarchy.algebra import signature

def checks():
 torch.set_num_threads(2);torch.manual_seed(527);u=torch.randn(10,32,6,dtype=torch.float64,requires_grad=True)
 q,h,g=continuous(u);clock=torch.linspace(0,1,32,dtype=u.dtype)[None,:,None].expand(len(u),-1,-1);p=torch.cat((u,clock),-1)
 assert len(q)==4 and len(h)==2 and torch.allclose(g,signature(p),atol=1e-12,rtol=1e-12) and torch.allclose(h[0],signature(p[:,:16]),atol=1e-12,rtol=1e-12) and torch.allclose(h[1],signature(p[:,16:]),atol=1e-12,rtol=1e-12)
 grad=torch.autograd.grad(g.square().sum(),u,retain_graph=True)[0];directgrad=torch.autograd.grad(signature(p).square().sum(),u)[0];assert torch.allclose(grad,directgrad,atol=1e-11,rtol=1e-11)
 models={r['arm']:api.build(d.config(r),'cpu').eval() for r in d.training_rows()}
 native=lambda m:{k:v for k,v in m.state_dict().items() if '.sequence_pool.' not in k}
 assert all(all(torch.equal(v,native(m)[k]) for k,v in native(models['SIG_TIME']).items()) for m in models.values())
 params=[dict(models[k].named_parameters()) for k in ('HIER_STAT','HIER_ORDER')];assert set(params[0])==set(params[1]) and all(torch.equal(v,params[1][k]) for k,v in params[0].items())
 pools={k:m.id_backbone.encoder.sequence_pool for k,m in models.items()};assert torch.equal(pools['HIER_STAT'].views[0].coarse_stat_lift,pools['HIER_ORDER'].views[0].coarse_stat_lift)
 x=torch.randn(6,2,256);y=torch.arange(6);rows=[]
 for arm,m in models.items():
  pool=pools[arm];expected=32416 if arm.startswith('HIER_') else 18976;assert sum(p.numel() for p in pool.parameters())==expected
  feat_count=sum(p.numel() for n,p in m.id_backbone.named_parameters() if not n.endswith('cls_head.weight'));assert feat_count==d.FEATURE_PARAMETER_COUNTS[arm]
  z=m.forward_identity_only(x)['id_feat_joint'].detach().clone();head=m.id_backbone.encoder.core.id_backbone.cls_head
  assert type(head).__name__=='ResidualPhysicalHead' and head.weight.shape==(6,160) and head.scale==30
  with torch.no_grad():old=head.weight.clone();head.weight.zero_();other=m.forward_identity_only(x)['id_feat_joint'];head.weight.copy_(old)
  assert torch.equal(z,other);e=export(m).eval().requires_grad_(False);assert torch.equal(z,e.forward_features(x)) and not any(n.endswith('cls_head.weight') for n,p in e.named_parameters())
  m.train();before={n:p.detach().clone() for n,p in pool.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
  torch.nn.functional.cross_entropy(m.forward_identity_only(x)['tx_logits'],y,label_smoothing=.01).backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in pool.parameters()) and head.weight.grad.norm()>0
  opt.step();opt.zero_grad(set_to_none=True);assert all(not torch.equal(v,dict(pool.named_parameters())[n]) for n,v in before.items())
  m.eval();teacher=copy.deepcopy(m);assert torch.equal(m.forward_identity_only(x)['id_feat_joint'],teacher.forward_identity_only(x)['id_feat_joint']) and all(not v._forward_hooks for v in m.modules())
  rows.append(dict(arm=arm,pool_parameters=expected,feature_parameters=feat_count,actual_feature_export_exact=True,W_independent=True,default_head_joint_gradient=True,added_parameters_numerically_updated=True,teacher_deepcopy_exact=True))
 badconfigs=[dict(d.config(d.training_rows()[0]),readouts=['REFITTED_HEAD']),dict(d.config(d.training_rows()[0]),model_seed=123),dict(d.config(d.training_rows()[0]),separate_head_training=True),dict(d.config(d.training_rows()[0]),epochs=51),dict(d.config(d.training_rows()[0]),feature_combination={})]
 for bad in badconfigs:
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Foreign/head/budget/feature configuration accepted')
 return dict(status='PASS',synthetic_only=True,source_or_target_IQ_read=False,continuous_composition_and_gradients_equal_direct=True,three_bridges_globalclock=True,same_initial_native_body_W=True,matched_hierarchy_trainable_initial_parameters=True,shared_coarse_control_lift=True,actual_network_rows=rows,protocol_negatives=len(badconfigs),training_rows=4,total_rows=5)
if __name__=='__main__':print(json.dumps(checks()))

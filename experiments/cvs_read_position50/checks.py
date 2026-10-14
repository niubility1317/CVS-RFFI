"""Actual native hook/gradient/export checks, synthetic IQ only."""
import copy,json,torch
from . import design as d,model as api
from experiments.cvs_signature_stats50.features import export
def checks():
 torch.set_num_threads(2);models={r['arm']:api.build(d.config(r),'cpu').eval() for r in d.training_rows()};a,b=(models[k] for k in ('SIG_TIME','SIG_TIME_PRE'));pa,pb=dict(a.named_parameters()),dict(b.named_parameters());assert set(pa)==set(pb) and all(torch.equal(p,pb[k]) for k,p in pa.items())
 torch.manual_seed(711);x=torch.randn(6,2,256);y=torch.arange(6);za=a.forward_identity_only(x)['id_feat_joint'];zb=b.forward_identity_only(x)['id_feat_joint'];assert torch.equal(za,zb)
 records=[]
 for arm,m in models.items():
  ref=m.id_backbone.encoder;block=ref.core.id_backbone.t2;pool=ref.sequence_pool;head=ref.core.id_backbone.cls_head
  assert block.drop.p==.1 and sum(p.numel() for p in pool.parameters())==18976
  before_hooks=sum(len(v._forward_hooks) for v in m.modules());obs={}
  h1=block.pool.register_forward_hook(lambda mm,aa,out:obs.update(pre=out.detach().clone()));h2=block.register_forward_hook(lambda mm,aa,out:obs.update(post=out.detach().clone()));h3=pool.register_forward_pre_hook(lambda mm,aa:obs.update(read=aa[0].detach().clone()))
  m.train();torch.manual_seed(818);out=m.forward_identity_only(x)
  h1.remove();h2.remove();h3.remove();assert not torch.equal(obs['pre'],obs['post']) and torch.equal(obs['read'],obs['pre' if arm==d.PRIMARY else 'post'])
  old={n:p.detach().clone() for n,p in pool.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=2e-4);torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01).backward()
  assert all(p.grad is not None and p.grad.norm()>0 and torch.isfinite(p.grad).all() for p in pool.parameters()) and head.weight.grad.norm()>0
  opt.step();opt.zero_grad(set_to_none=True);assert all(not torch.equal(v,dict(pool.named_parameters())[n]) for n,v in old.items())
  m.eval();z=m.forward_identity_only(x)['id_feat_joint'].detach().clone();oldW=head.weight.detach().clone()
  with torch.no_grad():head.weight.zero_();other=m.forward_identity_only(x)['id_feat_joint'];head.weight.copy_(oldW)
  e=export(m).eval().requires_grad_(False);teacher=copy.deepcopy(m).eval();assert torch.equal(z,other) and torch.equal(z,e.forward_features(x)) and torch.equal(z,teacher.forward_identity_only(x)['id_feat_joint']);assert e.encoder.sequence_read_position==ref.sequence_read_position and teacher.id_backbone.encoder.sequence_read_position==ref.sequence_read_position
  assert sum(len(v._forward_hooks) for v in m.modules())==before_hooks and not any(n.endswith('cls_head.weight') for n,p in e.named_parameters())
  records.append(dict(arm=arm,actual_read_position=ref.sequence_read_position,native_dropout_probability=.1,pool_parameters=18976,exact_export=True,W_independent=True,new_params_updated=True,default_head_joint_gradient=True,teacher_position_and_output_equal=True,hooks_removed=True))
 legacy=api.build(d.config(d.training_rows()[0]),'cpu').eval();del legacy.id_backbone.encoder.sequence_read_position;anchor=api.build(d.config(d.training_rows()[0]),'cpu').eval();assert torch.equal(legacy.forward_identity_only(x)['id_feat_joint'],anchor.forward_identity_only(x)['id_feat_joint'])
 negatives=0
 for bad in [dict(d.config(d.training_rows()[0]),readouts=['REFITTED_HEAD']),dict(d.config(d.training_rows()[0]),model_seed=123),dict(d.config(d.training_rows()[0]),separate_head_training=True)]:
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed protocol accepted')
 anchor.id_backbone.encoder.sequence_read_position='foreign'
 try:anchor.forward_identity_only(x)
 except ValueError:negatives+=1
 else:raise AssertionError('Unregistered hook accepted')
 from .publish import configure,INSPECT
 from experiments.cvs_report_ssl import publish as transport
 configure();context=dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,archive='x',sha256='x',commit='x');compile(transport.REMOTE.replace('CONFIG',repr(context)),'REMOTE','exec');compile(INSPECT.replace('CONFIG',repr(context)),'INSPECT','exec')
 assert "done['models']==3" in INSPECT and "done['new_trained_models']==2" in INSPECT and "done['prediction_count']==504000" in INSPECT
 return dict(status='PASS',synthetic_only=True,same_initial_full_parameters=True,same_initial_eval_features=True,actual_rows=records,legacy_default_position_unchanged=True,protocol_negatives=negatives,release_scripts_compile=True,source_or_target_IQ_read=False)
if __name__=='__main__':print(json.dumps(checks()))

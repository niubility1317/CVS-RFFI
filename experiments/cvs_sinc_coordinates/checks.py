import copy,json,random,torch,numpy as np
from torch.nn import functional as F
from . import design as d
from .model import build,installed,PREFIX,physical_state
from experiments.cvs_transfer_features.model import from_phase1
from experiments.cvs_default_head.predict import logits as default_logits
from experiments.cvs_default_head.source_health import preserving_rng
from experiments.cvs_feature_diagnostics.sinc_change import compare

def checks():
 torch.set_num_threads(2);first=[r for r in d.training_rows() if r['model_seed']==d.SEEDS[0]]
 from experiments.cvs_feature_multiscale50.model import build as original_build
 torch.manual_seed(d.SEEDS[0]);original=original_build(d.baseline_config(first[0]),'cpu');original_rng=torch.get_rng_state().clone()
 models=[];reports=[]
 for row in first:
  torch.manual_seed(d.SEEDS[0]);m=build(d.config(row));assert torch.equal(original_rng,torch.get_rng_state());models.append(m)
  expected=original.state_dict();actual=m.state_dict()
  for k,v in expected.items():
   if not k.startswith(PREFIX):assert torch.equal(v,actual[k]),k
  assert sum(p.numel() for p in m.id_backbone.encoder.parameters())==177025
  assert sum(p.numel() for p in m.id_backbone.encoder.core.id_backbone.sinc.parameters())==48
 torch.manual_seed(1071);x=torch.randn(6,2,256);y=torch.arange(6)
 original.eval()
 with torch.no_grad():expected=original.forward_identity_only(x)
 for m in models:
  m.eval()
  with torch.no_grad():
   result=m.forward_identity_only(x);assert torch.equal(expected['id_feat_joint'],result['id_feat_joint'])
   assert torch.equal(expected['tx_logits'],result['tx_logits'])
 for row,m in zip(first,models):
  c=d.config(row);s=m.id_backbone.encoder.core.id_backbone.sinc;h=m.id_backbone.encoder.core.id_backbone.cls_head
  before=physical_state(s);rng=torch.get_rng_state().clone();assert compare(before,before,s._native_type)['kernel_relative_L2']==0;assert torch.equal(rng,torch.get_rng_state())
  m.train();m.zero_grad(set_to_none=True);F.cross_entropy(m.forward_identity_only(x)['tx_logits'],y).backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in s.parameters())
  assert h.weight.grad is not None and h.weight.grad.norm()>0 and h.weight.shape==(6,160) and h.scale==30
  with installed(c) as native:
   parameters=native._optimizer_parameters(m,None)
   assert all(any(p is v for v in parameters) for p in s.parameters())
   teacher=copy.deepcopy(m).eval().requires_grad_(False);ts=teacher.id_backbone.encoder.core.id_backbone.sinc
   with torch.no_grad():tk0=ts._filters(torch.device('cpu'),torch.float32).clone()
   coords={n:p.detach().clone() for n,p in s.named_parameters()}
   opt=torch.optim.AdamW(parameters,lr=.0002,weight_decay=.0001);opt.step()
   changed={n:float((p-coords[n]).abs().max().detach()) for n,p in s.named_parameters()}
   moved=compare(before,physical_state(s),s._native_type)
   native._update_ema_model(teacher,m,.999,versioned=True)
   with torch.no_grad():tk1=ts._filters(torch.device('cpu'),torch.float32).clone()
   if row['arm']=='CENTER_25K':
    assert all(v>0 for v in changed.values()) and moved['kernel_relative_L2']>0
    assert not torch.equal(tk0,tk1) and all(torch.equal(getattr(s,n),getattr(ts,n)) for n in ('low_origin_hz','band_origin_hz'))
  m.eval().requires_grad_(False);state={k:v.clone() for k,v in m.state_dict().items()}
  feature=from_phase1(m).eval().requires_grad_(False);assert sum(p.numel() for p in feature.parameters())==176065
  with torch.no_grad():
   result=m.forward_identity_only(x);z=feature.forward_features(x)
   assert torch.equal(z,result['id_feat_joint']) and torch.equal(default_logits(m,x),result['tx_logits'])
   disposable=copy.deepcopy(m);disposable.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_()
   assert torch.equal(z,disposable.forward_identity_only(x)['id_feat_joint'])
   assert torch.isfinite(default_logits(m,torch.zeros(2,2,256))).all()
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  # A cached module must refresh after a strict load, including origin buffers.
  clone=copy.deepcopy(m);cs=clone.id_backbone.encoder.core.id_backbone.sinc
  with torch.no_grad():cs._filters(torch.device('cpu'),torch.float32)
  clone.load_state_dict(state,strict=True)
  with torch.no_grad():assert torch.equal(clone.forward_identity_only(x)['id_feat_joint'],z)
  reports.append(dict(arm=row['arm'],identity_parameters=177025,feature_export_parameters=176065,Sinc_trainable_parameters=48,numeric_coordinate_max_updates=changed,synthetic_step_kernel_relative_L2=moved['kernel_relative_L2'],optimizer_membership=True,EMA_cache_refreshed=True,joint160_export_exact=True,W_independent=True,default_head_joint_gradient=True,separate_head_updates=0))
 # Supplemental V evaluator must restore mode and all RNG streams.
 model=models[0].train();loader=type('Loader',(),{'generator':torch.Generator().manual_seed(42)})()
 rng=torch.get_rng_state().clone();nr=np.random.get_state();pr=random.getstate();lr=loader.generator.get_state()
 def evaluation(m,l,dev):
  m.eval();torch.rand(3);np.random.rand();random.random();torch.rand(3,generator=l.generator);return 1
 assert preserving_rng(model,loader,'cpu',evaluation)==1 and model.training
 assert torch.equal(rng,torch.get_rng_state()) and np.array_equal(nr[1],np.random.get_state()[1]) and pr==random.getstate() and torch.equal(lr,loader.generator.get_state())
 negatives=0
 for change in [dict(epochs=51),dict(label_epochs=32),dict(readouts=['FRESH_COSINE_ALL_L']),dict(classifier_training='head only'),dict(checkpoint_sources=['unknown']),dict(target_access=True),dict(sinc_coordinates={})]:
  c=copy.deepcopy(d.config(first[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered configuration accepted')
 return dict(status='PASS',synthetic_only=True,real_IQ_or_truth_read=False,rows=reports,initial_whole_model_logits_joint160_exact=True,all_other_initial_states_exact=True,constructor_and_Sinc_diagnostic_CPU_RNG_preserved=True,source_health_RNG_mode_preserved=True,negative_checks=negatives)
if __name__=='__main__':print(json.dumps(checks()))

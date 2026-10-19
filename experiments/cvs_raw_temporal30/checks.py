import ast,copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d,model as api,optimizer as optim
from experiments.cvs_feature_adapters30.loss import objective
from experiments.cvs_F_nontrue20 import model as base

def run():
 torch.set_num_threads(2);torch.manual_seed(20261019902);p=base.build(d.parent_config(),'cpu').eval().requires_grad_(False);state={k:v.clone() for k,v in p.state_dict().items()};x=torch.randn(2,2,256);second=x+.2*torch.randn_like(x);labels=torch.tensor([0,4]);rng=torch.get_rng_state().clone();NN=0;records={};negative=0;core_inputs=[]
 h=p.id_backbone.encoder.core.id_backbone.register_forward_pre_hook(lambda m,a:core_inputs.append(a[0].clone()))
 with torch.no_grad():teacher=p.forward_identity_only(x)['tx_logits'];NN+=2
 h.remove();assert len(core_inputs)==1
 tree=ast.parse((d.ROOT/'experiments/cvs_raw_temporal30/train.py').read_text(encoding='utf-8'));node=next(n.args[0] for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='save' and isinstance(n.args[0],ast.Call) and any(k.arg=='epoch' for k in n.args[0].keywords));initial_adapter=None
 for style in ('RAW','PAIR'):
  c=d.config(next(r for r in d.rows() if r['style']==style and r['As']));m=api.from_state(c,state);info=api.counts(m);core=m.id_backbone.encoder.core.id_backbone;head=m.get_parameter(d.HEAD);head_id=id(head);native_inputs=[];encoder_inputs=[];mac=[]
  h1=core.register_forward_pre_hook(lambda q,a:native_inputs.append(a[0].clone()));h2=core.raw_temporal.down.register_forward_pre_hook(lambda q,a:encoder_inputs.append(a[0].clone()));handles=[q.register_forward_hook(lambda q,a,o:mac.append(q.weight.numel()*o.shape[-1])) for q in core.raw_temporal.modules() if isinstance(q,torch.nn.Conv1d)]
  if initial_adapter is None:initial_adapter={n:v.clone() for n,v in core.raw_temporal.state_dict().items()}
  assert all(torch.equal(v,core.raw_temporal.state_dict()[n]) for n,v in initial_adapter.items())
  with torch.no_grad():before=m.forward_identity_only(x);NN+=2
  h1.remove();h2.remove()
  for handle in handles:handle.remove()
  expected=core_inputs[0] if style=='RAW' else F.avg_pool1d(core_inputs[0],2).repeat_interleave(2,dim=-1)
  assert torch.equal(native_inputs[0],core_inputs[0]) and torch.equal(encoder_inputs[0],expected) and (style!='PAIR' or not torch.equal(encoder_inputs[0],core_inputs[0])) and sum(mac)==214528
  assert torch.equal(before['tx_logits'],teacher) and before['id_feat_joint'].shape==(2,160) and torch.equal(head,state[d.HEAD]) and core.raw_temporal.packets==2
  opt=optim.build(m,c);frozen={n:v.clone() for n,v in m.state_dict().items() if n not in info['trainable_names']}
  for step in range(2):
   opt.zero_grad(set_to_none=True);a=m.forward_identity_only(x)['tx_logits'];b=m.forward_identity_only(second)['tx_logits'];NN+=4;loss,metrics=objective(a,b,teacher,labels);loss.backward()
   assert head.grad is not None and head.grad.abs().sum()>0 and all(getattr(core.raw_temporal,n).weight.grad.abs().sum()>0 for n in ('B1','B2'))
   if step==1:assert all(getattr(core.raw_temporal,n).weight.grad.abs().sum()>0 for n in ('down','mix','temporal'))
   optim.clip(m,c,opt);opt.step();assert id(m.get_parameter(d.HEAD))==head_id
  with torch.no_grad():after=m.forward_identity_only(x);NN+=2
  assert not torch.equal(head,state[d.HEAD]) and not torch.equal(before['id_feat_joint'],after['id_feat_joint']) and all(torch.equal(v,m.state_dict()[n]) for n,v in frozen.items()) and core.raw_temporal.packets==12
  cloned=copy.deepcopy(m)
  with torch.no_grad():clone_out=cloned.forward_identity_only(x);NN+=2
  assert torch.equal(clone_out['tx_logits'],after['tx_logits']) and torch.equal(clone_out['id_feat_joint'],after['id_feat_joint'])
  cp=eval(compile(ast.Expression(node),'actual_train_checkpoint_dict','eval'),dict(m=m,c=c,opt=opt));done=dict(config=c,epoch=30,optimizer_steps=2220,target_access=False,U_used=False,EMA_used=False,default_head_joint=True,head_joint_optimizer_steps=2220,raw_supplement_training_NN=568320);init=dict(inheritance_verified=True,source_roles_match=True,parent_config=d.parent_config(),target_contact=False)
  def read(path):
   name=Path(path).name
   if name=='completion.json':return done
   if name=='source_frozen.json':return dict(config=c,selection='fixed FINAL30',target_access=False)
   if name=='initialization.json':return init
   return dict(role_ids=dict(L_s=['synthetic-L'],V=['synthetic-V']))
  with patch.object(d,'read',read),patch.object(api,'parent',lambda device:p),patch.object(torch,'load',lambda *a,**k:cp):reloaded=api.own_model(c)
  with torch.no_grad():replay=reloaded.forward_identity_only(x);NN+=2
  assert torch.equal(replay['tx_logits'],after['tx_logits']) and torch.equal(replay['id_feat_joint'],after['id_feat_joint']) and not any(q.requires_grad for q in reloaded.parameters())
  head.requires_grad_(False)
  try:api.counts(m)
  except ValueError:negative+=1
  else:raise AssertionError('Frozen head accepted as joint')
  m.requires_grad_(False);head.requires_grad_(True)
  try:api.counts(m)
  except ValueError:negative+=1
  else:raise AssertionError('Head-only training accepted')
  records[style]=dict(total_parameters=902974,trainable_parameters=3212,adapter_parameters=2252,adapter_convolution_MAC=214528,zero_function_exact=True,original_native_input_preserved=True,supplement_input_correct=True,shared_initial_state=True,shared_encoder_once=True,B1_B2_and_inner_encoder_gradient=True,W_and_actual_joint160_updated=True,all_native_frozen_state_equal=True,deepcopy_prediction_and_joint_equal=True,actual_checkpoint_expression_and_loader_replayed=True,synthetic_optimizer_steps=2)
 for key,value in [('default_head_joint',False),('U_used',True),('keep_weight',2.)]:
  bad=copy.deepcopy(c);bad['training'][key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Changed head/U/loss recipe accepted')
 assert all(torch.equal(v,p.state_dict()[n]) for n,v in state.items()) and torch.equal(torch.get_rng_state(),rng)
 return dict(status='PASS',rows=records,protocol_negatives=negative,synthetic_NN=NN,real_checkpoint_loads=0,real_IQ=0,target_access=False,parent_and_global_RNG_unchanged=True)
if __name__=='__main__':print(json.dumps(run()))

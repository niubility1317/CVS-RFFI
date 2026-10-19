"""New behavior only: original W and real features update and reload together."""
import ast,json,math
from unittest.mock import patch
import torch
from . import design as d,model as api,optimizer as optim
from experiments.cvs_feature_adapters30.loss import objective
from experiments.cvs_F_nontrue20 import model as base
from experiments.cvs_rdrop_sinc_units import optimizer as units

def run():
 torch.set_num_threads(2);torch.manual_seed(20261019651);p=base.build(d.parent_config(),'cpu').eval().requires_grad_(False);state={k:v.clone() for k,v in p.state_dict().items()};x=torch.randn(2,2,256);second=x+.2*torch.randn_like(x);labels=torch.tensor([0,4]);rng=torch.get_rng_state().clone();NN=0;records={};negative=0
 with torch.no_grad():teacher=p.forward_identity_only(x)['tx_logits'];NN+=2
 tree=ast.parse((d.ROOT/'experiments/cvs_feature_head_joint30/train.py').read_text(encoding='utf8'));node=next(n.args[0] for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='save' and isinstance(n.args[0],ast.Call) and any(k.arg=='epoch' for k in n.args[0].keywords))
 for style in ('FULL','LORA','TEMP'):
  c=d.config(next(r for r in d.rows() if r['style']==style and r['As']));m=api.from_state(c,state);info=api.counts(m);head=m.get_parameter(d.HEAD);head_identity=id(head);assert info['trainable_parameters']=={'FULL':232193,'LORA':3264,'TEMP':3344}[style]
  with torch.no_grad():before=m.forward_identity_only(x);NN+=2
  assert torch.equal(before['tx_logits'],teacher) and before['id_feat_joint'].shape==(2,160) and torch.equal(head,state[d.HEAD])
  opt=optim.build(m,c);frozen={n:v.clone() for n,v in m.state_dict().items() if n not in info['trainable_names']}
  for step in range(2):
   opt.zero_grad(set_to_none=True);a=m.forward_identity_only(x)['tx_logits'];b=m.forward_identity_only(second)['tx_logits'];NN+=4;loss,metrics=objective(a,b,teacher,labels);loss.backward()
   assert head.grad is not None and head.grad.abs().sum()>0 and any(q.grad is not None and q.grad.abs().sum()>0 for n,q in m.named_parameters() if q.requires_grad and n!=d.HEAD)
   optim.clip(m,c,opt);opt.step();assert id(m.get_parameter(d.HEAD))==head_identity
  with torch.no_grad():after=m.forward_identity_only(x);NN+=2
  assert not torch.equal(head,state[d.HEAD]) and not torch.equal(before['id_feat_joint'],after['id_feat_joint']) and all(torch.equal(v,m.state_dict()[n]) for n,v in frozen.items())
  # Exercise the actual checkpoint dictionary expression and actual loader/factory.
  cp=eval(compile(ast.Expression(node),'actual_train_checkpoint_dict','eval'),dict(m=m,c=c,opt=opt))
  done=dict(config=c,epoch=30,optimizer_steps=2220,target_access=False,U_used=False,EMA_used=False,default_head_joint=True,head_joint_optimizer_steps=2220)
  init=dict(inheritance_verified=True,parent_config=d.parent_config(),target_contact=False)
  def read(path):
   name=__import__('pathlib').Path(path).name
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
  records[style]=dict(total_parameters=info['total_parameters'],trainable_parameters=info['trainable_parameters'],zero_function_exact=True,same_original_head_object=True,W_and_joint160_updated=True,frozen_state_bit_equal=True,actual_checkpoint_expression_and_loader_replayed=True,synthetic_optimizer_steps=2)
 # Native CENTER_UNITS must include W in the same physical clip/AdamW step.
 c=d.config(d.rows()[0]);m=api.from_state(c,state);opt=optim.build(m,c);named=dict(m.named_parameters());grad={n:torch.full_like(q,.02 if i%2 else -.03) for i,(n,q) in enumerate(named.items()) if q.requires_grad}
 for n,g in grad.items():named[n].grad=g.clone()
 norm=math.sqrt(sum(float((g.double()/(25000. if n in units.QNAMES else 1.)).square().sum()) for n,g in grad.items()));coef=min(1.,5./(norm+1e-6));sinc=m.get_submodule(units.PREFIX[:-1]);origins=dict(zip(units.QNAMES,(sinc.low_origin_hz,sinc.band_origin_hz)));optim.clip(m,c,opt);opt.step()
 for n,g in grad.items():
  lr=.0002*(1000. if n in units.QNAMES else 1.);eps=1e-8/(1000. if n in units.QNAMES else 1.);g=g*coef;expected=state[n]*(1-.0002*.0001)
  if n in origins:expected=expected-.0002*.0001*origins[n]/25000.
  expected-=lr*g/(g.abs()+eps);torch.testing.assert_close(named[n],expected,atol=2e-6,rtol=1e-6)
 for key,value in [('default_head_joint',False),('default_head_frozen',True),('U_used',True)]:
  from copy import deepcopy
  bad=deepcopy(c);bad['training'][key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Changed head/U recipe accepted')
 assert all(torch.equal(v,p.state_dict()[n]) for n,v in state.items()) and torch.equal(torch.get_rng_state(),rng)
 return dict(status='PASS',rows=records,protocol_negatives=negative,physical_FULL_joint_W_oracle=True,synthetic_NN=NN,real_checkpoint_loads=0,synthetic_checkpoint_schema_replays=3,real_IQ=0,target_access=False,parent_and_global_RNG_unchanged=True)

if __name__=='__main__':print(json.dumps(run()))

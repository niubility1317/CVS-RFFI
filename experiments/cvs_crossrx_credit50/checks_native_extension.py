"""Execute actual native AST blocks: compatibility and exactly-once gradients."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def block(test):
 p=Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
 tree=ast.parse(p.read_text(encoding='utf-8'))
 nodes=[node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test)==test]
 if len(nodes)!=1:raise ValueError('Actual native extension location changed')
 return nodes[0]


def execute(node,values):
 env=dict(torch=torch,**values)
 code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
 exec(compile(code,'actual-native-extension','exec'),env)
 return env


def check():
 early=block('source_labeled_feature_hook is not None')
 late=block('source_step_probe is not None')
 received={}
 def old(*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
  received['old']=True
  return identity_loss*0,{}
 def new(**kw):
  received['new']=kw['labeled_inputs']
  return kw['identity_loss']*0,{}
 new._accepts_labeled_inputs=True
 x=torch.arange(8*2*256,dtype=torch.float32).reshape(8,2,256)
 fields=dict(model=None,out_l={},y_l=torch.zeros(8,dtype=torch.long),extra_l={},d_l=None,
             r3_optimizer_steps=5,loss_tx_l=torch.tensor(1.),labeled_clean_count=6,x_l=x)
 execute(early,dict(fields,source_labeled_feature_hook=old))
 execute(early,dict(fields,source_labeled_feature_hook=new))
 assert received['old'] and torch.equal(received['new'],x[:6])
 a=torch.tensor(2.,requires_grad=True);b=torch.tensor(3.,requires_grad=True)
 base=a.square()+.16*b.square();aux=.1*a*b
 common=dict(model=None,optimizer=None,epoch=6,batch_idx=2,phase='pseudo',r3_optimizer_steps=1119,
             loss=base,scaled_closed_loss=base,feature_auxiliary_metrics={},args=SimpleNamespace(lambda_u=.16,lambda_ent=.01),
             loss_u=b.square(),loss_ent=b*0,labeled_clean_count=128,pseudo_total=256)
 for answer in (None,{},dict(old_observation=True)):
  out=execute(late,dict(common,feature_auxiliary_metrics={},source_step_probe=lambda **kw:answer))
  assert out['loss'] is base and out['scaled_closed_loss'] is base
 seen={}
 def probe(**kw):
  seen['full_base']=kw['total_loss']
  seen['U']=kw['unlabeled_loss']
  return dict(post_core_auxiliary_loss=aux,post_core_auxiliary_metrics={'aux':1.})
 out=execute(late,dict(common,feature_auxiliary_metrics={},source_step_probe=probe))
 assert seen['full_base'] is base and float(seen['U'].detach())==float((.16*b.square()).detach())
 ga,gb=torch.autograd.grad(out['loss'],(a,b),retain_graph=True)
 assert torch.allclose(ga,torch.tensor(4.3)) and torch.allclose(gb,torch.tensor(1.16))
 assert torch.equal(out['loss'],out['scaled_closed_loss']) and out['feature_auxiliary_metrics']=={'aux':1.}
 for invalid in (1.,torch.ones(1),torch.ones((),dtype=torch.long),torch.ones((),device='meta')):
  try:execute(late,dict(common,feature_auxiliary_metrics={},source_step_probe=lambda **kw:dict(post_core_auxiliary_loss=invalid)))
  except ValueError:pass
  else:raise AssertionError('Invalid post-core scalar accepted')
 return dict(status='VERIFIED',actual_native_AST_consumed=True,old_callbacks_unchanged=True,
             clean_L_inputs_only_by_optin=True,full_base_U_visible_to_probe=True,
             auxiliary_gradient_added_exactly_once=True,invalid_scalar_device_rejected=True,
             actual_data_or_checkpoint_reads=0,actual_optimizer_updates=0)

if __name__=='__main__':print(json.dumps(check()))

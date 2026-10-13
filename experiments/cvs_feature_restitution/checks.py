"""No-data checks for the persistent z and actual source evidence paths."""
import ast,json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .features import build,embedding,layout,evidence_mask,RelativeNormBlock
from experiments.cvs_feature_multiscale50.model import classifier

SEED=2026092701

def check():
 torch.set_num_threads(2);torch.manual_seed(421);x=torch.randn(12,2,256);y=torch.arange(12)%6
 base=build(SEED);reference=deepcopy(classifier(base).state_dict());records=[]
 for name,kind,competition in (('BASE','none',False),('ORDINARY','ordinary',False),('RESTORE','restore',False),('COMP','none',True),('RESTORE_COMP','restore',True)):
  m=build(SEED,kind,competition);c=classifier(m)
  assert c.weight.shape==(6,160) and c.scale==30 and all(torch.equal(v,reference[k]) for k,v in c.state_dict().items())
  m.train()
  with m._phase1_labeled_feature_context():out=m.forward_identity_only(x)
  assert out['id_feat_joint'].shape==(12,160) and out['tx_logits'].shape==(12,6)
  F.cross_entropy(out['tx_logits'],y).backward()
  core,_,_=layout(m)
  if kind!='none':
   block=core.t1;assert isinstance(block,RelativeNormBlock)
   assert sum(p.numel() for p in block.projection.parameters())+block.alpha_raw.numel()==1225
   assert block.alpha_raw.grad is not None and torch.isfinite(block.alpha_raw.grad) and block.alpha_raw.grad.abs()>0
   assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in block.projection.parameters())
   if kind=='restore':
    h=torch.randn(4,72,31)+.2;q=block.statistics(h);q2=block.statistics(h*2)
    assert q.shape==(4,16) and torch.max(torch.abs(q-q2))<1e-3
    increment=block.projection(q).reshape(4,8,9)
    assert torch.all(increment.var(-1,unbiased=False)>0)
  m.eval();z=embedding(m,x).detach();w=c.weight.detach().clone()
  with torch.no_grad():c.weight.copy_(torch.randn_like(c.weight))
  assert torch.equal(embedding(m,x),z) # Class-indexed W does not define z.
  with torch.no_grad():c.weight.copy_(w)
  clone=build(SEED,kind,competition);clone.load_state_dict(m.state_dict(),strict=True);clone.eval()
  assert torch.equal(embedding(clone,x),z)
  before=m._feature_l_steps
  with m._phase1_labeled_feature_context():assert torch.equal(embedding(m,x),z)
  assert m._feature_l_steps==before
  records.append(dict(arm=name,registered_parameters=sum(p.numel() for p in m.parameters()),identity_parameters=sum(p.numel() for p in m.id_backbone.parameters()),extra=m._feature_learning_report['extra_parameters'],head_initial_same=True,z_independent_of_W=True,all_new_parameters_live=True))
 # Make the independent reference increment nonzero; no source checkpoint/data.
 m=build(SEED,competition=True)
 for module in m.modules():
  if isinstance(module,nn.Dropout):module.p=0
 ref=next(v for _,v in m.named_modules() if type(v).__name__=='ReferenceCVS')
 with torch.no_grad():ref.response_gain.fill_(.3)
 m.train();teacher=deepcopy(m).eval();full=embedding(m,x).detach();tz=embedding(teacher,x).detach();mask_records=[]
 for group in ('time','frequency','PA','reference'):
  with evidence_mask(m,group) as info:
   masked=embedding(m,x).detach()
   assert not torch.allclose(masked,full)
   assert torch.equal(embedding(teacher,x),tz)
  assert torch.equal(embedding(m,x),full) and not m._evidence_active
  assert all(len(v._forward_hooks)==0 for _,v in m.named_modules())
  mask_records.append(info)
 for role in ('U_s','query','target'):
  try:
   with evidence_mask(m,'time',role=role):pass
  except ValueError:pass
  else:raise AssertionError('Non-L role could mask')
 assert teacher._feature_l_steps==0 and not teacher._evidence_active
 # The optional native context is at exactly the real labeled callsite.
 p=Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
 tree=ast.parse(p.read_text(encoding='utf-8-sig'));calls=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.With) and any(isinstance(it.context_expr,ast.Call) and isinstance(it.context_expr.func,ast.Name) and it.context_expr.func.id=='labeled_context' for it in node.items):calls.append(node)
 assert len(calls)==1
 call=calls[0].body[0].value
 assert isinstance(call,ast.Call) and call.args[0].id=='x_l_main' and any(k.arg=='y_tx' and isinstance(k.value,ast.Name) and k.value.id=='y_l' for k in call.keywords)
 return dict(status='PASS',records=records,actual_path_masks=mask_records,teacher_U_forward_unmasked=True,eval_all_paths_restored=True,wrong_role_rejected=True,explicit_native_L_callsite=True,checkpoint_loaded=False,target_data_read=False)

if __name__=='__main__':print(json.dumps(check()))

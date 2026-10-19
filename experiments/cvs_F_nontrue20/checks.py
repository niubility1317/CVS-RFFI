"""Focused loss/protocol negatives; scratch checkpoint smoke is launch step one."""
import copy
import torch
from .loss import nontrue
from . import design as d
def run():
 torch.manual_seed(103);s=torch.randn(24,6,requires_grad=True);t=torch.randn(24,6,requires_grad=True);y=torch.arange(24).remainder(6)
 with torch.no_grad():t.scatter_(1,y[:,None],10.)
 loss,info=nontrue(s,t,y);g=torch.autograd.grad(loss,(s,t),allow_unused=True)
 if not torch.equal(g[0].gather(1,y[:,None]),torch.zeros(24,1)) or g[1] is not None or not g[0].abs().sum()>0:raise AssertionError('Nontrue-only gradient invariant failed')
 s2=s.detach().clone();s2.scatter_add_(1,y[:,None],torch.full((24,1),50.));v,_=nontrue(s2,t,y)
 if not torch.allclose(loss,v):raise AssertionError('Student true-class logit leaked into target')
 p=torch.tensor([4,2,0,5,1,3]);inv=p.argsort();lp,_=nontrue(s[:,p],t[:,p],inv[y])
 if not torch.allclose(loss,lp,atol=1e-6):raise AssertionError('Class order dependent nontrue mapping')
 wrong=t.detach().clone();wrong.scatter_(1,y[:,None],-100.);zero,_=nontrue(s,wrong,y)
 if float(zero.detach())!=0.:raise AssertionError('Incorrect teacher mask/denominator differs')
 neg=0
 for change in ({'epochs':50},{'checkpoint_sources':['old.pth']},{'model_seed':2},{'training_mechanism':{}}):
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:neg+=1
  else:raise AssertionError('Unauthorized recipe mutation accepted')
 from .native_factory import import_regression
 return dict(status='PASS',true_logit_gradient_exact_zero=True,teacher_detached=True,class_permutation=True,incorrect_teacher_zero=True,protocol_negatives=neg,historical_import=import_regression())

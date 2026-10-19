"""Focused loss/gradient, fixed-matrix and hidden-label negative checks."""
import copy,json
from . import design as d
def math_checks():
 import torch
 from torch.nn import functional as F
 from .loss import correction
 g=torch.Generator().manual_seed(61019);a=torch.randn((64,6),generator=g,requires_grad=True);b=torch.randn((64,6),generator=g,requires_grad=True);t=torch.randn((64,6),generator=g);y=torch.arange(64)%6
 p,q=a.log_softmax(1),b.log_softmax(1);kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean();target=t.softmax(1);mask=t.argmax(1)==y
 kd=.5*sum((F.kl_div(F.log_softmax(z/6,1),target,reduction='none').sum(1)*mask).mean() for z in (a,b))
 for mode in ('CE','CSIL'):
  actual,record=correction(a,b,y,t,mode);expected=-.25*kl+(.25*6*kd if mode=='CSIL' else kd*0)
  if not torch.allclose(actual,expected,atol=1e-7):raise ValueError('Full-batch loss normalization differs')
  x=torch.autograd.grad(actual,(a,b),retain_graph=True);z=torch.autograd.grad(expected,(a,b),retain_graph=True)
  if any(not torch.allclose(v,w,atol=1e-7) for v,w in zip(x,z)):raise ValueError('Selected-As gradient normalization differs')
 for args in [(a[:63],b[:63],y[:63],t[:63],'CSIL'),(a,b,y,t.requires_grad_(True),'CSIL')]:
  try:correction(*args)
  except ValueError:pass
  else:raise ValueError('Invalid teacher or selected view accepted')
 return dict(loss_gradient='PASS',all64_mean=True,student_NN=0)
def run():
 from .native_factory import import_regression
 dependency=import_regression()
 count=0
 for r in d.rows():
  c=d.config(r);d.validate(c)
  for k,v in [('epochs',11),('As',False),('distill_mode','other'),('parent_source','unknown'),('checkpoint_sources',['unknown'])]:
   x=copy.deepcopy(c);x[k]=v
   try:d.validate(x)
   except ValueError:count+=1
   else:raise ValueError('Invalid fixed protocol accepted')
  x=copy.deepcopy(c);x['sameIQ_teacher']['all64_CE']=False
  try:d.validate(x)
  except ValueError:count+=1
  else:raise ValueError('Removing As supervision accepted')
 return dict(status='PASS',protocol_negatives=count,math=math_checks(),dependency=dependency,query_access=False,updates=0)
if __name__=='__main__':print(json.dumps(run()))

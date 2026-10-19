import torch
from .loss import row,direction,geometry
def run():
 torch.manual_seed(2026101931)
 x=torch.randn(3,2,256,requires_grad=True);w=torch.randn(6,512,requires_grad=True);z=torch.tanh(x.flatten(1))@w.T;y=torch.tensor([0,2,5]);g=row(z,x,y,create_graph=True,retain_graph=True);target=torch.randn_like(g,requires_grad=True);mask=torch.tensor([True,False,True])
 v,c=direction(g,target,mask);p=torch.autograd.grad(v,(w,target),allow_unused=True)
 expected=((1.-c.detach())*mask).sum()/3
 if not torch.allclose(v.detach(),expected):raise AssertionError('Must normalize over all source packets')
 if p[0] is None or not torch.isfinite(p[0]).all() or not p[0].abs().sum()>0 or p[1] is not None:raise AssertionError('Second order student/teacher detach differs')
 a,_=direction(g.detach(),g.detach()*7,torch.ones(3,dtype=torch.bool));b,_=direction(g.detach(),-g.detach(),torch.ones(3,dtype=torch.bool))
 if abs(float(a))>1e-6 or abs(float(b)-2)>1e-6:raise AssertionError('Scale-invariant signed derivative target differs')
 zeros,_=direction(g.detach(),target.detach(),torch.zeros(3,dtype=torch.bool))
 if float(zeros)!=0:raise AssertionError('Incorrect teacher must remove auxiliary only')
 ref=geometry([('body',w)],(torch.ones_like(w),),(2*torch.ones_like(w),))
 if abs(ref['cosine']-1)>1e-12:raise AssertionError('Physical metric geometry failed')
 from . import design as d
 negatives=0
 for change in ({'unique_source_V':1800},{'target_access':True},{'optimizer_updates':1},{'head_fit_updates':1},{'teacher_or_student_state_update':True}):
  c=d.config();c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unauthorized source probe mutation accepted')
 return dict(status='PASS',student_second_order=True,teacher_detached=True,scale_invariant=True,sign_sensitive=True,all_batch_denominator=True,protocol_negatives=negatives,optimizer_updates=0,source_IQ=0,target_access=False)
if __name__=='__main__':print(run())

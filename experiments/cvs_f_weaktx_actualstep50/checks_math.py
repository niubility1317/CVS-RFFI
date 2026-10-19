"""Independent small-cone and stored FP32 step/control checks; no IQ."""
import torch
from .core import project,route,contributions

def run():
 # Sign is loss-gradient dot physical displacement <=0.
 h=torch.eye(3,dtype=torch.float64);x=torch.tensor([1.,-2.,3.],dtype=torch.float64);p,info=project(x,h)
 torch.testing.assert_close(p,torch.tensor([0.,-2.,0.],dtype=x.dtype),rtol=0,atol=1e-12);assert not info['solver_fallback']
 for hh in (torch.tensor([[1.,0.],[-1.,0.],[0.,1.]],dtype=x.dtype),torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=x.dtype)):
  z=torch.tensor([1.,-2.],dtype=x.dtype);q,_=project(z,hh);torch.testing.assert_close(q,torch.tensor([0.,-2.],dtype=x.dtype),rtol=0,atol=1e-12)
 hh=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]],dtype=x.dtype);q,_=project(torch.ones(2,dtype=x.dtype),hh);torch.testing.assert_close(q,torch.zeros(2,dtype=x.dtype),rtol=0,atol=1e-12)
 q,_=project(-torch.ones(3,dtype=x.dtype),h);torch.testing.assert_close(q,-torch.ones(3,dtype=x.dtype),rtol=0,atol=0)
 zero,_=project(torch.zeros(3,dtype=x.dtype),h*0);assert zero.count_nonzero()==0
 # High-dimensional float32 material correction and matched stored step norm.
 torch.manual_seed(3021);before=(torch.randn(64)*.1).float();delta=(torch.randn(64)*.002).float();after=(before+delta).float();g=torch.stack([torch.ones(64),-torch.arange(64).float()/64,torch.sin(torch.arange(64).float())]).double();blocks=[(0,16),(16,64)]
 pc,ct,detail=route(before,after,g,blocks,1.)
 assert detail['stored_norm_match_error']<=.001 and detail['full_strength_max_residual_relative']<=.001 and not detail['storage_match_fallback'] and not detail['projection_storage_fallback']
 assert not torch.equal(pc,after) and not torch.equal(pc,ct) and pc.dtype==ct.dtype==torch.float32
 # A common change of physical coordinates has the same normalized cone.
 bd=before.double();ad=after.double();p0,c0,inf0=route(bd,ad,g,blocks,1.)
 factor=torch.cat([torch.full((16,),25000.,dtype=x.dtype),torch.ones(48,dtype=x.dtype)]);p1,c1,inf1=route(bd*factor,ad*factor,g/factor,blocks,1.)
 torch.testing.assert_close(p1,p0*factor,rtol=1e-10,atol=1e-10);torch.testing.assert_close(c1,c0*factor,rtol=1e-10,atol=1e-10)
 assert max(abs(a-b) for a,b in zip(inf0['raw_normalized_dots'],inf1['raw_normalized_dots']))<1e-9
 no,co,nn=route(before,before,g,blocks,1.);assert torch.equal(no,before) and torch.equal(co,before)
 # All three visible-L class contributions retain actual sum/128, no reweight.
 first=torch.arange(768,dtype=torch.float64).sin().reshape(128,6).requires_grad_();second=first.flip(0);y=torch.arange(128)%6;parts,counts=contributions(first,second,y)
 ce=.5*(torch.nn.functional.cross_entropy(first,y,reduction='none',label_smoothing=.01)+torch.nn.functional.cross_entropy(second,y,reduction='none',label_smoothing=.01))
 for value,mask in zip(parts,(y==1,y==3,(y!=1)&(y!=3))):torch.testing.assert_close(value,ce[mask].sum()/128,rtol=0,atol=0)
 negative=0
 for yy in (y-8,y+6):
  try:contributions(first,second,yy)
  except ValueError:negative+=1
  else:raise AssertionError('Only visible source TX allowed')
 return dict(status='PASS',independent_three_risk_QP=True,degenerate_opposing_redundant_zero_constraints=True,stored_FP32_norm_match=True,full_strength_first_order_constraint=True,common_physical_unit_invariance=True,actual_sum128_class_contributions=True,protocol_negative_cases=negative,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':
 import json
 print(json.dumps(run()))

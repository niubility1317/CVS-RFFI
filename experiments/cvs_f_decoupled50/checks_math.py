"""Closed-form multi-positive gradient decomposition and protocol cases."""
import torch
from .loss import anchors,penalty
def run():
 torch.manual_seed(7301);dtype=torch.float64;y=torch.tensor([0,0,0,1,1,1,2,2]);rx=torch.tensor([1,3,4,1,3,4,1,3]);z=torch.randn(8,7,dtype=dtype,requires_grad=True)
 values,detail=anchors(z,y,rx);valid=detail['valid'];q=detail['q'];gs=torch.autograd.grad(values['NATIVE'].sum(),z,retain_graph=True)[0];gq=torch.autograd.grad(values['Q_DCL'].sum(),z,retain_graph=True)[0];gp=torch.autograd.grad(((1-q)*values['POS_COMP']).sum(),z,retain_graph=True)[0]
 assert torch.allclose(gs,gq+gp,atol=1e-12,rtol=1e-10)
 gd=torch.autograd.grad(values['DCL'].sum(),z,retain_graph=True)[0];assert (gd-gq).norm()>1e-7 and (gs-gq).norm()>1e-7
 # One positive/anchor has exactly zero positive-competition derivative.
 ys=torch.tensor([0,0,1,1]);rs=torch.tensor([1,3,1,3]);zs=torch.randn(4,9,dtype=dtype,requires_grad=True);vv,ss=anchors(zs,ys,rs);gg=[torch.autograd.grad(vv[m].sum(),zs,retain_graph=True)[0] for m in ('NATIVE','Q_DCL')];assert torch.allclose(*gg,atol=1e-12,rtol=1e-10)
 # Permutation, feature rotation and fixed old-mask equivalence.
 p=torch.tensor([7,1,5,3,2,0,4,6]);vp,_=anchors(z[p],y[p],rx[p]);assert all(torch.allclose(values[k][p],vp[k],atol=1e-12) for k in values)
 Q,_=torch.linalg.qr(torch.randn(7,7,dtype=dtype));vr,_=anchors(z@Q,y,rx);assert all(torch.allclose(values[k],vr[k],atol=1e-12) for k in values)
 for yy,rr in [(torch.zeros(8,dtype=torch.long),rx),(y,torch.ones(8,dtype=torch.long))]:
  va,da=anchors(z,yy,rr);assert not da['valid'].any() and all(torch.equal(v,z.sum(1)*0) for v in va.values())
 bads=[(z,y.float(),rx),(z,y,torch.tensor([2]*8)),(z,torch.full((8,),-1),rx),(z[:7],y,rx),(z*float('nan'),y,rx)];neg=0
 for args in bads:
  try:anchors(*args)
  except (ValueError,FloatingPointError):neg+=1
  else:raise AssertionError('Bad/hidden/non-source descriptors accepted')
 maps={'time_down':torch.randn(8,4,7,dtype=dtype,requires_grad=True),'t2.pool':torch.randn(8,5,3,dtype=dtype,requires_grad=True)}
 for mode in ('Q_DCL','DCL'):
  value,stats=penalty(maps,y,rx,mode);grad=torch.autograd.grad(value,list(maps.values()),retain_graph=True);assert len(stats)==2 and all(torch.isfinite(g).all() and g.norm()>0 for g in grad)
 return dict(status='PASS',multi_positive_gradient_decomposition=True,single_positive_no_competition=True,permutation_rotation_invariance=True,missing_positive_negative_zero=True,shallow_middle_gradients=True,protocol_negative_cases=neg,IQ_reads=0,model_forwards=0,head_fit_updates=0,target_reads=0)
if __name__=='__main__':import json;print(json.dumps(run()))

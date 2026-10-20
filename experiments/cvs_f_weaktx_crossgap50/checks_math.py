"""Independent geometric counterexample and source-input negatives."""
import torch
from .loss import penalty,estimates

def run():
 a=torch.tensor([.2,1.],dtype=torch.float64);b=torch.tensor([.2,-1.],dtype=torch.float64)
 cross,selfgap=estimates(a,b)
 assert abs(float(cross)+.96)<1e-12 and abs(float(selfgap)-1.04)<1e-12
 assert torch.allclose(selfgap-cross,.5*(a-b).square().sum())
 # Opposing within-cell noise raises the self estimate while cross estimate
 # keeps the common class difference distinct. No physical-data claim.
 common=torch.tensor([.3,0.],dtype=torch.float64)
 cc,ss=estimates(common,common);assert float(cc)==float(ss)
 z=torch.tensor([[1.,.2],[1.,-.2],[.2,1.],[-.2,1.],[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([1,1,3,3,0,2]);dom=torch.zeros(6,dtype=torch.long);ids=tuple('packet'+str(i) for i in range(6));outs=[]
 for mode in ('SELF_GAP','CROSS_GAP'):
  val,detail=penalty(z,z,y,dom,ids,mode);assert detail['eligible'] and detail['valid_cells']==1 and detail['cells'][0]['counts']==[2,2]
  # Orthogonal rotation and weak-label swap preserve the geometric penalty.
  rot=torch.tensor([[0.,-1.],[1.,0.]],dtype=z.dtype)
  rv,_=penalty(z@rot,z@rot,y,dom,ids,mode);sw=y.clone();sw[y==1]=3;sw[y==3]=1;sv,_=penalty(z,z,sw,dom,ids,mode)
  torch.testing.assert_close(val,rv);torch.testing.assert_close(val,sv);outs.append(detail)
 # Close overlapping classes make both losses active with finite feature gradients.
 torch.manual_seed(9701);q=(torch.randn(60,4,dtype=torch.float64)*.1+torch.tensor([1.,0.,0.,0.])).requires_grad_();yy=torch.tensor([1,1,3,3]*15);dd=torch.arange(15).repeat_interleave(4);pid=tuple('i'+str(i) for i in range(60));grads=[]
 for mode in ('SELF_GAP','CROSS_GAP'):
  value,detail=penalty(q,q,yy,dd,pid,mode);g=torch.autograd.grad(value,q,retain_graph=True)[0];assert value>0 and torch.isfinite(g).all() and g.norm()>0 and detail['valid_cells']==15;grads.append(g)
 assert (grads[0]-grads[1]).norm()>1e-6
 for bad in ('hidden_label','duplicate_ID','bad_domain','wrong_feature','nonfinite'):
  by=yy.clone();bd=dd.clone();bi=pid;bq=q.detach().clone()
  if bad=='hidden_label':by[0]=-1
  elif bad=='duplicate_ID':bi=(pid[0],)*len(pid)
  elif bad=='bad_domain':bd[0]=15
  elif bad=='wrong_feature':bq=bq[:-1]
  else:bq[0,0]=float('nan')
  try:penalty(bq,bq,by,bd,bi,'CROSS_GAP')
  except ValueError:pass
  else:raise AssertionError('Invalid source input accepted '+bad)
 none,detail=penalty(q[:3],q[:3],yy[:3],dd[:3],pid[:3],'CROSS_GAP');assert not detail['eligible'] and none==0
 return dict(status='PASS',noise_counterexample=True,exact_self_cross_decomposition=True,orthogonal_rotation_and_label_swap=True,disjoint_groups=True,source_domain_day_cells=15,gradient_difference=float((grads[0]-grads[1]).norm()),protocol_negative_cases=5,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':
 import json
 print(json.dumps(run()))

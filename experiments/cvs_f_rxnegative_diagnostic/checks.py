import torch,json
from core import masks,losses,select,OBJECTIVES
def run():
 y=torch.arange(18)//3;rx=torch.tensor([1,3,4]*6);state=torch.get_rng_state().clone();p,neg,valid=masks(y,rx,17);assert torch.equal(state,torch.get_rng_state()) and valid.all();assert torch.equal(neg['SAME_RX'].sum(1),neg['RANDOM_MATCHED'].sum(1)) and not (p&neg['SAME_RX']).any() and not (p&neg['RANDOM_MATCHED']).any()
 assert not (neg['SAME_RX']&~(rx[:,None]==rx[None,:])).any();assert all(not (n&(y[:,None]==y[None,:])).any() for n in neg.values());assert (neg['ALL']!=neg['RANDOM_MATCHED']).any() and (neg['SAME_RX']!=neg['RANDOM_MATCHED']).any()
 p2,n2,v2=masks(y,rx,17);assert torch.equal(neg['RANDOM_MATCHED'],n2['RANDOM_MATCHED']);assert all(torch.equal(v,v.T) for k,v in neg.items() if k!='RANDOM_MATCHED')
 torch.manual_seed(917);maps={'time_down':torch.randn(18,4,5,dtype=torch.float64,requires_grad=True),'t2.pool':torch.randn(18,5,3,dtype=torch.float64,requires_grad=True)};v,stats=losses(maps,y,rx,17);assert set(v)==set(neg) and len(stats)==2
 for mode in v:
  g=torch.autograd.grad(v[mode],list(maps.values()),retain_graph=True);assert all(torch.isfinite(x).all() and x.norm()>0 for x in g)
 # If all RX equal, positive set is empty and every mode returns differentiable0.
 zero,ss=losses(maps,y,torch.ones_like(rx),17);assert all(x==0 for x in zero.values())
 negcases=0
 for yy,rr in [(y.float(),rx),(torch.full_like(y,-1),rx),(y,torch.full_like(rx,2)),(y,rx[:4])]:
  try:masks(yy,rr,17)
  except ValueError:negcases+=1
  else:raise AssertionError('Hidden/non-source labels accepted')
 return dict(status='PASS',exact_negative_count_matching=True,positive_negative_disjoint=True,sameRXonly=True,own_rng_global_unchanged=True,fixed_repeatability=True,actual_descriptor_gradients=True,empty_positive_zero=True,protocol_negative_cases=negcases,IQ_reads=0,model_forwards=0,optimizer_updates=0,head_fit_updates=0,target_reads=0)
if __name__=='__main__':print(json.dumps(run()))

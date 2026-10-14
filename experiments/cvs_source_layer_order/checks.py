import json,torch
from . import design as d
from .objective import relation_indices,losses,metadata

def checks():
 torch.set_num_threads(2);y=[];rx=[];day=[]
 for t in range(6):
  for dd in d.DAYS:
   for r in d.RX:y.append(t);rx.append(r);day.append(dd)
 y,rx,day=(torch.tensor(v,dtype=torch.long) for v in (y,rx,day));ids=list(range(90));q,groups=relation_indices(y,rx,day,ids);assert len(q)==1800 and len(groups)==360
 assert torch.equal(y[q[:,0]],y[q[:,1]]) and (y[q[:,0]]!=y[q[:,2]]).all() and (rx[q[:,0]]!=rx[q[:,1]]).all() and torch.equal(rx[q[:,1]],rx[q[:,2]]) and torch.equal(day[q[:,0]],day[q[:,1]]) and torch.equal(day[q[:,1]],day[q[:,2]])
 z=torch.nn.functional.one_hot(y,6).float().requires_grad_(True);order,supcon,v=losses(z,q,groups);assert order==0 and torch.autograd.grad(order,z)[0].count_nonzero()==0
 zz=torch.full((90,6),1/6**.5,requires_grad=True);hinge,sc,values=losses(zz,q,groups);assert abs(float(hinge)-.1)<1e-6 and abs(float(sc)-__import__('math').log(6))<1e-6
 negatives=0
 for yy,rr,dd,pids,role in [(y,rx,day,ids,'U_s'),(y,rx,day,ids,'V'),(y,rx,day,ids,'query'),(y,rx*0,day,ids,'L_s'),(y,rx,day*0,ids,'L_s'),(y,rx,day,[0]*90,'L_s'),(y.float(),rx,day,ids,'L_s')]:
  try:relation_indices(yy,rr,dd,pids,role)
  except ValueError:negatives+=1
  else:raise AssertionError('Illegal source relationship accepted')
 for bad in (zz[:89],zz*float('nan')):
  try:losses(bad,q,groups)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid feature view accepted')
 from .probe import cohorts
 L=dict(y=y.repeat_interleave(70),rx=rx.repeat_interleave(70),day=day.repeat_interleave(70),ids=list(range(6300)))
 ix=cohorts(L);assert ix.shape==(70,90) and torch.unique(ix).numel()==6300 and torch.equal(ix,cohorts(L))
 return dict(status='PASS',protocol_negative_tests=negatives,triplet_count=1800,reference_groups=360,exact_satisfied_loss_and_gradient_zero=True,cohorts_cover_each_physical_packet_once=True,head_fit_updates=0,optimizer_updates=0,query_read=False)
if __name__=='__main__':print(json.dumps(checks()))

"""Focused memory math, RNG, gradient and source-role tests."""
import json,torch
from .loss import describe,Memory,penalty
from experiments.cvs_f_supcon50.mechanism import penalty as native

def run():
 torch.manual_seed(6191);y=torch.tensor([0,1,2,3,4,5]*3);rx=torch.tensor([1]*6+[3]*6+[4]*6);ids=tuple('p'+str(i) for i in range(18));maps={k:torch.randn(18,8,12,requires_grad=True) for k in ('time_down','t2.pool')};banks={k:Memory() for k in maps};rng=torch.get_rng_state().clone()
 ref,_=native(maps,y,rx,'CROSS_RX');current,detail=penalty(maps,y,rx,ids,banks,'ANCHOR',1111)
 assert torch.allclose(ref,current,atol=1e-6) and len(detail)==2 and all(v['historical_keys']==0 for v in detail)
 g=torch.autograd.grad(current,list(maps.values()),retain_graph=True);assert all(torch.isfinite(v).all() and v.norm()>0 for v in g)
 for level,h in maps.items():banks[level].push(describe(h).detach(),y,rx,ids,1111)
 oldmaps={k:v.detach().clone().requires_grad_() for k,v in maps.items()};newids=tuple('new'+str(i) for i in range(18));value,info=penalty(oldmaps,y,rx,newids,banks,'MEM4',1112)
 assert all(v['historical_keys']==18 and v['key_count']==36 and v['max_history_age']==1 for v in info)
 assert all(not e['v'].requires_grad and e['v'].grad_fn is None for b in banks.values() for e in b.entries)
 grad=torch.autograd.grad(value,list(oldmaps.values()));assert all(torch.isfinite(v).all() and v.norm()>0 for v in grad)
 same,ss=penalty(maps,y,rx,ids,banks,'MEM4',1112);assert torch.allclose(same,current,atol=1e-6) and all(v['historical_keys']==0 and v['current_ID_duplicates_removed']==18 for v in ss)
 for step in range(1112,1116):
  for level,h in maps.items():banks[level].push(describe(h).detach(),y,rx,tuple(str(step)+'_'+p for p in ids),step)
 assert all(len(b.entries)==3 for b in banks.values());_,ss=penalty(maps,y,rx,newids,banks,'MEM4',1116);assert all(v['historical_keys']==54 and v['max_history_age']==3 for v in ss)
 # A stale ID is retained only once, at its most recent occurrence.
 for level,h in maps.items():banks[level].push(describe(h).detach(),y,rx,ids,1116);banks[level].push(describe(h).detach(),y,rx,ids,1117)
 _,ss=penalty(maps,y,rx,newids,banks,'MEM4',1118);assert all(v['historical_keys']==36 for v in ss)
 assert torch.equal(rng,torch.get_rng_state())
 negatives=0
 for yy,rr,ii in [(y-6,rx,ids),(y.float(),rx,ids),(y,rx*0,ids),(y,rx,ids[:-1]),(y,rx,('dup',)*18)]:
  try:penalty(maps,yy,rr,ii,banks,'MEM4',1118)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid source metadata accepted')
 return dict(status='PASS',current_value_exact_native=True,detached_history_no_gradient=True,positive_negative_mask_physical_dedup=True,three_step_expiry=True,private_RNG_unchanged=True,protocol_negative_cases=negatives)
if __name__=='__main__':print(json.dumps(run()))

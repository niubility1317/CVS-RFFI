"""Native equality with empty memory, finite historical partial gradients."""
import json,torch
from .loss import Memory,describe,penalty
from experiments.cvs_f_supcon50.mechanism import penalty as native

def run():
 torch.manual_seed(6192);y=torch.tensor([0,1,2,3,4,5]*3);rx=torch.tensor([1]*6+[3]*6+[4]*6);ids=tuple('p'+str(i) for i in range(18));maps={k:torch.randn(18,8,12,requires_grad=True) for k in ('time_down','t2.pool')};banks={k:Memory() for k in maps};rng=torch.get_rng_state().clone()
 ref,_=native(maps,y,rx,'CROSS_RX');current,dd=penalty(maps,y,rx,ids,banks,'SYM4',1111);assert torch.allclose(ref,current,atol=1e-6)
 g0=torch.autograd.grad(ref,list(maps.values()),retain_graph=True);g1=torch.autograd.grad(current,list(maps.values()),retain_graph=True);assert all(torch.allclose(x,z,rtol=1e-5,atol=1e-6) for x,z in zip(g0,g1))
 for level,h in maps.items():banks[level].push(describe(h).detach(),y,rx,ids,1111)
 value,dd=penalty(maps,y,rx,tuple('new'+x for x in ids),banks,'SYM4',1112);g=torch.autograd.grad(value,list(maps.values()),retain_graph=True);assert all(torch.isfinite(v).all() and v.norm()>0 for v in g) and all(x['historical_keys']==18 for x in dd)
 # Current keys keep gradients; history remains immutable and detached.
 assert all(not e['v'].requires_grad and e['v'].grad_fn is None for b in banks.values() for e in b.entries)
 same,dd=penalty(maps,y,rx,ids,banks,'SYM4',1112);gs=torch.autograd.grad(same,list(maps.values()),retain_graph=True);assert torch.allclose(same,ref,atol=1e-6) and all(torch.allclose(x,z,rtol=1e-5,atol=1e-6) for x,z in zip(g0,gs))
 for step in range(1112,1116):
  for level,h in maps.items():banks[level].push(describe(h).detach(),y,rx,tuple(str(step)+'_'+p for p in ids),step)
 _,dd=penalty(maps,y,rx,tuple('new'+x for x in ids),banks,'SYM4',1116);assert all(x['historical_keys']==54 and x['max_history_age']==3 for x in dd) and torch.equal(rng,torch.get_rng_state())
 negatives=0
 for yy,rr,ii,mode in [(y-6,rx,ids,'SYM4'),(y.float(),rx,ids,'SYM4'),(y,rx*0,ids,'SYM4'),(y,rx,ids[:-1],'SYM4'),(y,rx,('dup',)*18,'SYM4'),(y,rx,ids,'MEM4')]:
  try:penalty(maps,yy,rr,ii,banks,mode,1116)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid source metadata/mode accepted')
 return dict(status='PASS',empty_memory_value_and_gradient_exact_native=True,history_detached_current_keys_live=True,source_physical_dedup=True,three_step_expiry=True,RNG_unchanged=True,protocol_negative_cases=negatives)
if __name__=='__main__':print(json.dumps(run()))

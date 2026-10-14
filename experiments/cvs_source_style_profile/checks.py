import torch
from .profile import mix,metadata,moments

def checks():
 y=torch.zeros(6,dtype=torch.long);rx=torch.tensor([1,3,4,6,8,1]);day=torch.ones(6,dtype=torch.long);ids=[str(i) for i in range(6)];x=(torch.arange(480).float().reshape(6,8,10)/100).requires_grad_(True)
 before=torch.random.get_rng_state().clone();out,info=mix(x,y,rx,day,ids,3)
 assert info['enabled'] and info['changed_packets']>0 and torch.equal(before,torch.random.get_rng_state())
 assert all(i==j or (y[i]==y[j] and day[i]==day[j] and rx[i]!=rx[j] and ids[i]!=ids[j]) for i,j in enumerate(info['donors']))
 g=torch.autograd.grad(out.square().mean(),x)[0];assert torch.isfinite(g).all() and g.abs().sum()>0 and x.grad is None
 negative=0
 for yy,rr,dd,ii in [(y-1,rx,day,ids),(y+6,rx,day,ids),(y.float(),rx,day,ids),(y,rx*0+14,day,ids),(y,rx,day+3,ids),(y,rx,day,['same']*6)]:
  try:metadata(yy,rr,dd,ii)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source metadata accepted')
 return dict(status='PASS',synthetic_only=True,protocol_negative_checks=negative,protected_donors=True,gradient_live=True,global_Torch_RNG_preserved=True,model_or_source_or_target_read=False,head_fit=False)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))

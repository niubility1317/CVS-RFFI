"""Check realized statistics, Jacobian, donor legality, and symmetric controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .bounded_style import transform
from . import design as d

def run(device='cpu'):
    torch.set_num_threads(2)
    y=torch.tensor([0,0,1,1,2,2,2]);rx=torch.tensor([1,2,1,2,1,2,3]);day=torch.ones(7,dtype=torch.long);ids=[0,1,2,3,4,5,4]
    raw=torch.randn(7,12,17,generator=torch.Generator().manual_seed(411))
    scales=torch.tensor([.02,20.,.01,10.,.1,8.,3.])[:,None,None]
    means=torch.tensor([-20.,20.,-30.,30.,-50.,50.,-10.])[:,None,None]
    x=(raw*scales+means).to(device).requires_grad_();rng=torch.get_rng_state().clone();nstate=np.random.get_state()
    evidence=[]
    for eta in (0.,.3,1.):
        plus,a=transform(x,y,rx,day,ids,155,eta=eta,direction=1)
        minus,b=transform(x,y,rx,day,ids,155,eta=eta,direction=-1)
        assert a['donors']==b['donors'] and a['movement_mean']==b['movement_mean']
        for i,j in enumerate(a['donors']):
            if i!=j:assert y[i]==y[j] and day[i]==day[j] and rx[i]!=rx[j] and ids[i]!=ids[j]
        gp=torch.autograd.grad(plus.sum(),x,retain_graph=True)[0]
        gm=torch.autograd.grad(minus.sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(gp*gm,torch.ones_like(x),rtol=2e-6,atol=2e-6)
        assert bool((gp>=2**(-eta)-1e-6).all()) and bool((gp<=2**eta+1e-6).all())
        assert bool(torch.isfinite(plus).all()) and bool(torch.isfinite(minus).all())
        torch.testing.assert_close((plus.mean(-1)+minus.mean(-1))/2,x.mean(-1),rtol=2e-5,atol=2e-5)
        sigma=(x.detach().var(-1,unbiased=False,keepdim=True)+1e-6).sqrt()
        for out,info in ((plus,a),(minus,b)):
            observed=(out.detach().mean(-1,keepdim=True)-x.detach().mean(-1,keepdim=True)).abs()/sigma
            # Mean subtraction around large synthetic offsets is float32-limited.
            assert float(observed.max())<=eta+.003
            assert info['mean_shift_in_sigma_max']<=eta+1e-6 and info['moments_detached']
        if eta==0:assert torch.equal(plus,x) and torch.equal(minus,x)
        else:assert a['mean_clipped_fraction']>0 and a['log_scale_clipped_fraction']>0
        evidence.append(dict(ramp=eta,positive=a,negative=b,reciprocal_Jacobian=True,opposite_mean_shift=True))
    unchanged,info=transform(x,y,torch.ones_like(rx),day,ids,155,direction=-1)
    assert torch.equal(unchanged,x) and info['eligible_packets']==info['changed_packets']==0
    flat=torch.full_like(x,3.).detach().requires_grad_()
    out,info=transform(flat,y,rx,day,ids,155,direction=-1)
    assert torch.equal(out,flat) and torch.equal(torch.autograd.grad(out.sum(),flat)[0],torch.ones_like(flat))
    assert torch.equal(rng,torch.get_rng_state())
    now=np.random.get_state();assert nstate[0]==now[0] and np.array_equal(nstate[1],now[1]) and nstate[2:]==now[2:]
    for settings in (dict(direction=0),dict(eta=1.1),dict(eta=-.1),dict(p=.5),dict(alpha=.2)):
        try:transform(x,y,rx,day,ids,155,**settings)
        except ValueError:pass
        else:raise AssertionError('Unregistered transformation accepted')
    return dict(status='PASS',device=device,synthetic_only=True,target_access=False,cases=evidence,duplicate_PID_donor_exclusion=True,no_donor_bitwise_identity=True,constant_features_finite=True,private_RNG_preserved=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    v=run(a.device);d.write(a.output,v);print(json.dumps(dict(status=v['status'],device=a.device,cases=3)))

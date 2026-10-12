import argparse,json
from pathlib import Path
import torch
from . import design as d
from .model import build,canonical
def run(device='cpu'):
 torch.set_num_threads(2);c=d.config(d.rows()[0]);m=build(c,None,device);x=torch.randn(8,2,256,device=device);m.train();z=m(x);loss=torch.nn.functional.cross_entropy(z,torch.arange(8,device=device)%6);loss.backward()
 if z.shape!=(8,6) or not torch.isfinite(z).all() or any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Finite CNN full gradient required')
 m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():both=m(x);single=torch.cat([m(t[None]) for t in x]);zero=m(torch.zeros_like(x));scaled=canonical(x*100)
 if not torch.allclose(both,single,atol=1e-5,rtol=1e-5) or not torch.isfinite(zero).all() or not torch.allclose(canonical(x),scaled,atol=1e-6,rtol=1e-5) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Perpacket frozen forward differs')
 bad=dict(c,source_U=True)
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Altered permissions accepted')
 try:canonical(x[:,:,:128])
 except ValueError:pass
 else:raise AssertionError('Altered input length accepted')
 return dict(status='PASS',actual_parameters=sum(p.numel() for p in m.parameters()),full_gradient=True,perpacket_independent=True,zero_finite=True,source_only=True,explicit_CFO_identity_input=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))

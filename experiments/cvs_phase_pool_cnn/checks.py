import argparse,json,math
from pathlib import Path
import torch
from . import design as d
from .model import build,rotate,canonical
def run(device='cpu'):
 torch.set_num_threads(2);x=torch.randn(8,2,256,device=device);phi=torch.linspace(-math.pi,math.pi,8,device=device);xx=rotate(x,phi)
 if not torch.allclose(x.square().sum(1),xx.square().sum(1),atol=1e-5,rtol=1e-5) or not torch.allclose(canonical(xx),rotate(canonical(x),phi),atol=1e-5,rtol=1e-5):raise ValueError('Unitphase power/compensation equivariance differs')
 records=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,None,device);m.train();z=m(x);loss=torch.nn.functional.cross_entropy(z,torch.arange(8,device=device)%6);loss.backward()
  if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()) or m.augmented_packets!=(8 if c['phase_train'] else 0):raise ValueError('Actualphase/pooling execution differs')
  m.eval();before={k:v.clone() for k,v in m.state_dict().items()};aug=m.augmented_packets
  with torch.no_grad():a=m(x);b=torch.cat([m(t[None]) for t in x]);zero=m(torch.zeros_like(x))
  if not torch.allclose(a,b,atol=1e-5,rtol=1e-5) or not torch.isfinite(zero).all() or m.augmented_packets!=aug or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozenperpacket inference differs')
  records.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),phase_train=c['phase_train'],pool=c['pool']))
 bad=dict(c,phase_train=not c['phase_train'])
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Changedmethod accepted')
 return dict(status='PASS',unitrotation_power_preserved=True,canonical_equivariance=True,actualgradient=True,inference_noaugmentation=True,perpacket=True,records=records)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))

import argparse,json
from copy import deepcopy
from pathlib import Path
import torch
from .geometry import features,projection,Head
from . import design as d
from .run import score

def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(61);k=projection().to(device);x=torch.randn(12,2,256,device=device)
 theta=.63;a=torch.tensor([[.8,-.6],[.6,.8]],device=device)*1.7
 mixed=a[None]@x
 w=features(x,'COMPLEX_REL',k);wm=features(mixed,'COMPLEX_REL',k)
 if not torch.allclose(w,wm,rtol=1e-4,atol=1e-5):raise AssertionError('Complex scalar invariance failed')
 gl=torch.tensor([[1.4,.3],[-.2,.8]],device=device);p=features(x,'GL2_REL',k);pm=features(gl[None]@x,'GL2_REL',k)
 if not torch.allclose(p,pm,rtol=1e-4,atol=1e-5):raise AssertionError('Full-rank GL2 invariance failed')
 if torch.allclose(features(x,'RAW_REL',k),features(mixed,'RAW_REL',k)):raise AssertionError('RAW control accidentally invariant')
 # Counterexample: linear IQ transmitter differences are erased by GL2.
 if torch.allclose(w,features(gl[None]@x,'COMPLEX_REL',k)):raise AssertionError('Weak quotient erased GL2 difference')
 for arm in d.ARMS:
  z=features(torch.zeros_like(x),arm,k)
  if not torch.isfinite(z).all() or torch.count_nonzero(z):raise AssertionError('Zero packet must be retained and finite')
  torch.manual_seed(4);h=Head().to(device);loss=h(features(x,arm,k)).square().mean();loss.backward()
  if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()):raise AssertionError('Head gradient missing')
 c=d.config(d.rows()[0]);d.validate(c)
 for key,value in [('target_access',True),('U_access',True),('teacher',True),('epochs',21),('held_rx',99)]:
  bad=deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Illegal config accepted: '+key)
 logits=torch.eye(6)*10;y=torch.arange(6);v=score(logits,y,y,y)
 if v['accuracy']!=1. or len(v['class_pairs'])!=15 or sum(map(sum,v['confusion']))!=6:raise AssertionError('Source metric recount')
 return dict(status='PASS',device=device,complex_error=float((w-wm).abs().max()),GL2_error=float((p-pm).abs().max()),zero_retained=True,negative_config_cases=5,TX_GL2_counterexample=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--device',default='cpu');a.add_argument('--output',type=Path,required=True);c=a.parse_args();v=run(c.device);d.write(c.output,v);print(json.dumps(v))

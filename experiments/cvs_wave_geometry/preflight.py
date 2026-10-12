import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .geometry import features,projection,Head
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config
from experiments.cvs_nocfo_foundation.model import build

def run():
 v=checks();c=d.config(d.rows()[0]);L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None:raise ValueError('Source U open')
 for r in d.RX:
  if int((L['rx']==r).sum())!=1260 or int((V['rx']==r).sum())!=5400:raise ValueError('Source RX coverage differs')
 proof=verify(d.SEED);model=build(teacher_config(d.SEED),'cpu',Path(proof['checkpoint'])).eval()
 before={n:t.clone() for n,t in model.state_dict().items()}
 with torch.no_grad():z=model(L['x'][:4])
 if z.shape!=(4,6) or not torch.isfinite(z).all() or any(not torch.equal(t,model.state_dict()[n]) for n,t in before.items()):raise ValueError('Real checkpoint no-query smoke')
 # This checkpoint is smoke-only, never used by the fresh holdout classifiers.
 for arm in d.ARMS:
  head=Head();loss=head(features(L['x'][:12],arm,projection())).square().mean();loss.backward()
  if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Real source branch gradient')
 return dict(status='PASS',checks=v,source_L=len(L['ids']),source_V=len(V['ids']),source_RX=list(d.RX),real_checkpoint_smoke=proof,checkpoint_used_in_probe=False,target_access=False,U_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status=v['status'],target_access=False,U_opened=False)))

import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .run import parent
from experiments.cvs_nocfo_foundation.source import load_source
def run():
 checks();torch.set_num_threads(2);proofs=[]
 for row in d.rows():
  c=d.config(row);L,U,V,contract=load_source(c['parent_config']['source_loader_config']);model,proof=parent(c,'cpu');before={n:t.clone() for n,t in model.state_dict().items()}
  with torch.no_grad():p=model(L['x'][L['rx']!=c['held_rx']][:4])
  if p.shape!=(4,6) or not torch.isfinite(p).all() or any(not torch.equal(t,model.state_dict()[n]) for n,t in before.items()):raise ValueError('Real matching checkpoint no-query smoke')
  proofs.append(proof)
 return dict(status='PASS',proofs=proofs,target_access=False,U_access=False,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',models=5,target_access=False)))

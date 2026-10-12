import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_bootstrap_ema.teachers import verify
from experiments.cvs_bootstrap_ema import design as e
from experiments.cvs_nocfo_foundation.source import load_source
def run():
 torch.set_num_threads(2);checks();c=d.config(d.rows()[0]);L,U,V,_=load_source(c['source_loader_config']);ec=e.config(next(r for r in e.rows() if r['held_rx']==1 and r['arm']=='BOOTSTRAP_EMA'));proof,models=verify(ec,L,'cpu',True);m=models[0];before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():p=m(L['x'][L['rx']!=c['held_rx']][:2])
 if p.shape!=(2,6) or not torch.isfinite(p).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Realmatching noquerysmoke')
 return dict(status='PASS',source_only=True,target_access=False,new_training_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))

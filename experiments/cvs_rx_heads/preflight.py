import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_bootstrap_ema.teachers import verify
from experiments.cvs_bootstrap_ema import design as e
def run():
 torch.set_num_threads(2);result=checks();L,U,V,contract=load_source(d.config(d.rows()[0])['source_loader_config']);c=e.config(next(r for r in e.rows() if r['held_rx']==1 and r['arm']=='BOOTSTRAP_EMA'));proof,models=verify(c,L,'cpu',True)
 for m in models:
  before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m(L['x'][L['rx']!=1][:2])
  if not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Realmatching checkpoint noquery smoke')
 return dict(status='PASS',checks=result,real_checkpoint_smoke=proof,target_access=False,U_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',checks=v['checks'])))

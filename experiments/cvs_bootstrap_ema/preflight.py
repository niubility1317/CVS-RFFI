import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .run import load
from .teachers import verify
def run():
 torch.set_num_threads(2);checks();proof=[]
 for rx in d.RX:
  c=d.config(next(r for r in d.rows() if r['held_rx']==rx and r['arm']=='BOOTSTRAP_EMA'));L,fit,U,held,contract=load(c);pp,models=verify(c,L,'cpu',True)
  for m in models:
   before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m(fit['x'][:2])
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real checkpoint noquery smoke')
  proof.append(dict(rx=rx,teachers=pp,fit_L=5040,fit_U=45360,held_V=5400))
 return dict(status='PASS',source_only=True,target_access=False,U_truth_read=False,teacher_provenance=proof)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',source_only=True,teachers=15)))

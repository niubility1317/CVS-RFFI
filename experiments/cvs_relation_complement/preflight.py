import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .model import build,groups
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build as native
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config
def run():
 torch.set_num_threads(2);checks();L,U,V,contract=load_source(d.config(d.rows()[0])['source_loader_config']);proof=verify(d.SEED);old=native(teacher_config(d.SEED),'cpu',Path(proof['checkpoint'])).eval();before={k:t.clone() for k,t in old.state_dict().items()}
 with torch.no_grad():p=old(L['x'][:4])
 if p.shape!=(4,6) or not torch.isfinite(p).all() or any(not torch.equal(t,old.state_dict()[k]) for k,t in before.items()):raise ValueError('Real matching checkpoint no-query smoke')
 diagnostics=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));fit=L['x'][L['rx']!=c['held_rx']];m=build(c,fit,'cpu');pg,_=groups(m,c);loss=torch.nn.functional.cross_entropy(m(fit[:128]),L['y'][L['rx']!=c['held_rx']][:128]);loss.backward()
  if not all(torch.isfinite(p.grad).all() for g in pg for p in g['params'] if p.grad is not None) or m._current_q is not None:raise ValueError('Real complementary source gradient/context')
  diagnostics.append(dict(arm=arm,layers=m.layer_info,fit_L=len(fit),parameters=sum(p.numel() for g in pg for p in g['params'])))
 return dict(status='PASS',old_checkpoint_smoke_only=proof,diagnostics=diagnostics,target_access=False,U_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',target_access=False,diagnostics=v['diagnostics'])))

import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .objective import objective
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config
def run():
 torch.set_num_threads(2);checks();L,U,V,contract=load_source(d.config(d.rows()[0])['source_loader_config'])
 if U is not None:raise ValueError('U source closed')
 proof=verify(d.SEED);old=build(teacher_config(d.SEED),'cpu',Path(proof['checkpoint'])).eval();before={n:t.clone() for n,t in old.state_dict().items()}
 with torch.no_grad():z=old(L['x'][:4])
 if z.shape!=(4,6) or not torch.isfinite(z).all() or any(not torch.equal(t,old.state_dict()[n]) for n,t in before.items()):raise ValueError('Real checkpoint no-query smoke')
 del old,before
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c['source_loader_config'],'cpu');pg,_=groups(m,c['source_loader_config']);loss,parts=objective(m,L['x'][:128],L['y'][:128],c);loss.backward()
  gradients=[p.grad for g in pg for p in g['params'] if p.grad is not None]
  if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):raise ValueError('Real native finite margin gradients')
 for rx in d.RX:
  if int((L['rx']==rx).sum())!=1260 or int((V['rx']==rx).sum())!=5400:raise ValueError('Source RX layout')
 return dict(status='PASS',real_checkpoint_smoke=proof,checkpoint_used_for_training=False,real_source_native_arms=list(d.ARMS),target_access=False,U_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',real_source_native_arms=v['real_source_native_arms'],target_access=False)))

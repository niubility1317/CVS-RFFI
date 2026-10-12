import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config
def run():
 checks();torch.set_num_threads(2);rows=[]
 for seed in d.SEEDS:
  c=d.config(next(r for r in d.rows() if r['model_seed']==seed));L,U,V,contract=load_source(c['source_loader_config']);proof=verify(seed);old=build(teacher_config(seed),'cpu',Path(proof['checkpoint'])).eval()
  with torch.no_grad():scores=old(L['x'][:4])
  if scores.shape!=(4,6) or not torch.isfinite(scores).all() or U is not None:raise ValueError('Real no-query checkpoint smoke or source U boundary')
  rows.append(dict(seed=seed,source_L=len(L['ids']),source_V=len(V['ids']),smoke_only=proof,model_initialization='scratch; actual checkpoint not reused'))
 return dict(status='PASS',rows=rows,target_access=False,U_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',seeds=list(d.SEEDS),target_access=False)))

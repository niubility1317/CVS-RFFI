import argparse,json
from pathlib import Path
import torch
from . import design as d
from .source import load
from .objective import objective
from .predict import observation_metadata
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config

def run():
 torch.set_num_threads(2);seed=d.SEEDS[0];proof=verify(seed);old=build(teacher_config(seed),'cpu',Path(proof['checkpoint'])).eval();before={k:t.clone() for k,t in old.state_dict().items()}
 with torch.no_grad():z=old(torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(t,old.state_dict()[k]) for k,t in before.items()):raise ValueError('Real checkpoint no-query smoke')
 del old,before
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==seed));L,U,V,contract=load(c);m=build(c['source_loader_config'],'cpu');groups(m,c['source_loader_config']);m.train()
  loss,parts,detail=objective(m,L['x'][:8],L['y'][:8],None,None,None,None,c,1111);loss.backward()
  gg=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
  if not gg or not all(torch.isfinite(v).all() for v in gg):raise ValueError('Real source challenge gradient')
 ids,views=observation_metadata()
 return dict(status='PASS',real_checkpoint_no_query_smoke=True,old_weights_used='smoke only; never training initialization',new_training_inherits_old_weights=False,source_roles='EXACT_MATCH',observation_handles='VALIDATED_ONCE',query_index_count=len(ids),query_IQ_or_truth_read=False,rows=9)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))

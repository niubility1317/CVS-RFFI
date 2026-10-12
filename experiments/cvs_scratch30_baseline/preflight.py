import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from .source import load,objective
from .predict import observation_metadata
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config
from experiments.cvs_stage2_hybrid.semisupervised import VisitPassBank
def run():
 checks();torch.set_num_threads(2);proofs=[]
 for seed in d.SEEDS:
  proof=verify(seed);old=build(teacher_config(seed),'cpu',Path(proof['checkpoint'])).eval();before={k:t.clone() for k,t in old.state_dict().items()}
  with torch.no_grad():z=old(torch.zeros(2,2,256))
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(t,old.state_dict()[k]) for k,t in before.items()):raise ValueError('Real checkpoint no-query smoke')
  del old,before
  for arm in d.ARMS:
   c=d.config(next(r for r in d.rows() if r['model_seed']==seed and r['arm']==arm));L,U,V,contract=load(c);m=build(c['source_loader_config'],'cpu');pg,_=groups(m,c['source_loader_config']);ux=U['x'][:128] if U else None;ids=U['ids'][:128] if U else None;passes=torch.ones(128,dtype=torch.long) if U else None;bank=VisitPassBank(2,.95) if U else None
   loss,parts,detail=objective(m,L['x'][:128],L['y'][:128],ux,ids,passes,bank,c,1111);loss.backward();gg=[p.grad for group in pg for p in group['params'] if p.grad is not None]
   if not gg or not all(torch.isfinite(v).all() for v in gg):raise ValueError('Real source scratch30 gradient')
  proofs.append(proof)
 ids,views=observation_metadata()
 return dict(status='PASS',old_weights_smoke_only=proofs,new_training_inherits_old_weights=False,seeds=list(d.SEEDS),rows=6,source_roles='EXACT_MATCH',view_status='VALIDATED_ONCE',query_index_count=len(ids),query_IQ_or_truth_read=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',rows=6,query_IQ_or_truth_read=False)))

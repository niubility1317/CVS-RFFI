import argparse,json
from pathlib import Path
import torch
from . import design as d
from .model import build_member,groups
from .predict import baseline_proof
from experiments.cvs_stochastic_ensemble.checks import run as member_checks
from experiments.cvs_stochastic_ensemble import design as pilot
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_scratch30_baseline.predict import own_model

def run():
 torch.set_num_threads(2);decision=d.read(pilot.BASE/'source_decision.json')
 if decision!=dict(eligible=True,selected='EQUAL_PROBABILITY_3',target_access=False,automatic_target_launch=False):raise ValueError('Source qualification must precede target confirmation')
 member_checks();L,U,V,contract=load_source(d.config(d.rows()[0])['source_loader_config'])
 if U is not None:raise ValueError('Final ensemble U must stay closed')
 parent_paths=[]
 for row in d.rows():parent_paths.append(str(baseline_proof(d.config(row))))
 baseline=own_model(d.baseline_config(d.SEEDS[0]),'cpu');before={k:t.clone() for k,t in baseline.state_dict().items()}
 with torch.no_grad():p=baseline(L['x'][:4])
 if p.shape!=(4,6) or not torch.isfinite(p).all() or any(not torch.equal(t,baseline.state_dict()[k]) for k,t in before.items()):raise ValueError('Matching final30 real checkpoint no-query smoke')
 diagnostics=[]
 for member in (1,2):
  c=d.member_config(d.rows()[0],member);m=build_member(c,'cpu');pg,_=groups(m,c);loss=torch.nn.functional.cross_entropy(m(L['x'][:64]),L['y'][:64]);loss.backward()
  if not all(torch.isfinite(p.grad).all() for g in pg for p in g['params'] if p.grad is not None):raise ValueError('Actual fullsource member gradients')
  diagnostics.append(dict(member=member,actual_init_seed=c['member_init_seed'],parameters=sum(p.numel() for g in pg for p in g['params'])))
 return dict(status='PASS',source_eligible=True,real_checkpoint_source_roles='EXACT_MATCH',baseline_paths=parent_paths,diagnostics=diagnostics,target_access=False,U_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',source_eligible=True,target_access=False)))

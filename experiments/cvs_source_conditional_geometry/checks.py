import argparse,json
from pathlib import Path
import torch
from . import design as d
from .core import readout,summarize

def fixture(n=2):
    grid=[(day,y,rx) for day in (1,2,3) for y in range(6) for rx in (1,3,4,6,8) for _ in range(n)]
    return dict(role='L_s',x=torch.randn(len(grid),2,256),day=torch.tensor([v[0] for v in grid]),y=torch.tensor([v[1] for v in grid]),rx=torch.tensor([v[2] for v in grid]),ids=list(range(len(grid))))

def run(device):
    from experiments.cvs_scratch_energy_sensitivity import design as training
    from experiments.cvs_scratch_energy_sensitivity.model import build
    from experiments.cvs_scratch_energy_sensitivity.moment_identifiability import permutation_counterexample,exhaustive_cross_expectation
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from unittest.mock import patch
    torch.set_num_threads(2);torch.manual_seed(512)
    for row in d.rows():d.validate(d.config(row))
    c=training.config(training.rows()[0])
    with patch('torch.load',side_effect=AssertionError('Unexpected initial checkpoint access')):model=build(c,device)
    L=fixture()
    with numerical_context(d.FULL_FP32_POLICY):
        flags=actual_flags();values,proof=readout(model,L,c['model_seed'],device,batch_size=64)
    summary=summarize(values,L,expected_cell_size=2)
    assert proof['model_forwards']==3 and proof['L_occurrences']==180 and proof['first_order_derivative_calls']==6
    assert len(summary['records'])==18 and all(len(r['cell_metrics'])==30 for r in summary['records'])
    for r in summary['records']:
        assert abs(sum(r['fractions'].values())-1)<1e-10 and r['conditional_interaction_energy']>=0
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid source role or cell accepted')
    reject(lambda:readout(model,dict(L,role='V'),c['model_seed'],device))
    reject(lambda:readout(model,dict(L,ids=[0]*180),c['model_seed'],device))
    reject(lambda:summarize(values,L,expected_cell_size=70))
    reject(lambda:summarize(values,dict(L,role='U_s'),expected_cell_size=2))
    return dict(status='PASS',device=device,synthetic_only=True,target_access=False,native_random_network=True,backend_flags=flags,readout=proof,
        records=len(summary['records']),role_PID_cell_rejections=True,counterexample=permutation_counterexample(device),expectation=exhaustive_cross_expectation(device))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))

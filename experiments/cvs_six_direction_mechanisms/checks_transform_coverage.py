"""Absent cross-RX state overlap is a scientific negative, not a run crash."""
import json,tempfile
from pathlib import Path
import torch
from . import design as d
from .transforms import diagnose, RECIPE

def check():
    torch.set_num_threads(2)
    entries=[(tx,rx,day,i) for tx in range(6) for rx in d.RX for day in d.DAYS for i in range(9)]
    n=len(entries); y=torch.tensor([a for a,b,c,i in entries]);rx=torch.tensor([b for a,b,c,i in entries]);day=torch.tensor([c for a,b,c,i in entries]);folds=torch.tensor([i%3 for a,b,c,i in entries])
    cache=dict(x=torch.ones(n,2,256),y=y,rx=rx,day=day,ids=[f'L-{i}' for i in range(n)])
    generator=torch.Generator().manual_seed(829)
    h=torch.randn(n,160,generator=generator);q=torch.zeros(n,12)
    for ri,r in enumerate(d.RX):q[rx==r,0]=ri
    model=torch.nn.Module();model.head=torch.nn.Linear(160,6,bias=False).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    ctx=dict(L=cache,V=dict(cache,ids=[f'V-{i}' for i in range(n)]),features={role:{'joint160':h} for role in ('L_s','V')},state_q_raw=q,folds=folds,seed=2026092701,model=model)
    recipe=dict(RECIPE)
    with tempfile.TemporaryDirectory(prefix='transform_coverage_',dir=d.ROOT/'local_artifacts') as folder:
        result=diagnose(ctx,Path(folder)/'diagnostic','cpu')
        saved=torch.load(Path(folder)/'diagnostic/audit.pt',weights_only=False)
    assert not result['applicability']['passed']
    assert len(result['blocks'])==18 and all(b['fit_state_cells']==0 for b in result['blocks'])
    assert result['fit_updates']==0 and len(saved['blocks'])==18
    assert all(b['MSE']['shared_lowrank'] is None for b in result['blocks'])
    assert all(b['status']=='SOURCE_INAPPLICABLE' for b in result['blocks'])
    assert all(r['endpoint_coverage']==0 and not r['passed'] for r in result['fit_results'])
    assert RECIPE==recipe and result['identity_updates']==0 and not result['target_access']
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    json.dumps(result,allow_nan=False)
    return dict(status='PASS',scientific_negative_preserves18blocks=True,fit_updates=0,recipe_unchanged=True,identity_unchanged=True,target_access=False)

if __name__=='__main__':print(json.dumps(check()))

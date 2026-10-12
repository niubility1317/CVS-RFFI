import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .core import scatters,stable_projection,folds,summarize,extract
from .input_policy import apply,NoExplicitCFO

def mathematical():
    labels=[];rxs=[];days=[];ids=[]
    for y in range(6):
        for r in d.config(d.rows()[0])['receivers']:
            for day in (1,2,3):
                for j in range(6):labels.append(y);rxs.append(r);days.append(day);ids.append(f'{y}-{r}-{day}-{j}')
    L=dict(y=torch.tensor(labels),rx=torch.tensor(rxs),day=torch.tensor(days),ids=ids)
    gen=torch.Generator().manual_seed(9981);z=torch.nn.functional.one_hot(L['y'],6).double()*3+torch.randn(len(ids),6,generator=gen,dtype=torch.float64)*.02
    c=d.config(d.rows()[0]);s,_,closure=scatters(z,L);W,e=stable_projection(s,c['stable_projection'])
    assert closure['relative_closure_error']<1e-8 and closure['identity_mean_rank']<=5 and e['rank']==5 and e['denominator_min_eigenvalue']>0
    assert all(torch.linalg.eigvalsh(v).min()>-1e-10 for v in s.values())
    original=folds(L,c['crossfit_seed']);assert torch.equal(original,folds(L,c['crossfit_seed']))
    # Frozen readout may classify clean synthetic evidence; no in-fit query.
    test=summarize({level:z for level in c['levels']},z,L,c)
    assert len(test['geometry'])==3 and len(test['crossfit'])==45
    assert all(x['rank']<=5 and x['fit_query_disjoint'] and x['held_RX_excluded_from_statistics_and_prototypes'] for x in test['crossfit'])
    assert all(x['accuracy']==1 for x in test['records'] if x['scope']=='overall_crossfit')
    zero={k:torch.zeros(6,6,dtype=torch.float64) for k in s};W,e=stable_projection(zero,c['stable_projection']);assert W.shape==(6,0) and e['rank']==0 and e['denominator_min_eigenvalue']>0
    # Permutation per RX preserves mixed distributions; the diagnostic must
    # retain a nonzero identity-specific interaction rather than label it noise.
    perm=torch.nn.functional.one_hot((L['y']+torch.tensor([c['receivers'].index(int(r)) for r in L['rx']]))%6,6).double()
    p,_,_=scatters(perm,L);assert p['interaction'].trace()>0
    return dict(status='PASS',PSD_scatter_and_closure=True,global_TX_rank_at_most5=True,generalized_eigen_residual=True,zero_rank_safe=True,crossfit_disjoint=True,each_physical_ID_once_per_readout=True,all_RX_permutation_interaction_retained=True)

def native(device):
    from experiments.cvs_scratch_engineering import design as native_d
    from experiments.cvs_scratch_engineering.model import build,groups
    from experiments.cvs_equivariant_identity.precision import numerical_context
    c=native_d.config(native_d.rows()[0]);torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY),patch('torch.load',side_effect=AssertionError('Unexpected checkpoint read')):
        model=build(c,device).eval().requires_grad_(False);gen=torch.Generator().manual_seed(322);x=torch.randn(8,2,256,generator=gen)
        from .run import snapshot
        before=snapshot(model);rng=torch.get_rng_state().clone();evidence=apply(model)
        response=next(m for m in model.modules() if isinstance(m,NoExplicitCFO));reference=response.components(x.to(device))['features'];actual=response(x.to(device))
        assert torch.count_nonzero(actual[:,27])==0 and torch.equal(actual[:,:27],reference[:,:27]) and torch.equal(actual[:,28:],reference[:,28:])
        L=dict(x=x,ids=[str(i) for i in range(8)]);features,logits,budget=extract(model,L,device)
        after=snapshot(model);assert all(torch.equal(before[k],after[k]) for k in before) and torch.equal(rng,torch.get_rng_state())
        assert budget['model_forwards']==1 and budget['L_occurrences']==8 and logits.shape==(8,6) and features['id_feat_joint'].shape==(8,160)
        assert not any(p.grad is not None for p in model.parameters())
        return dict(status='PASS',device=device,input_policy=evidence,coordinate_zero_and_other_features_identical=True,actual_native_maps={k:list(v.shape) for k,v in features.items()},parameters_buffers_RNG_unchanged=True,identity_updates=0,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    value=dict(status='PASS',mathematical=mathematical(),native=native(a.device),synthetic_only=True,target_access=False);d.write(a.output,value);print(json.dumps(dict(status=value['status'],device=a.device)))

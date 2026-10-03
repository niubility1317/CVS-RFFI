import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.cvs_response_geometry.run import components,conditions,validate_config
from experiments.cvs_response_geometry.statistics import analyze_arrays,factor_scatter,unit
from experiments.cvs_channel_response_identity.model import build,VARIANTS

torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_feature_capture_reproduces_original_model_without_mutation(variant):
    torch.manual_seed(42);model=build(variant).eval()
    with torch.no_grad():model.response_branch.compensation.context[-1].weight.normal_(0,.1)
    original=copy.deepcopy(model.state_dict());x=torch.randn(6,2,256)
    scores,features=components(model,x)
    assert tuple(scores)==conditions(variant)
    with torch.no_grad():assert torch.equal(scores['all_on'],model(x))
    assert features['coefficients'].shape==(6,8)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in original.items())


def fixture_arrays():
    rng=np.random.default_rng(6);base=rng.normal(size=(24,160));response=.5*base
    weight=rng.normal(size=(6,160))
    return dict(ids=np.asarray([str(i) for i in range(24)]),truth=np.arange(24)%6,receiver=np.arange(24)%2,day=np.arange(24)%3,base=base,response=response,coefficients=rng.normal(size=(24,8)),classifier_weight=weight,classifier_scale=np.asarray(30.),all_on=30*unit(base+response)@unit(weight).T,auxiliary_off=30*unit(base)@unit(weight).T)


def test_parallel_response_does_not_change_normalized_classifier():
    result=analyze_arrays(fixture_arrays(),require_complete=False)
    assert result['head_difference_rank']==5
    assert result['prediction_changes']==0
    assert result['geometry']['parallel_response_energy_fraction']['mean']==pytest.approx(1)
    assert result['geometry']['unit_embedding_movement']['maximum']<1e-14
    assert result['classification']['all_on']['ce']==pytest.approx(result['classification']['auxiliary_off']['ce'])


def test_nested_scatter_exact_additive_factors():
    records=[(t,r,d,k) for t in (0,1) for r in (0,1) for d in (1,2) for k in (-1,1)]
    labels=np.asarray(records);z=np.stack([2*labels[:,0],3*labels[:,1],labels[:,3]],1)
    result=factor_scatter(z,labels[:,0],labels[:,1],labels[:,2])
    assert result['between_tx']==pytest.approx(16)
    assert result['within_tx_between_rx_day']==pytest.approx(36)
    assert result['within_cell']==pytest.approx(16)
    assert sum(result['fractions'].values())==pytest.approx(1)


def test_saved_logit_and_physical_matrix_corruption_rejected():
    q=fixture_arrays();q['all_on'][0,0]+=.1
    with pytest.raises(ValueError,match='logits'):analyze_arrays(q,require_complete=False)
    with pytest.raises(ValueError,match='Incomplete'):analyze_arrays(fixture_arrays())


def test_zero_response_handled_without_nan():
    q=fixture_arrays();q['response']*=0;q['all_on']=q['auxiliary_off'].copy()
    result=analyze_arrays(q,require_complete=False)
    assert result['valid_cosine_packets']==0
    assert result['geometry']['response_base_cosine']['mean'] is None
    assert result['nested_scatter']['response_unit']['fractions']['between_tx'] is None


def test_registered_matrix_and_embedded_scripts():
    folder=Path(__file__).resolve().parents[1]/'experiments/cvs_response_geometry'
    spec=json.loads((folder/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(spec['rows'])==12
    assert sum(len(validate_config(row)['conditions']) for row in spec['rows'])==24
    from experiments.cvs_response_geometry.publish import REMOTE,INSPECT
    from experiments.cvs_response_geometry.collect import REMOTE as RECOUNT
    for script in (REMOTE,INSPECT,RECOUNT):compile(script,'embedded','exec')

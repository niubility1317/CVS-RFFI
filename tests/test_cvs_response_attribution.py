import copy
import torch
import pytest
from experiments.cvs_channel_response_identity.model import build,VARIANTS
from experiments.cvs_response_attribution.run import components,conditions,validate_config

torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_interventions_preserve_model_and_match_full_forward(variant):
    torch.manual_seed(41);model=build(variant).eval()
    with torch.no_grad():
        model.response_branch.compensation.context[-1].weight.normal_(0,.15)
        model.response_branch.compensation.context[-1].bias.normal_(0,.15)
    original=copy.deepcopy(model.state_dict());x=torch.randn(5,2,256)
    with torch.no_grad():expected=model(x)
    scores,stats=components(model,x)
    assert tuple(scores)==conditions(variant)
    assert torch.allclose(scores['all_on'],expected,atol=1e-6,rtol=1e-6)
    assert torch.equal(scores['auxiliary_off'],scores['g_identity'])
    assert torch.count_nonzero(stats['identity_auxiliary_max_abs'])==0
    assert all(value.shape==(5,) and torch.isfinite(value).all() for value in stats.values())
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in original.items())
    assert not model.response_branch.compensation.context[-1]._forward_hooks
    with torch.no_grad():assert torch.equal(expected,model(x))

@pytest.mark.parametrize('variant',VARIANTS)
def test_identity_initialized_branch_makes_all_conditions_identical(variant):
    model=build(variant).eval();scores,_=components(model,torch.randn(3,2,256))
    assert all(torch.equal(scores['all_on'],value) for value in scores.values())

def test_all_registered_configs_and_exact_condition_budget():
    from pathlib import Path
    import json
    path=Path(__file__).resolve().parents[1]/'experiments/cvs_response_attribution/launch_spec.json'
    spec=json.loads(path.read_text(encoding='utf-8'))
    assert len(spec['rows'])==12
    assert sum(len(validate_config(row)['conditions']) for row in spec['rows'])==48
    row=copy.deepcopy(spec['rows'][0]);row['role']='query'
    with pytest.raises(ValueError):validate_config(row)
    row=copy.deepcopy(spec['rows'][0]);row['source_commit']='unknown'
    with pytest.raises(ValueError):validate_config(row)

def test_embedded_scripts_compile():
    from experiments.cvs_response_attribution.publish import REMOTE,INSPECT
    from experiments.cvs_response_attribution.collect import RECOUNT
    for name,script in [('publish',REMOTE),('inspect',INSPECT),('recount',RECOUNT)]:
        compile(script,name,'exec')

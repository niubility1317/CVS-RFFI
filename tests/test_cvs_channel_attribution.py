import copy
import json
from pathlib import Path
import pytest
import torch

from experiments.cvs_channel_attribution.run import components,conditions,validate_config
from experiments.cvs_channel_order_identity.model import build

torch.set_num_threads(2)


@pytest.mark.parametrize('variant',['channel_dual','channel_order'])
def test_nonzero_branch_decomposition_and_identity_intervention(variant):
    torch.manual_seed(901)
    model=build(variant).eval();branch=model.channel_branch
    with torch.no_grad():
        branch.compensation.context[-1].weight.normal_(std=.1)
        branch.compensation.context[-1].bias.normal_(std=.1)
        branch.u_project.weight.normal_(std=.02);branch.v_project.weight.normal_(std=.02)
        if branch.d_project is not None:branch.d_project.weight.normal_(std=.05)
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.randn(5,2,256)
    actual,stats=components(model,x)
    assert set(actual)==set(conditions(variant))
    with torch.no_grad():torch.testing.assert_close(actual['all_on'],model(x),rtol=1e-5,atol=1e-5)
    original=branch.compensation.coefficients
    try:
        branch.compensation.coefficients=lambda z:torch.zeros(len(z),2,4,device=z.device)
        with torch.no_grad():torch.testing.assert_close(actual['g_identity'],model(x),rtol=1e-5,atol=1e-5)
    finally:branch.compensation.coefficients=original
    assert all(torch.equal(frozen[k],v) for k,v in model.state_dict().items())
    assert all(torch.isfinite(v).all() for v in stats.values())
    assert (stats['g_input_relative']>0).all()
    if variant=='channel_order':
        assert (stats['d_projection_relative']>0).all()
        assert ((stats['d_cross_temporal_coherence']>=0)&(stats['d_cross_temporal_coherence']<=1+1e-5)).all()
    else:assert not any(k.startswith('d_') for k in stats)


def config():
    path=Path(__file__).resolve().parents[1]/'experiments/cvs_channel_attribution/launch_spec.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_actual_matrix():
    spec=config()
    assert len(spec['rows'])==8
    assert sum(len(c['conditions']) for c in spec['rows'])==44
    for c in spec['rows']:validate_config(c)


@pytest.mark.parametrize('key,value',[('role','target'),('target_truth','bad'),('source_commit','bad'),('model_seed',1),('source_output','bad'),('output_root','bad'),('conditions',['all_on'])])
def test_reject_changed_permissions_and_paths(key,value):
    c=copy.deepcopy(config()['rows'][0]);c[key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_embedded_remote_scripts_compile():
    from experiments.cvs_channel_attribution.publish import REMOTE,INSPECT
    compile(REMOTE.replace('CONFIG',repr({})),'remote','exec')
    compile(INSPECT.replace('CONFIG',repr({})),'inspect','exec')

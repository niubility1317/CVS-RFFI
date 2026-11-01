import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned
from experiments.pamf_phase3_restoration.feature_condition_diagnostic import source_paths,validate_records,PATHS,SCENES


def test_actual_path_reproduces_parent_without_mutation():
    torch.set_num_threads(2);model=scratch_feature_conditioned(11).eval()
    with torch.no_grad():
        model.output.weight.fill_(.003)
        for layer in model.conditions:
            layer.weight.fill_(.002);layer.bias.fill_(.01)
        generator=torch.Generator().manual_seed(31)
        received=torch.randn(2,2,256,generator=generator);teacher=torch.randn(2,160,generator=generator)
        before={k:v.clone() for k,v in model.state_dict().items()}
        paths,latent=source_paths(model,received,teacher);actual,aux=model(received,return_aux=True)
    assert tuple(paths)==PATHS
    assert torch.equal(paths['actual'],actual)
    assert torch.equal(paths['raw_preprojection'],aux['raw_restored'])
    assert torch.equal(latent,aux['predicted_features'])
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not torch.equal(paths['zero_feature'],paths['no_FiLM'])
    assert not torch.equal(paths['actual'],paths['clean_feature_oracle'])
    for path,value in paths.items():
        assert torch.isfinite(value).all()
        if path!='raw_preprojection':assert torch.allclose(value.square().sum(1).mean(1),torch.ones(2),atol=1e-6,rtol=0)


def test_teacher_reference_does_not_enter_actual_path():
    torch.set_num_threads(2);model=scratch_feature_conditioned(12).eval()
    with torch.no_grad():
        model.output.weight.fill_(.003)
        for layer in model.conditions:layer.weight.fill_(.002)
        received=torch.randn(2,2,256,generator=torch.Generator().manual_seed(32))
        first,z1=source_paths(model,received,torch.zeros(2,160))
        second,z2=source_paths(model,received,torch.ones(2,160))
    assert torch.equal(z1,z2)
    for path in ['actual','zero_feature','no_FiLM','raw_preprojection']:assert torch.equal(first[path],second[path])
    assert not torch.equal(first['clean_feature_oracle'],second['clean_feature_oracle'])
    with pytest.raises(ValueError):source_paths(model,received,torch.zeros(2,6))


def test_source_panel_requires_unique_complete_twenty_draws():
    records=dict(epoch=np.repeat(np.arange(1,21),512),source_index=np.tile(np.arange(512),20),scene=np.resize(np.asarray(SCENES),10240))
    validate_records(records)
    broken={k:v.copy() for k,v in records.items()};broken['source_index'][1]=0
    with pytest.raises(ValueError):validate_records(broken)
    broken={k:v.copy() for k,v in records.items()};broken['epoch'][-1]=21
    with pytest.raises(ValueError):validate_records(broken)

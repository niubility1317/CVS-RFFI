import pytest
import torch
from experiments.pamf_r46_prior_collapse.verification import verify_outputs


def outputs():
    native=(torch.ones(2,2,256),torch.ones(2),torch.full((2,160),.05086),torch.zeros(2,6))
    converted=tuple(v.clone() for v in native)
    converted[0].add_(1e-7)
    converted[2][0,63]+=3.1106173992156982e-6
    converted[3][0,0]=1e-5
    return native,converted,tuple(v.clone() for v in converted)


def test_local_reproduction_does_not_use_F_near_zero_as_restorer_equivalence_gate():
    native,converted,reloaded=outputs()
    with pytest.raises(AssertionError):
        for a,b in zip(native,converted):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    value=verify_outputs(native,converted,reloaded)
    assert value['restorer_close'] and value['strict_reload_exact']
    assert not value['outputs']['features']['elementwise_close']
    assert not value['outputs']['logits']['elementwise_close']
    assert value['outputs']['features']['max_abs']>3e-6


@pytest.mark.parametrize('index',range(4))
def test_every_nonfinite_output_blocks_smoke(index):
    a,b,c=outputs();b[index].flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='finite'):verify_outputs(a,b,c)


def test_true_restorer_change_still_blocks_with_unchanged_tolerance():
    a,b,c=outputs();b[0][0,0,0]=2.
    with pytest.raises(AssertionError):verify_outputs(a,b,c)


def test_strict_export_reload_must_remain_bitwise_identical_including_F():
    a,b,c=outputs();c[3][0,0]+=1e-7
    with pytest.raises(ValueError,match='reload'):verify_outputs(a,b,c)


def test_classifier_decision_change_is_recorded_and_does_not_select_version():
    a,b,c=outputs();b[3][0,2]=2e-5;c=tuple(v.clone() for v in b)
    value=verify_outputs(a,b,c)
    assert value['argmax_disagreements']==1 and not value['F_equivalence_gate']


@pytest.mark.parametrize('bad',('variance','shape'))
def test_positive_variance_and_registered_shapes_required(bad):
    a,b,c=outputs()
    if bad=='variance':b[1][0]=0.
    else:b=(b[0],b[1],b[2][:,:159],b[3])
    with pytest.raises(ValueError):verify_outputs(a,b,c)

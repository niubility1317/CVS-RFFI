import numpy as np
import pytest
import torch
from argparse import Namespace
from experiments.pamf_r46_source_phase import design as d
from experiments.pamf_r46_source_phase import runner
from experiments.pamf_r46_source_phase.runner import rotate, select_rows, compare, validate_paths


@pytest.mark.parametrize('turns',range(4))
def test_rotation_is_unitary_and_reversible(turns):
    x=torch.randn(5,2,256,dtype=torch.float64)
    y=rotate(x,turns)
    assert torch.equal(rotate(y,(-turns)%4),x)
    assert torch.allclose(y.square().sum((1,2)),x.square().sum((1,2)),rtol=1e-14,atol=1e-12)


@pytest.mark.parametrize('turns',[-1,4,True,.5])
def test_nonregistered_rotation_rejected(turns):
    with pytest.raises(ValueError):rotate(torch.zeros(1,2,256),turns)


def test_role_sampler_balanced_without_tx_labels():
    roles=np.asarray(['L_s']*6300+['U_s']*56700)
    a=select_rows(roles);b=select_rows(roles)
    assert np.array_equal(a,b) and len(np.unique(a))==128
    assert sum(roles[a]=='L_s')==sum(roles[a]=='U_s')==64


def test_invalid_role_pool_rejected():
    with pytest.raises(ValueError):select_rows(np.asarray(['V']*63000))


def test_identity_restorer_commutes_and_perfect_iq_has_zero_error():
    clean=torch.randn(5,2,256)
    result=compare(rotate(clean,1),clean,clean,clean,1)
    assert torch.equal(result['equivariance_nmse'],torch.zeros(5))
    assert torch.equal(result['paired_clean_nmse'],torch.zeros(5))


def test_wrong_coordinate_inverse_is_detected():
    clean=torch.ones(3,2,256)
    clean[:,1]=0
    result=compare(clean,clean,clean,clean,1)
    assert torch.allclose(result['equivariance_nmse'],torch.full((3,),2.))


def test_zero_reference_is_finite_and_does_not_hide_nonzero_error():
    zero=torch.zeros(2,2,256)
    result=compare(torch.ones_like(zero),zero,zero,zero,0)
    assert torch.isfinite(result['paired_clean_nmse']).all()
    assert (result['paired_clean_nmse']>0).all()


@pytest.mark.parametrize('field,value',[
    ('parent','target_checkpoint'),('action','other_data'),('source','target_cache'),('output','old_run')])
def test_foreign_paths_fail_before_io(tmp_path,field,value):
    args=dict(parent=tmp_path/d.PARENT,action=tmp_path/d.ACTION,source=tmp_path/'source_cache',output=tmp_path/d.RUN)
    args[field]=tmp_path/value
    with pytest.raises(ValueError):validate_paths(**args)


def test_actual_no_argument_code_commit_interface_prevents_io(tmp_path,monkeypatch):
    args=Namespace(parent=tmp_path/d.PARENT,action=tmp_path/d.ACTION,source=tmp_path/'source_cache',output=tmp_path/d.RUN,expected_commit='b'*40)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'code_commit',lambda:'a'*40)
    with pytest.raises(ValueError,match='Pinned method commit'):runner.execute(args)
    assert not args.output.exists()

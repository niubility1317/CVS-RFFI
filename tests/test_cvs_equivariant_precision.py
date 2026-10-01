import copy
import pytest
import torch
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY,numerical_context,actual_flags


def test_only_cudnn_flag_changes_and_restores_on_failure(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',True)
    before=actual_flags()
    assert {k:v for k,v in before.items() if k!='cudnn_allow_tf32'}=={k:v for k,v in FULL_FP32_POLICY.items() if k!='cudnn_allow_tf32'}
    with pytest.raises(RuntimeError):
        with numerical_context(FULL_FP32_POLICY):
            assert actual_flags()==FULL_FP32_POLICY
            raise RuntimeError('intentional no-data failure')
    assert actual_flags()==before


def test_nonregistered_or_changed_runtime_policy_is_rejected(monkeypatch):
    wrong=copy.deepcopy(FULL_FP32_POLICY);wrong['cudnn_allow_tf32']=True
    with pytest.raises(ValueError):
        with numerical_context(wrong):pass
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',True)
    before=actual_flags()
    with pytest.raises(ValueError,match='Actual backend'):
        with numerical_context(FULL_FP32_POLICY):pass
    assert actual_flags()==before

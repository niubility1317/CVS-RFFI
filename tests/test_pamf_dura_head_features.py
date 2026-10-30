from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_dura_efficiency.head_features import head_feature_basis,projected_feature_error,frozen_head_feature_loss


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Linear(512,160,bias=False)
        self.weight=nn.Parameter(torch.randn(6,160))
        self.core=SimpleNamespace(id_backbone=SimpleNamespace(cls_head=SimpleNamespace(weight=self.weight)))
    def features(self,value):return self.project(value.flatten(1))


def test_basis_preserves_all_original_head_pair_differences():
    reader=Reader().double().eval().requires_grad_(False);basis=head_feature_basis(reader)
    assert basis.shape==(160,5)
    torch.testing.assert_close(basis.T@basis,torch.eye(5,dtype=torch.float64))
    differences=F.normalize(reader.weight,dim=1,eps=1e-4)
    differences=differences[:,None,:]-differences[None,:,:]
    torch.testing.assert_close(differences@basis@basis.T,differences,atol=1e-12,rtol=1e-12)


def test_normalized_residual_in_head_nullspace_has_zero_loss():
    basis=torch.eye(160,dtype=torch.float64)[:,:5]
    actual=torch.zeros(1,160,dtype=torch.float64);reference=actual.clone()
    actual[0,5]=1;reference[0,6]=1
    torch.testing.assert_close(projected_feature_error(actual,reference,basis),torch.zeros(1,dtype=torch.float64))
    actual[0,0]=1
    assert projected_feature_error(actual,reference,basis).item()>0


def test_loss_detaches_reference_and_basis_but_preserves_IQ_gradient():
    reader=Reader().eval().requires_grad_(False);basis=head_feature_basis(reader).float().requires_grad_(True)
    value=torch.randn(3,2,256,requires_grad=True);reference=torch.randn(3,160,requires_grad=True)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    frozen_head_feature_loss(reader,value,reference,basis).mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert reference.grad is None and basis.grad is None and all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())
    reader.train()
    with pytest.raises(ValueError,match='parameters frozen'):head_feature_basis(reader)


def test_exact_feature_match_zero_and_basis_sign_does_not_matter():
    reader=Reader().eval().requires_grad_(False);basis=head_feature_basis(reader).float()
    actual=torch.randn(4,160);reference=torch.randn(4,160)
    torch.testing.assert_close(projected_feature_error(actual,actual,basis),torch.zeros(4),atol=1e-7,rtol=0)
    torch.testing.assert_close(projected_feature_error(actual,reference,basis),projected_feature_error(actual,reference,-basis))


def test_source_calibration_is_fixed_and_rejects_conflicting_view():
    from experiments.pamf_dura_efficiency.head_geometry import summarize
    records=[]
    for epoch in (1,20):
        for _ in range(16):
            stats=dict(base_projected=dict(right_to_left_norm=2.,cosine=.2),
                task_projected=dict(cosine=.8),task_cosine=dict(cosine=.4))
            records.append(dict(epoch=epoch,geometry={'all':stats},metrics={'teacher_argmax_agreement':.95}))
    result=summarize(records)
    assert result['alpha']==.125 and result['candidate_eligible']
    for r in records[16:]:r['geometry']['all']['base_projected']['cosine']=-.01
    assert not summarize(records)['candidate_eligible']

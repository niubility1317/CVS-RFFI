import json
import pytest
import torch
from experiments.pamf_phase3_restoration.variance_trial import configuration,variance_feature_guide,load_scale,TEMPLATE,ARM,GEOMETRY_RUN


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.feature=torch.nn.Linear(512,160).requires_grad_(False)
    def features(self,x):return self.feature(x.flatten(1))
    def classify_features(self,x):raise AssertionError('Classifier/logit guidance forbidden')


def test_source_feature_objective_cap_and_frozen_teacher_without_head():
    torch.manual_seed(7);reader=Reader();y=torch.randn(3,2,256);x=torch.randn_like(y,requires_grad=True)
    target=reader.features(y).detach();scale=torch.ones(160);scale[:80]=.01
    loss,info=variance_feature_guide(reader,x,y,target,scale)
    actual=torch.autograd.grad(loss.sum(),x)[0].flatten(1).norm(dim=1)
    assert (actual<=.5*info['IQ_gradient_norm']*(1+1e-5)).all() and actual.sum()>0
    assert info['feature_standardized_error'].min()>0
    assert all(p.grad is None and not p.requires_grad for p in reader.parameters())


def test_same_source_model_schedule_and_budget_with_new_fidelity_metric_only():
    c=configuration(ARM)
    assert all(c[k]==TEMPLATE[k] for k in TEMPLATE if k not in ['run_id','arm','objective'])
    assert c['epochs']==30 and c['updates']==7410 and c['ancestor_checkpoints']==[]
    assert not c['geometry_uses_G_outputs'] and not c['geometry_uses_TX_labels']
    with pytest.raises(ValueError):configuration('MSAT_M20')


def test_unverified_or_target_geometry_rejected_before_source_training(tmp_path):
    folder=tmp_path/'runs'/GEOMETRY_RUN;folder.mkdir(parents=True)
    (folder/'results.json').write_text(json.dumps({'F0_identity':{}}));(folder/'feature_geometry.json').write_text('{}')
    with pytest.raises(ValueError,match='Ground-trained'):load_scale(tmp_path)

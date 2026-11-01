import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from experiments.pamf_phase3_restoration import feature_strong_trial as strong
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned


def test_configuration_changes_only_latent_weight_and_run_identity():
    c=strong.configuration(strong.ARM);old=strong.TEMPLATE
    changed={k for k in set(c)|set(old) if c.get(k)!=old.get(k)}
    assert changed=={'run_id','arm','latent_feature_weight','objective','matched_control','scientific_change'}
    assert c['latent_feature_weight']==1. and c['training_stages']==1
    assert c['initialization']=='scratch' and c['ancestor_checkpoints']==[]
    with pytest.raises(ValueError):strong.configuration('MSAT_FeatureCond')


def test_stronger_latent_loss_scales_gradient_once_and_keeps_teacher_detached():
    generator=torch.Generator().manual_seed(41)
    predicted=torch.randn(4,160,generator=generator,requires_grad=True)
    target=torch.randn(4,160,generator=generator,requires_grad=True);scale=torch.linspace(.02,.8,160)
    base,info=strong.BASE_LATENT(predicted,target,scale)
    loss,terms=strong.strong_latent_feature_loss(predicted,target,scale)
    before=torch.autograd.grad(base.sum(),predicted,retain_graph=True)[0]
    after=torch.autograd.grad(loss.sum(),predicted,retain_graph=True)[0]
    assert torch.allclose(loss,10*base,rtol=2e-7,atol=2e-7)
    assert torch.allclose(after,10*before,rtol=2e-6,atol=2e-7)
    assert all(torch.equal(v,info[k]) for k,v in terms.items())
    assert torch.autograd.grad(loss.sum(),target,allow_unused=True)[0] is None


def test_real_launcher_routes_to_new_single_arm_with_identical_scratch_model():
    code='''
import torch
from experiments.pamf_phase3_restoration import feature_strong_trial as s
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned
torch.set_num_threads(2)
old=scratch_feature_conditioned(2026092701)
s.install(torch.ones(160))
assert s.trial.RUN==s.RUN and s.trial.MODULE==s.MODULE and s.trial.ARMS==(s.ARM,)
assert s.trial.configuration(s.ARM)==s.configuration(s.ARM)
assert s.parent.latent_feature_loss is s.strong_latent_feature_loss
new=s.trial.factory(s.ARM)
assert all(torch.equal(v,old.state_dict()[k]) for k,v in new.state_dict().items())
assert sum(p.numel() for p in new.parameters())==4670178
'''
    subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=Path(__file__).parents[1],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)

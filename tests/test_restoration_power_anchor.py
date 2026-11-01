import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from experiments.pamf_phase3_restoration.power_anchor_trial import power_anchor_loss,configuration,ARM,TEMPLATE
from experiments.pamf_phase3_restoration.feature_conditioned import unit_power


def test_anchor_constrains_radial_scale_hidden_by_output_projection():
    clean=unit_power(torch.randn(2,2,256,generator=torch.Generator().manual_seed(51))).detach().requires_grad_(True)
    raw=(2*clean.detach()).requires_grad_(True)
    loss,metrics=power_anchor_loss({'raw_restored':raw},clean)
    assert torch.allclose(unit_power(raw),clean,atol=2e-7,rtol=2e-7)
    assert torch.allclose(loss,torch.full((2,),.1),atol=1e-7,rtol=0)
    assert torch.allclose(metrics['raw_complex_power'],torch.full((2,),4.),atol=1e-6,rtol=0)
    gradient=torch.autograd.grad(loss.sum(),raw,retain_graph=True)[0]
    assert (gradient*raw).sum()>0
    assert torch.autograd.grad(loss.sum(),clean,allow_unused=True)[0] is None


def test_anchor_radial_derivative_is_distinct_from_projected_waveform_error():
    generator=torch.Generator().manual_seed(52)
    raw=(2*unit_power(torch.randn(2,2,256,generator=generator))).requires_grad_(True)
    clean=unit_power(torch.randn(2,2,256,generator=generator))
    wave=(unit_power(raw)-clean).square().mean();anchor,_=power_anchor_loss({'raw_restored':raw},clean)
    tangent=torch.autograd.grad(wave,raw,retain_graph=True)[0];radial=torch.autograd.grad(anchor.sum(),raw)[0]
    assert abs(float((tangent*raw).sum().detach()))<1e-6
    assert abs(float((tangent*radial).sum()))<1e-5*float(tangent.norm()*radial.norm())+1e-9


def test_launcher_preserves_old_weights_guides_and_new_single_arm():
    c=configuration(ARM)
    assert {k for k in set(c)|set(TEMPLATE) if c.get(k)!=TEMPLATE.get(k)}=={
        'run_id','arm','objective','matched_control','scientific_change','raw_power_anchor_weight','raw_power_anchor_reference'}
    with pytest.raises(ValueError):configuration('MSAT_FeatureStrong')
    code='''
import torch
from experiments.pamf_phase3_restoration import power_anchor_trial as s
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned,latent_feature_loss
assert s.parent.OUTPUT_REGULARIZER is None
old=scratch_feature_conditioned(2026092701)
s.install(torch.ones(160))
assert s.parent.OUTPUT_REGULARIZER is s.power_anchor_loss
assert s.parent.latent_feature_loss is latent_feature_loss
assert s.trial.RUN==s.RUN and s.trial.MODULE==s.MODULE and s.trial.ARMS==(s.ARM,)
assert s.trial.configuration(s.ARM)==s.configuration(s.ARM)
new=s.trial.factory(s.ARM)
assert all(torch.equal(v,old.state_dict()[k]) for k,v in new.state_dict().items())
assert sum(p.numel() for p in new.parameters())==4670178
'''
    subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=Path(__file__).parents[1],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)

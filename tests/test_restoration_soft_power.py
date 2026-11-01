import pytest
import torch
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned,unit_power
from experiments.pamf_phase3_restoration.power_anchor_trial import power_anchor_loss
from experiments.pamf_phase3_restoration.soft_power import scratch_soft_power,soft_power_anchor
from experiments.pamf_phase3_restoration import soft_power_trial as trial

torch.set_num_threads(2)


def test_same_initial_weights_and_unprojected_function():
    soft=scratch_soft_power(2026092701).eval();control=scratch_feature_conditioned(2026092701).eval()
    assert sum(p.numel() for p in soft.parameters())==4670178
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in soft.state_dict().items())
    x=2*torch.randn(3,2,256)
    with torch.no_grad():
        actual,aux=soft(x,return_aux=True)
        assert torch.equal(actual,x) and actual is aux['raw_restored']
        control.output.weight.normal_(0,.001);soft.load_state_dict(control.state_dict())
        actual,aux=soft(x,return_aux=True);projected,old=control(x,return_aux=True)
        assert torch.equal(actual,old['raw_restored']) and torch.equal(aux['predicted_features'],old['predicted_features'])
        assert torch.equal(unit_power(actual),projected)
        torch.testing.assert_close(actual,torch.cat([soft(v[None]) for v in x]),atol=2e-6,rtol=2e-5)


def test_rms_anchor_same_formula_and_finite_at_zero():
    raw=torch.randn(3,2,256,requires_grad=True);clean=torch.randn_like(raw)
    new,metrics=soft_power_anchor(dict(raw_restored=raw),clean)
    old,old_metrics=power_anchor_loss(dict(raw_restored=raw),clean)
    torch.testing.assert_close(new,old,atol=1e-7,rtol=2e-5)
    for key in metrics:torch.testing.assert_close(metrics[key],old_metrics[key],atol=2e-6,rtol=2e-5)
    raw=torch.zeros(2,2,256,requires_grad=True)
    loss,_=soft_power_anchor(dict(raw_restored=raw),torch.ones_like(raw));loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(raw.grad).all()


def test_unit_norm_clean_does_not_require_unit_norm_conditional_mean():
    x=torch.zeros(2,2,256);x[0,0]=1.;x[1,1]=1.
    mean=x.mean(0,keepdim=True)
    assert mean.square().sum(1).mean()==.5
    raw_risk=(mean-x).square().sum(1).mean()
    sphere_risk=(unit_power(mean)-x).square().sum(1).mean()
    assert raw_risk<sphere_risk


def test_no_new_training_or_label_permissions():
    c=trial.configuration(trial.ARM)
    assert (c['epochs'],c['updates'],c['training_stages'])==(30,7410,1)
    assert c['initialization']=='scratch' and c['ancestor_checkpoints']==[]
    assert c['output_complex_power'] is None and c['unit_power_projection'] is False
    assert c['raw_power_anchor_weight']==.1 and c['checkpoint_selection']=='final30_only'
    assert all(c[k]==0 for k in ['TX_labels_read','V_reads','OE_reads','target_training_reads'])
    with pytest.raises(ValueError):trial.configuration('MSAT_PowerAnchor')


def test_queue_checks_technical_completion_without_result_access(tmp_path):
    from unittest.mock import patch
    parent=tmp_path/'runs'/trial.WAIT_PARENT;parent.mkdir(parents=True)
    (parent/'complete.json').write_text('{"status":"ARTIFACTS_COMPLETE"}',encoding='utf-8')
    with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}):trial.wait_for_parent(tmp_path)
    (parent/'terminal-train.json').write_text('{"exit_code":1}',encoding='utf-8')
    with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),pytest.raises(RuntimeError):trial.wait_for_parent(tmp_path)

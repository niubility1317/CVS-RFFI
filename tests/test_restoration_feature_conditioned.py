import pytest
import torch
from experiments.pamf_phase3_restoration.feature_conditioned import scratch_feature_conditioned,unit_power,latent_feature_loss
from experiments.pamf_phase3_restoration import feature_conditioned_trial as trial


def test_scratch_initialization_is_projected_identity_and_complete_regressor():
    torch.set_num_threads(2);G=scratch_feature_conditioned(7);x=torch.randn(3,2,256)
    y,aux=G(x,return_aux=True)
    torch.testing.assert_close(y,unit_power(x),atol=0,rtol=0)
    torch.testing.assert_close(y.square().sum(1).mean(1),torch.ones(3))
    assert aux['predicted_features'].shape==(3,160) and torch.equal(aux['raw_restored'],x)
    assert sum(p.numel() for p in G.parameters())==4670178
    assert G.initialization_report['ancestor_checkpoints']==[] and G.initialization_report['training_stages']==1


def test_projection_uses_complex_power_and_rejects_undefined_outputs():
    x=torch.ones(2,2,256,requires_grad=True);y=unit_power(x)
    torch.testing.assert_close(y,torch.full_like(y,2**-.5))
    (y*torch.randn_like(y)).sum().backward();assert torch.isfinite(x.grad).all()
    for bad in [torch.zeros(2,2,256),torch.full((2,2,256),float('nan'))]:
        with pytest.raises(ValueError,match='nonzero predicted IQ'):unit_power(bad)


def test_continuous_latent_teacher_is_detached_without_identity_labels_or_head():
    torch.manual_seed(10);prediction=torch.randn(2,160,requires_grad=True);target=torch.randn(2,160,requires_grad=True)
    loss,info=latent_feature_loss(prediction,target,torch.ones(160))
    loss.mean().backward()
    assert target.grad is None and torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum()>0
    assert info['latent_standardized_error'].shape==(2,)
    with pytest.raises(ValueError):latent_feature_loss(prediction,target[:1],torch.ones(160))


def test_auxiliary_feature_regression_trains_encoder_at_zero_IQ_output_initialization():
    torch.set_num_threads(2);G=scratch_feature_conditioned(11)
    _,aux=G(torch.randn(2,2,256),return_aux=True)
    loss,_=latent_feature_loss(aux['predicted_features'],torch.randn(2,160),torch.ones(160));loss.mean().backward()
    assert G.stem.weight.grad.abs().sum()>0 and G.feature_regressor[-1].weight.grad.abs().sum()>0
    assert torch.count_nonzero(G.output.weight)==0


def activated():
    G=scratch_feature_conditioned(12).eval()
    with torch.no_grad():
        G.output.weight.normal_(std=.003)
        for layer in G.conditions:layer.weight.normal_(std=.002)
    return G


def test_predicted_condition_and_all_four_FiLM_stages_receive_reconstruction_gradients():
    G=activated();y,aux=G(torch.randn(2,2,256),return_aux=True);aux['predicted_features'].retain_grad()
    (y*torch.randn_like(y)).sum().backward()
    assert aux['predicted_features'].grad.abs().sum()>0
    for layer in G.conditions:assert layer.weight.grad.abs().sum()>0 and torch.isfinite(layer.weight.grad).all()
    assert G.feature_regressor[-1].weight.grad.abs().sum()>0


def test_condition_and_projection_remain_per_record_and_permutation_independent():
    torch.set_num_threads(2);G=activated();x=torch.randn(3,2,256);p=torch.tensor([2,0,1])
    with torch.no_grad():
        y,a=G(x,return_aux=True);z,b=G(x[p],return_aux=True);single,c=G(x[:1],return_aux=True)
    torch.testing.assert_close(z,y[p],rtol=2e-5,atol=2e-5);torch.testing.assert_close(single,y[:1],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(b['predicted_features'],a['predicted_features'][p],rtol=2e-5,atol=2e-5)


def test_source_budget_and_predicted_only_condition_contract_reject_old_G_config(monkeypatch,tmp_path):
    c=trial.configuration(trial.ARM)
    for k in ['source_roles','source_draws','model_seed','data_seed','epochs','updates','batch','optimizer','lr','weight_decay']:
        assert c[k]==trial.TEMPLATE[k]
    assert c['ancestor_checkpoints']==[] and c['output_complex_power']==1. and c['continuous_feature_dimension']==160
    assert c['deployment_inputs']=='current_received_IQ_only/no_F0_inside_G'
    base=trial.trial
    for key in ['RUN','MODULE','ARMS','configuration','factory','train','feature_guide','gpu_choice']:monkeypatch.setattr(base,key,getattr(base,key))
    trial.install(torch.ones(160));path=tmp_path/'old.pt'
    torch.save(dict(config=trial.TEMPLATE,code_commit='new',epoch=30,updates=7410,state={}),path)
    with pytest.raises(ValueError,match='Exact scratch30'):base.load(path,trial.ARM,'new','cpu')

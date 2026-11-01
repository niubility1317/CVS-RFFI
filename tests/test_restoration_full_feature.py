import copy
import pytest
import torch
from experiments.pamf_phase3_restoration import full_feature_trial as full
from experiments.pamf_phase3_restoration.multiscale import scratch_multiscale


class FeaturesOnly:
    def features(self,x):return x.flatten(1)[:,:160]
    def classify_features(self,x):raise AssertionError('Head outputs not permitted for feature guidance')


def test_feature_gradients_are_unscaled_and_can_exceed_old_IQ_cap():
    torch.manual_seed(11);reader=FeaturesOnly();clean=torch.randn(2,2,256)
    restored=(clean+.001).requires_grad_();target=reader.features(clean).detach().requires_grad_()
    scale=torch.ones(160)*1e-5
    loss,info=full.full_feature_guide(reader,restored,clean,target,scale)
    actual=reader.features(restored)
    expected=.1*(1-torch.nn.functional.cosine_similarity(actual,target.detach(),dim=1,eps=1e-8)
        +((actual-target.detach()).square()/scale).mean(1))
    grad,target_grad=torch.autograd.grad(loss.sum(),(restored,target),allow_unused=True,retain_graph=True)
    reference=torch.autograd.grad(expected.sum(),restored)[0]
    torch.testing.assert_close(grad,reference,atol=0,rtol=0)
    IQ=2*(restored.detach()-clean).flatten(1).norm(dim=1)/(512*clean.square().mean((1,2)))
    assert (grad.flatten(1).norm(dim=1)>.5*IQ).all() and torch.equal(info['weight'],torch.ones(2))
    assert target_grad is None


def test_matched_source_configuration_preserves_data_model_budget_and_only_changes_cap():
    c=full.configuration(full.ARM);old=full.TEMPLATE
    for key in ['model_seed','data_seed','epochs','updates','batch','training_stages','initialization','ancestor_checkpoints',
            'architecture','source_roles','source_cache','source_draws','source_recipe','optimizer','lr','minimum_lr',
            'warmup_updates','weight_decay','gradient_clip','checkpoint_selection','feature_geometry_source','feature_geometry_code']:
        assert c[key]==old[key]
    assert c['per_record_feature_input_gradient_cap'] is False and c['global_parameter_gradient_clip']==1.
    with pytest.raises(ValueError):full.configuration('MSAT_VarF')
    with pytest.raises(ValueError):full.full_feature_guide(FeaturesOnly(),torch.zeros(2,2,256),None,torch.zeros(1,160),torch.ones(160))


def test_same_scratch_initial_state_and_old_checkpoint_config_rejected(monkeypatch,tmp_path):
    torch.set_num_threads(2);trial=full.trial
    for name in ['RUN','MODULE','ARMS','configuration','feature_guide','gpu_choice']:
        monkeypatch.setattr(trial,name,getattr(trial,name))
    full.install(torch.ones(160));new=trial.factory(full.ARM);old=scratch_multiscale(2026092701)
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in new.state_dict().items())
    path=tmp_path/'old.pt';state={k:v.detach().clone() for k,v in new.state_dict().items()}
    torch.save(dict(config=copy.deepcopy(full.TEMPLATE),code_commit='new-code',epoch=30,updates=7410,state=state),path)
    with pytest.raises(ValueError,match='Exact scratch30'):trial.load(path,full.ARM,'new-code','cpu')

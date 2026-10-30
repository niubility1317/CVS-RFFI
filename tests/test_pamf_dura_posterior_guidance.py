from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_efficiency.posterior_loss import frozen_posterior_loss
from experiments.pamf_dura_direct_train import runner as base,f_guided,posterior_guided


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(160,6,bias=False)
        with torch.no_grad():self.head.weight.zero_();self.head.weight[:,:6].copy_(torch.eye(6))
    def features(self,x):return x.flatten(1)[:,:160]
    def classify_features(self,x):return self.head(x)


def test_posterior_loss_ignores_common_logit_shift_even_when_features_differ():
    reader=Reader().eval().requires_grad_(False)
    x=torch.zeros(2,2,256);reference=torch.zeros(2,160)
    reference[:,:6]=torch.arange(6);x.flatten(1)[:,:6]=reference[:,:6]+3
    assert not torch.equal(reader.features(x),reference)
    torch.testing.assert_close(frozen_posterior_loss(reader,x,reference),torch.zeros(2),atol=2e-7,rtol=0)


def test_gradient_matches_soft_distribution_difference_and_never_updates_teacher():
    reader=Reader().eval().requires_grad_(False)
    x=torch.zeros(1,2,256);x[0,0,1]=3;x.requires_grad_()
    reference=torch.zeros(1,160);reference[0,0]=3;reference.requires_grad_()
    loss=frozen_posterior_loss(reader,x,reference);loss.sum().backward()
    expected=torch.softmax(reader.classify_features(reader.features(x.detach())),1)-torch.softmax(reader.classify_features(reference.detach()),1)
    assert loss.item()>0
    torch.testing.assert_close(x.grad.flatten(1)[:,:6],expected)
    assert reference.grad is None and all(p.grad is None for p in reader.parameters())


def test_trainable_or_training_F_is_rejected():
    reader=Reader().eval().requires_grad_(False);x=torch.zeros(1,2,256);reference=torch.zeros(1,160)
    reader.train()
    with pytest.raises(ValueError,match='Fixed eval F'):frozen_posterior_loss(reader,x,reference)
    reader.eval().requires_grad_(True)
    with pytest.raises(ValueError,match='Fixed eval F'):frozen_posterior_loss(reader,x,reference)


def test_posterior_recipe_keeps_single_stage_budget_and_routes_all_children():
    keys=('RUN','configuration','command','learning_rate','direct_objective','save_model','load_model','smoke','train','training_pair')
    original={key:getattr(base,key) for key in keys};objective=f_guided.objective;metric=f_guided.GUIDANCE_METRIC
    try:
        posterior_guided.configure();config=base.configuration()
        assert (config['epochs'],config['updates'],config['training_stages'])==(20,4940,1)
        assert config['training_F_access'] and not config['hard_pseudo_TX_labels'] and config['F_posterior_weight']==.1
        assert not config['ancestor_checkpoints'] and config['F_teacher']['source_commit']==f_guided.F_SOURCE_COMMIT
        assert config['F_guidance_metric']==f_guided.GUIDANCE_METRIC=='F_posterior_KL'
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==posterior_guided.MODULE for mode in ('worker','score'))
    finally:
        for key,value in original.items():setattr(base,key,value)
        f_guided.objective=objective;f_guided.GUIDANCE_METRIC=metric

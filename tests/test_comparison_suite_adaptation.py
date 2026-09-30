"""Meaningful objective/update/protocol checks on tiny networks (no target truth)."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from comparison_suite.adaptation import (
    RadioNetADAHead, RadioNetADANet, fit_adaptation, radionet_ada_batch_step,
)
from paper_reproduction.cvs_aligned.supervised_da import mrior_sda_batch_step


class TinyModel(nn.Module):
    def __init__(self, width=4, classes=3):
        super().__init__()
        self.encoder = nn.Linear(width, 5)
        self.classifier = nn.Linear(5, classes)
        self.estimate_network = nn.Linear(5, 1)
        self.domain_classifier = nn.Linear(5, 2)

    def forward(self, x, grl_lambda=1., return_activations=False):
        from baselines.common.grl import gradient_reverse
        z = self.encoder(x)
        out = {"features": z, "global_features": z, "local_features": z,
               "tx_logits": self.classifier(z), "logits": self.classifier(z),
               "estimate_logits": self.estimate_network(z),
               "domain_logits": self.domain_classifier(gradient_reverse(z, grl_lambda))}
        if return_activations:
            out["activations"] = [z]
        return out


@pytest.fixture
def task():
    torch.set_num_threads(1)
    torch.manual_seed(92)
    model = TinyModel()
    support_x = torch.randn(6, 4)
    support_y = torch.tensor([2, 2, 0, 0, 1, 1])
    source_x = torch.randn(12, 4)
    source_y = torch.tensor([0, 1, 2]*4)
    return model, support_x, support_y, source_x, source_y


@pytest.mark.parametrize("method", ["mrior_sda", "dadda_sda", "twostage_sda", "sft", "feature_separation_ft", "coral_sft"])
def test_native_update_score_allclasses_query_readonly(method, task):
    model, sx, sy, rx, ry = task
    original = copy.deepcopy(model.state_dict())
    logs = []
    state = fit_adaptation(method, model, sx, sy, [2, 0, 1],
        {"epochs": 1, "dann_epochs": 1, "lmmd_epochs": 1, "finetune_epochs": 1,
         "estimate_steps": 2, "seed": 3, "logger": logs.append}, rx, ry)
    assert logs and state.classes == (2, 0, 1)
    assert any(not torch.equal(original[k], state.model.base.state_dict()[k]) for k in original)
    assert all(torch.equal(original[k], model.state_dict()[k]) for k in original)
    before = copy.deepcopy(state.model.state_dict())
    query = torch.randn(5, 4)
    together = state.score(query)
    separated = torch.cat([state.score(query[i:i+1]) for i in range(len(query))])
    assert together.shape == (5, 3)
    assert torch.allclose(together, separated, atol=1e-6)
    assert all(torch.equal(before[k], state.model.state_dict()[k]) for k in before)
    assert state.metadata["query_used_for_fit"] is False
    assert state.resources["trainable_parameters_during_fit"] > 0


@pytest.mark.parametrize("method", ["ncm", "protonet_cda", "ridge", "linear", "adabn"])
def test_controls_support_only_reproducible_and_class_mapping(method, task):
    model, sx, sy, _, _ = task
    before = copy.deepcopy(model.state_dict())
    a = fit_adaptation(method, model, sx, sy, [2, 0, 1], {"epochs": 2, "seed": 88})
    b = fit_adaptation(method, model, sx, sy, [2, 0, 1], {"epochs": 2, "seed": 88})
    assert a.score(sx).shape == (6, 3)
    assert torch.equal(a.score(sx), b.score(sx))
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)


def test_refuses_missing_replay_and_unregistered_support(task):
    model, sx, sy, _, _ = task
    with pytest.raises(ValueError, match="requires explicitly permitted source"):
        fit_adaptation("mrior_sda", model, sx, sy, [0, 1, 2], {"epochs": 1})
    with pytest.raises(ValueError, match="outside registered"):
        fit_adaptation("ncm", model, sx, sy, [0, 1], {})


def test_mrior_estimate_network_ascends_not_joint_minimizes():
    torch.manual_seed(15)
    model = TinyModel()
    sx, tx = torch.randn(6, 4), torch.randn(6, 4)+2
    y = torch.arange(6)%3
    opt_t = torch.optim.SGD(model.estimate_network.parameters(), lr=.001)
    opt_ec = torch.optim.SGD(list(model.encoder.parameters())+list(model.classifier.parameters()), lr=0.)
    from paper_reproduction.mitigating_receiver_impact_da.losses import dv_kl_domain_alignment
    def dv():
        return dv_kl_domain_alignment(model(sx)["estimate_logits"], model(tx)["estimate_logits"])
    before = float(dv().detach())
    result = mrior_sda_batch_step(model, sx, y, tx, y, optimizer_t=opt_t, optimizer_ec=opt_ec, estimate_steps=2)
    assert float(dv().detach()) >= before-1e-6
    assert torch.allclose(result["estimate_loss"], -result["estimate_zeta"])
    assert all(p.requires_grad for p in model.estimate_network.parameters())


def test_dadda_dynamic_factor_logged_and_true_support_lmmd(task):
    model, sx, sy, rx, ry = task
    state = fit_adaptation("dadda_sda", model, sx, sy, [0, 1, 2], {"epochs": 1}, rx, ry)
    row = state.history[0]
    assert 0 <= row["alpha"] <= 1
    assert row["mmd"] >= 0 and row["lmmd"] >= 0
    assert abs(row["dynamic_joint"]-((1-row["alpha"])*row["mmd"]+row["alpha"]*row["lmmd_sum"])) < 1e-5


def test_radio_alternating_updates_both_networks_and_knn_freezes(task):
    model, sx, sy, rx, ry = task
    torch.manual_seed(5)
    disc = RadioNetADAHead(1)
    ec_before = copy.deepcopy(model.state_dict())
    d_before = copy.deepcopy(disc.state_dict())
    a = torch.optim.SGD(model.parameters(), lr=.01)
    b = torch.optim.SGD(disc.parameters(), lr=.01)
    result = radionet_ada_batch_step(model, disc, rx, ry, sx, a, b)
    assert result["ec_gradient_norm"] > 0 and result["discriminator_gradient_norm"] > 0
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in ec_before.items())
    assert any(not torch.equal(v, disc.state_dict()[k]) for k, v in d_before.items())
    state = fit_adaptation("radionet_ada_knn", model, sx, sy, [0, 1, 2], {"iterations": 2}, rx, ry)
    assert state.decision == "knn" and state.knn_k == 2
    assert state.metadata["support_ft_epochs"] == 0
    assert all(not p.requires_grad for p in state.model.parameters())
    assert state.score(sx).shape == (len(sx), 3)


def test_radio_ada_topology_distinct_from_ordinary_df():
    torch.set_num_threads(1)
    model = RadioNetADANet(6)
    model.eval()
    output = model(torch.randn(2, 2, 256))
    assert output["features"].shape == (2, 64)
    assert output["tx_logits"].shape == (2, 6)
    assert sum(isinstance(m, nn.BatchNorm1d) for m in model.modules()) == 7


def test_radio_alternating_freezes_other_network_bn_buffers():
    torch.set_num_threads(1)
    model, discriminator = RadioNetADANet(2), RadioNetADAHead(1)
    sx, tx, sy = torch.randn(2, 2, 256), torch.randn(2, 2, 256), torch.tensor([0, 1])
    opt_ec = torch.optim.SGD(model.parameters(), lr=.001)
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=.001)
    radionet_ada_batch_step(model, discriminator, sx, sy, tx, opt_ec, opt_d)
    # Each network saw two train-mode forwards, but only its own training step
    # persists BN updates. Both frozen-step BN updates were restored.
    assert all(int(m.num_batches_tracked) == 1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    assert all(int(m.num_batches_tracked) == 1 for m in discriminator.modules() if isinstance(m, nn.BatchNorm1d))


def test_query_is_not_an_argument_to_fit():
    import inspect
    assert not any("query" in key for key in inspect.signature(fit_adaptation).parameters)


def test_cosine_knn_zero_distance_and_kshot(task):
    model, sx, sy, rx, ry = task
    state = fit_adaptation("radionet_ada_knn", model, sx, sy, [0, 1, 2], {"iterations": 0}, rx, ry)
    scores = state.score(sx)
    assert torch.equal(scores.argmax(1), sy)
    assert torch.isfinite(scores).all()

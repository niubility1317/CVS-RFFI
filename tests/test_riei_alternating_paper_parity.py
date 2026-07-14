import copy

import torch
from torch import nn


class _TinyRIEI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fed = nn.Linear(3, 4, bias=False)
        self.ec = nn.Linear(2, 2, bias=False)
        self.rc = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        z = self.fed(x)
        z_e, z_r = z[:, :2], z[:, 2:]
        return {
            "z_e": z_e,
            "z_r": z_r,
            "emitter_logits": self.ec(z_e),
            "receiver_logits": self.rc(z_r),
            "cross_emitter_logits": self.ec(z_r),
            "cross_receiver_logits": self.rc(z_e),
        }


def test_riei_disentanglement_step_updates_fed_but_not_frozen_classifiers():
    from baselines.riei_fd.train import alternating_training_step

    torch.manual_seed(7)
    model_with_dis = _TinyRIEI()
    model_ce_only = copy.deepcopy(model_with_dis)
    batch = {
        "iq": torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),
        "label": torch.tensor([0, 1, 0, 1]),
        "receiver_target": torch.tensor([0, 0, 1, 1]),
    }

    def run(model, lambda_mi, lambda_ie):
        optimizer_all = torch.optim.Adam(list(model.ec.parameters()) + list(model.rc.parameters()), lr=0.01)
        optimizer_fed = torch.optim.Adam(model.fed.parameters(), lr=0.01)
        alternating_training_step(
            model,
            batch,
            optimizer_all,
            optimizer_fed,
            lambda_mi=lambda_mi,
            lambda_ie=lambda_ie,
            ce_reduction="sum",
            mi_reduction="sum",
            ie_reduction="sum",
        )
        return optimizer_all, optimizer_fed

    optimizer_classifiers, optimizer_fed = run(model_with_dis, 1.2, 1.2)
    run(model_ce_only, 0.0, 0.0)

    assert torch.equal(model_with_dis.ec.weight, model_ce_only.ec.weight)
    assert torch.equal(model_with_dis.rc.weight, model_ce_only.rc.weight)
    assert not torch.equal(model_with_dis.fed.weight, model_ce_only.fed.weight)
    assert int(optimizer_classifiers.state[model_with_dis.ec.weight]["step"]) == 1
    assert int(optimizer_fed.state[model_with_dis.fed.weight]["step"]) == 2

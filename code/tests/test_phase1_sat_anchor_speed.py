import torch
from torch import nn

from SSDG import train_ssdg


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, **_kwargs):
        self.calls += 1
        flat = x.flatten(1)
        return {
            "tx_logits": flat[:, :3],
            "z_id": flat[:, :4],
            "z_dom": flat[:, -4:],
        }


def test_sat_anchor_student_clean_and_satellite_rows_share_one_forward():
    model = _CountingModel()
    clean = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8)
    satellite = clean + 100.0
    satellite_indices = torch.tensor([1, 3])

    outputs = train_ssdg._forward_sat_anchor_student_views(
        model,
        clean,
        satellite,
        torch.tensor([0, 1, 0, 1]),
        satellite_indices=satellite_indices,
        grl_lambda=0.1,
        detach_backbone=False,
    )

    assert model.calls == 1
    assert outputs["clean"]["z_id"].shape[0] == 4
    assert outputs["satellite"]["z_id"].shape[0] == 2
    assert outputs["satellite_indices"].tolist() == [1, 3]


def test_disabled_satellite_objectives_do_not_add_satellite_rows():
    model = _CountingModel()
    clean = torch.zeros(4, 2, 8)

    outputs = train_ssdg._forward_sat_anchor_student_views(
        model,
        clean,
        None,
        torch.tensor([0, 1, 0, 1]),
        satellite_indices=torch.empty(0, dtype=torch.long),
        grl_lambda=0.1,
        detach_backbone=False,
    )

    assert model.calls == 1
    assert outputs["satellite"] is None


def test_all_disabled_u_objectives_skip_the_student_forward_entirely():
    model = _CountingModel()
    clean = torch.zeros(4, 2, 8)

    outputs = train_ssdg._forward_sat_anchor_student_views(
        model,
        clean,
        None,
        torch.tensor([0, 1, 0, 1]),
        satellite_indices=torch.empty(0, dtype=torch.long),
        grl_lambda=0.0,
        detach_backbone=False,
        include_clean=False,
    )

    assert model.calls == 0
    assert outputs["clean"] is None
    assert outputs["satellite"] is None


def test_satellite_only_objective_forwards_only_selected_satellite_rows():
    model = _CountingModel()
    clean = torch.zeros(4, 2, 8)
    satellite = clean + 1.0

    outputs = train_ssdg._forward_sat_anchor_student_views(
        model,
        clean,
        satellite,
        torch.tensor([0, 1, 0, 1]),
        satellite_indices=torch.tensor([1, 3]),
        grl_lambda=0.0,
        detach_backbone=False,
        include_clean=False,
    )

    assert model.calls == 1
    assert outputs["clean"] is None
    assert outputs["satellite"]["tx_logits"].shape[0] == 2


def test_pair_interval_two_runs_full_pair_only_on_every_second_step():
    assert train_ssdg._sat_anchor_pair_step(1, 2) is False
    assert train_ssdg._sat_anchor_pair_step(2, 2) is True
    assert train_ssdg._sat_anchor_pair_step(3, 2) is False
    assert train_ssdg._sat_anchor_pair_step(4, 2) is True
    assert train_ssdg._sat_anchor_pair_step(7, 1) is True

import torch
import pytest
from torch import nn

from experiments.pamf_physics_branch.objective import PhysicalPL, physical_directions, select_indices, select_u_candidates


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        enc = self.id_backbone.encoder = nn.Module()
        enc.core = nn.Module()
        core = enc.core.id_backbone = nn.Module()
        for name in ('t_proj', 'f_proj', 'freq_stats_proj', 'pa_proj', 'pa_stats_proj'):
            setattr(core, name, nn.Linear(3, 3, bias=False))
        enc.sequence_pool = nn.Linear(3, 3, bias=False)
        enc.response_projection = nn.Linear(3, 3, bias=False)
        self._cd_U_seen = self._cd_U_accepted = 0
        self._feature_l_steps = 1

    def forward(self, x):
        return self.forward_identity_only(x, None)['tx_logits']

    def forward_identity_only(self, x, domain_labels):
        enc = self.id_backbone.encoder
        core = enc.core.id_backbone
        z = torch.stack((x[:, 0, 0], x[:, 0, 1], x[:, 1, 1]), dim=1)
        parts = [core.t_proj(z), core.f_proj(z), core.freq_stats_proj(z),
                 core.pa_proj(z), core.pa_stats_proj(z), enc.sequence_pool(z), enc.response_projection(z)]
        feature = sum(parts)
        return {'tx_logits': torch.stack((20 * x[:, 0, 0] + feature[:, 0],
                                          -20 * x[:, 0, 0] - feature[:, 0]), dim=1)}


def test_measurement_restores_all_modes_and_global_rng():
    torch.manual_seed(7)
    model = TinyStudent().train()
    model.id_backbone.encoder.sequence_pool.eval()  # mixed modes must be restored
    x = torch.randn(5, 2, 256)
    before_rng = torch.random.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    first, norms = physical_directions(model, x, 17)
    second, _ = physical_directions(model, x, 17)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert set(first) == {'time', 'frequency', 'pa', 'path', 'reference'}
    assert all(torch.equal(first[k], second[k]) and not first[k].requires_grad for k in first)
    assert all(bool(torch.isfinite(v).all()) for v in norms.values())


def test_selection_does_not_touch_global_rng_and_obeys_acceptance():
    mask = torch.tensor([True, False, True, True, False, True])
    before = torch.random.get_rng_state().clone()
    selected = select_indices(mask, 3, 123)
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(selected) == 3 and bool(mask[selected].all())
    assert torch.equal(selected, select_indices(mask, 3, 123))


def test_u_physical_candidates_do_not_depend_on_model_acceptance():
    a = torch.tensor([True, False, True, False, True, False])
    b = ~a
    selected_a = select_u_candidates(a, 4, 123)
    selected_b = select_u_candidates(b, 4, 123)
    assert torch.equal(selected_a, selected_b)
    assert len(selected_a) == 4
    assert int(a[selected_a].sum()) + int(b[selected_b].sum()) == 4


def test_physical_pl_preserves_raw_pl_and_uses_only_accepted_hidden_u():
    torch.manual_seed(8)
    model = TinyStudent().train()
    x = torch.randn(4, 2, 256) * .05
    x[:, 0, 0] = torch.tensor([1., 1., -1., -1.])
    meta = {'rx_i': torch.tensor([1, 2, 1, 2]), 'day_i': torch.ones(4, dtype=torch.long),
            'branch_record_id': torch.arange(4), 'tx_label_visible': torch.zeros(4, dtype=torch.bool)}
    model._cd_U_iter = iter([(x, torch.full((4,), -1), torch.ones(4, dtype=torch.long), meta)])
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
         'source_rxs': [1, 2], 'source_days': [1], 'physical_relation': {'beta': .5, 'u_weight': .05}}
    objective = PhysicalPL(c)
    loss = objective.loss(model, 'cpu')
    loss.backward()
    state = objective.state_dict()
    assert state['seen'] == state['accepted'] == 4 and state['U_truth_read'] is False
    assert state['physical_U']['no_grad_forward_packets'] == 8
    assert objective.metrics['physical_U_valid_anchors'] == 4
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.parameters())


def test_physical_u_rejects_visible_transmitter_labels():
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.}}
    objective = PhysicalPL(c)
    batch = (torch.zeros(2, 2, 256), torch.full((2,), -1), torch.zeros(2),
             {'tx_label_visible': torch.tensor([False, True])})
    with pytest.raises(ValueError, match='hidden transmitter'):
        objective.validate_unlabeled_batch(batch)

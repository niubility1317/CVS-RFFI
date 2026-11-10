import torch
from torch import nn

from experiments.pamf_branch_training.relation import branch_relation_loss, capture_branches
from experiments.pamf_branch_training.source_ids import indexed_dataset_type
from experiments.pamf_branch_training.objective import BranchPL


def test_cross_receiver_relation_reaches_each_branch_and_excludes_rejected():
    torch.manual_seed(7)
    features = {name: torch.randn(6, 7, requires_grad=True) for name in ('time', 'frequency', 'pa', 'path')}
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    receivers = torch.tensor([1, 2, 1, 2, 3, 3])
    accepted = torch.tensor([True, True, True, True, False, False])
    loss, counts = branch_relation_loss(features, labels, receivers, torch.arange(6), accepted, single_receiver=False)
    loss.backward()
    assert torch.isfinite(loss)
    assert counts['cross_receiver_anchors'] == 4
    assert counts['same_receiver_anchors'] == 0
    for value in features.values():
        assert value.grad[:4].norm() > 0
        assert torch.equal(value.grad[4:], torch.zeros_like(value.grad[4:]))


def test_multi_receiver_does_not_silently_make_same_rx_positives():
    features = {'time': torch.randn(4, 5, requires_grad=True)}
    labels = torch.tensor([0, 0, 1, 1])
    receivers = torch.tensor([1, 1, 2, 2])
    loss, counts = branch_relation_loss(features, labels, receivers, torch.arange(4),
                                        torch.ones(4, dtype=torch.bool), single_receiver=False)
    assert loss.item() == 0
    assert counts['same_receiver_anchors'] == counts['cross_receiver_anchors'] == 0


def test_single_receiver_uses_distinct_records_and_never_self_pairs():
    features = {'pa': torch.randn(4, 5, requires_grad=True)}
    labels = torch.tensor([0, 0, 1, 1])
    ids = torch.tensor([8, 8, 9, 10])
    loss, counts = branch_relation_loss(features, labels, torch.ones(4, dtype=torch.long), ids,
                                        torch.ones(4, dtype=torch.bool), single_receiver=True)
    assert torch.isfinite(loss)
    assert counts['same_receiver_anchors'] == 2
    assert counts['cross_receiver_anchors'] == 0


def test_no_eligible_pairs_gives_finite_zero_and_zero_gradients():
    feature = torch.randn(3, 4, requires_grad=True)
    loss, counts = branch_relation_loss({'pa': feature}, torch.tensor([0, 1, 2]),
                                        torch.tensor([1, 2, 3]), torch.arange(3),
                                        torch.zeros(3, dtype=torch.bool), single_receiver=False)
    loss.backward()
    assert loss.item() == 0
    assert counts['cross_receiver_anchors'] == 0
    assert torch.equal(feature.grad, torch.zeros_like(feature))


def test_capture_reconstructs_same_pass_branches_and_removes_hooks():
    class Paths(nn.Module):
        def __init__(self):
            super().__init__()
            self.t_proj = nn.Linear(4, 4)
            self.f_proj = nn.Linear(4, 4)
            self.freq_stats_proj = nn.Linear(4, 4)
            self.pa_proj = nn.Linear(4, 4)
            self.pa_stats_proj = nn.Linear(4, 4)

    core = Paths()
    sequence = nn.Linear(4, 4)
    x = torch.randn(5, 4)
    with capture_branches(core, sequence) as packet:
        for module in (core.t_proj, core.f_proj, core.freq_stats_proj, core.pa_proj, core.pa_stats_proj, sequence):
            module(x)
    assert torch.allclose(packet['frequency'], core.f_proj(x) + core.freq_stats_proj(x))
    assert torch.allclose(packet['pa'], core.pa_proj(x) + 1.25 * core.pa_stats_proj(x))
    assert all(len(m._forward_hooks) == 0 for m in (core.t_proj, core.f_proj, core.pa_proj, sequence))


def test_original_network_includes_reference_projection():
    core = nn.Module()
    for name in ('t_proj', 'f_proj', 'freq_stats_proj', 'pa_proj', 'pa_stats_proj'):
        setattr(core, name, nn.Linear(2, 2))
    sequence, reference = nn.Linear(2, 2), nn.Linear(2, 2)
    x = torch.randn(3, 2)
    with capture_branches(core, sequence, reference) as packet:
        for module in (core.t_proj, core.f_proj, core.freq_stats_proj, core.pa_proj, core.pa_stats_proj, sequence, reference):
            module(x)
    assert torch.allclose(packet['reference'], reference(x))
    assert not reference._forward_hooks


def test_capture_cleans_hooks_on_forward_failure():
    core = nn.Module()
    for name in ('t_proj', 'f_proj', 'freq_stats_proj', 'pa_proj', 'pa_stats_proj'):
        setattr(core, name, nn.Linear(2, 2))
    sequence = nn.Linear(2, 2)
    try:
        with capture_branches(core, sequence):
            raise RuntimeError('failed forward')
    except RuntimeError as error:
        assert str(error) == 'failed forward'
    assert not core.t_proj._forward_hooks and not sequence._forward_hooks


def test_source_index_adds_opaque_equality_key_without_touching_hidden_labels():
    class Source(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros(2, 256), -1, 1, {'rx_i': 2}

    source = indexed_dataset_type(Source)()
    first, second = source[0], source[1]
    assert first[1] == second[1] == -1
    assert first[3]['branch_record_id'] == 0
    assert second[3]['branch_record_id'] == 1


def test_branch_pl_executes_on_hidden_u_with_original_student_forward():
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('t_proj', 'f_proj', 'freq_stats_proj', 'pa_proj', 'pa_stats_proj'):
                setattr(self, name, nn.Linear(3, 3, bias=False))

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone = nn.Module()
            self.id_backbone.encoder = nn.Module()
            self.id_backbone.encoder.core = nn.Module()
            self.id_backbone.encoder.core.id_backbone = Core()
            self.id_backbone.encoder.sequence_pool = nn.Linear(3, 3, bias=False)
            self._cd_U_seen = self._cd_U_accepted = 0

        def forward(self, x):
            return torch.stack((10 * x[:, 0], -10 * x[:, 0]), dim=1)

        def forward_identity_only(self, x, domain_labels):
            core = self.id_backbone.encoder.core.id_backbone
            path = self.id_backbone.encoder.sequence_pool
            parts = [core.t_proj(x), core.f_proj(x), core.freq_stats_proj(x),
                     core.pa_proj(x), core.pa_stats_proj(x), path(x)]
            z = sum(parts)
            return {'tx_logits': torch.stack((z[:, 0], -z[:, 0]), dim=1)}

    torch.manual_seed(4)
    student = TinyStudent()
    x = torch.tensor([[1., .2, .1], [1., -.3, .4], [-1., .6, .2], [-1., -.4, .3]])
    metadata = {'rx_i': torch.tensor([1, 2, 1, 2]), 'branch_record_id': torch.arange(4)}
    batch = (x, torch.full((4,), -1), torch.tensor([1, 2, 1, 2]), metadata)
    student._cd_U_iter = iter([batch])
    objective = BranchPL({'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
                          'source_rxs': [1, 2], 'branch_relation': {'u_weight': .05}})
    loss = objective.loss(student, 'cpu')
    loss.backward()
    assert objective.metrics['branch_U_cross_anchors'] == 4
    assert objective.state_dict()['U_truth_read'] is False
    assert student._cd_U_seen == 4
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in student.parameters())

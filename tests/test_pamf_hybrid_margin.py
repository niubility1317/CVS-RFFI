import torch

from experiments.pamf_physics_tri.hybrid_margin import capped_margin_loss, source_margin_cap


def test_capped_margin_stops_when_satisfied_and_ignores_unaccepted_packets():
    raw = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], requires_grad=True)
    augmented = torch.tensor([[1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    weights = torch.eye(2, requires_grad=True)
    pseudo = torch.tensor([0, 0, 0])
    accepted = torch.tensor([True, True, False])
    loss, diagnostics = capped_margin_loss(raw, augmented, weights, pseudo, accepted, .5)
    torch.testing.assert_close(loss, torch.tensor(.5))
    assert diagnostics['violating'].tolist() == [False, True, False]
    loss.backward()
    assert raw.grad is None and weights.grad is None
    assert augmented.grad[0].abs().sum() == 0
    assert augmented.grad[1].abs().sum() > 0
    assert augmented.grad[2].abs().sum() == 0


def test_negative_raw_margin_has_zero_reference_but_corrects_negative_augmented_margin():
    raw = torch.tensor([[0., 1.]])
    augmented = torch.tensor([[0., 1.]], requires_grad=True)
    loss, diagnostics = capped_margin_loss(raw, augmented, torch.eye(2), torch.tensor([0]), torch.tensor([True]), .5)
    torch.testing.assert_close(diagnostics['reference'], torch.tensor([0.]))
    torch.testing.assert_close(loss, torch.tensor(1.))


def test_source_cap_uses_only_correct_labeled_examples():
    features = torch.tensor([[1., 0.], [.8, .6], [0., 1.]])
    labels = torch.tensor([0, 0, 0])
    cap, correct = source_margin_cap(features, labels, torch.eye(2))
    assert correct == 2
    torch.testing.assert_close(cap, torch.tensor(.6))


def test_margin_objective_uses_clean_raw_input_and_restores_training_mode():
    from experiments.pamf_physics_tri.hybrid_margin_objective import HybridMargin

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.id_backbone = torch.nn.Module()
            self.core.id_backbone.cls_head = torch.nn.Linear(160, 2, bias=False)
            with torch.no_grad():
                self.core.id_backbone.cls_head.weight.zero_()
                self.core.id_backbone.cls_head.weight[:, :2].copy_(torch.eye(2))

        def features(self, x):
            return torch.nn.functional.pad(x, (0, 158))

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone = torch.nn.Module()
            self.id_backbone.encoder = Encoder()

        def forward_identity_only(self, x, domain_labels=None):
            encoder = self.id_backbone.encoder
            h = encoder.features(x)
            return {'tx_logits': encoder.core.id_backbone.cls_head(h)}

    student = Student().train()
    objective = object.__new__(HybridMargin)
    objective.c = {'hybrid_auxiliary_weight': 1.,
                   'hard_pl_joint_contrast': {'weight': 1., 'CE_normalizer': 1.}}
    objective.margin_cap = .5
    objective.settings = {'weight': 1.}
    objective.aux_student = student
    objective.raw_aux_view = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    objective.aux_view = torch.tensor([[1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    objective.aux_domain = None
    objective.hybrid_labels = (torch.tensor([0, 0, 0]), torch.tensor([True, True, False]))
    objective.trace_batch = {}
    objective.metrics = {}
    objective.hybrid_calls = objective.hybrid_anchors = 0
    objective.margin_violations = objective.margin_accepted = 0
    traces = []
    objective.write_trace = traces.append
    # Contradictory baseline features expose accidental reuse of noisy baseline views.
    views = [torch.tensor([[0., 1.], [0., 1.], [0., 1.]])] * 2
    loss = objective.additional_fusion_loss(views, objective.hybrid_labels[1])
    torch.testing.assert_close(loss, torch.tensor(.5))
    loss.backward()
    assert student.training
    assert student.id_backbone.encoder.core.id_backbone.cls_head.weight.grad is None
    assert traces[0]['margin_violating'] == [False, True, False]
    assert objective.margin_violations == 1 and objective.margin_accepted == 2


def test_margin_matrix_matches_fusion_only_selected_seeds_without_parameter_sweep():
    from experiments.pamf_physics_tri import hybrid_margin_plan as margin, hybrid_structure_plan as structure

    expected = {(r['setting'], r['model_seed']) for r in structure.candidate_rows() if r['structure_arm'] == 'C'}
    rows = margin.candidate_rows()
    assert len(rows) == 4
    assert {(r['setting'], r['model_seed']) for r in rows} == expected
    assert all(r['structure_arm'] == 'C' and r['auxiliary_weight'] == .1 for r in rows)

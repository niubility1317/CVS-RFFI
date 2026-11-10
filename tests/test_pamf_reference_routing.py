import torch
from types import SimpleNamespace

from experiments.pamf_physics_branch.geometry import soft_distances
from experiments.pamf_physics_branch.reference_routing import stable_reference_directions, route_stable_physical_directions


def test_reference_routing_retains_stable_directions_and_zeroes_only_changed_routes():
    vectors = torch.tensor([[1., 0.], [0., 1.], [2**-.5, 2**-.5]])
    raw = torch.tensor([1, 2, 3])
    perturbed = torch.tensor([1, 1, 3])
    out = stable_reference_directions(vectors, raw, perturbed)
    assert torch.equal(out, torch.tensor([[1., 0.], [0., 0.], [2**-.5, 2**-.5]]))
    assert torch.equal(vectors[1], torch.tensor([0., 1.]))


def test_route_corrected_distances_keep_all_pairs_and_physical_lower_bounds():
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    directions = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    masked = stable_reference_directions(directions, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2]))
    ordinary, corrected = soft_distances(features, masked, beta=.5)
    assert corrected.shape == (3, 3)
    assert torch.all(corrected <= ordinary + 1e-6)
    assert torch.all(corrected >= .5 * ordinary - 1e-6)
    assert corrected[1, 2] == ordinary[1, 2]
    assert corrected[0, 1] >= .75 * ordinary[0, 1] - 1e-6


def test_live_wrapper_masks_only_reference_and_counts_every_selected_packet():
    class Response:
        def __init__(self):
            self.calls = 0

        def components(self, inputs):
            self.calls += 1
            result = [0, 1] if self.calls == 1 else [0, 2]
            return {'selected_delay_samples': torch.tensor(result, device=inputs.device)}

    response = Response()
    model = SimpleNamespace(id_backbone=SimpleNamespace(encoder=SimpleNamespace(response=response)))
    features = {'time': torch.tensor([[1., 0.], [1., 0.]]),
                'reference': torch.tensor([[0., 1.], [1., 0.]])}
    norms = {name: torch.ones(2) for name in features}
    result, returned_norms = route_stable_physical_directions(
        model, torch.randn(2, 2, 256), 4, lambda model, inputs, seed: (features, norms))
    assert torch.equal(result['time'], features['time'])
    assert torch.equal(result['reference'], torch.tensor([[0., 1.], [0., 0.]]))
    assert returned_norms is norms
    assert (model._reference_routing_calls, model._reference_routing_seen,
            model._reference_routing_changed) == (1, 2, 1)


def test_all_stable_route_has_original_loss_and_gradients(monkeypatch):
    from contextlib import nullcontext
    from torch.nn import functional as F
    from test_pamf_physics_branch_objective import TinyStudent
    from experiments.pamf_physics_branch import objective as module
    from experiments.pamf_physics_branch.route_training_scope import RouteStablePL
    torch.manual_seed(11)
    initial = TinyStudent().state_dict()
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
         'source_rxs': [1, 2], 'source_days': [1],
         'physical_relation': {'beta': .5, 'l_weight': .05, 'u_weight': .05}}

    def batch(n, hidden):
        i = torch.arange(n)
        y = i.remainder(4).div(2, rounding_mode='floor')
        x = torch.randn(n, 2, 256) * .05
        x[:, 0, 0] = torch.where(y == 0, 1., -1.)
        meta = {'rx_i': i.remainder(2) + 1, 'day_i': torch.ones(n, dtype=torch.long),
                'branch_record_id': i, 'tx_label_visible': torch.full((n,), not hidden, dtype=torch.bool)}
        return x, torch.full_like(y, -1) if hidden else y, torch.zeros(n, dtype=torch.long), meta

    labeled, unlabeled = batch(64, False), batch(8, True)
    original_measure = module.physical_directions

    def run(routed):
        model = TinyStudent().train()
        model.load_state_dict(initial)
        model.id_backbone.encoder.response = SimpleNamespace(
            components=lambda inputs: {'selected_delay_samples': torch.zeros(len(inputs), dtype=torch.long)})
        criterion = RouteStablePL(c) if routed else module.PhysicalPL(c)
        model._cd_U_iter = iter([unlabeled])
        model._phase1_labeled_feature_context = lambda: nullcontext()

        def original_hook(**kwargs):
            value = criterion.loss(model, 'cpu')
            model._cd_pending = None
            return value, {}

        model._source_labeled_feature_hook = original_hook
        module.install_labeled(model, c)
        x, y, _, meta = labeled
        model._cd_pending = {'x': x}
        with model._phase1_labeled_feature_context():
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        extra, _ = model._source_labeled_feature_hook(labels=y, extra=(None, meta))
        loss = loss + extra
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters()], criterion.state_dict()

    baseline, baseline_grad, _ = run(False)
    monkeypatch.setattr(module, 'physical_directions',
                        lambda model, x, seed: route_stable_physical_directions(model, x, seed, original_measure))
    actual, actual_grad, state = run(True)
    assert torch.allclose(actual, baseline, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(actual_grad, baseline_grad))
    assert state['reference_routing'] == {'calls': 2, 'selected_samples': 72, 'changed_samples': 0,
                                          'target_reads': 0, 'U_truth_reads': 0}

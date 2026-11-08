import torch

from experiments.pamf_phase3_restoration import feature_response_reference_cache_probe as probe
from experiments.pamf_phase3_restoration.response_joint_prior import selected_perturbation


def test_cache_only_keeps_both_restored_forwards(monkeypatch):
    torch.manual_seed(23)
    clean = torch.randn(2, 2, 256)
    received = torch.randn(2, 2, 256)
    indices = torch.tensor([0, 6301])
    cache = torch.arange(2*4*160, dtype=torch.float32).reshape(2,4,160)
    targets = torch.randn(2,160)
    scale = torch.ones(160)
    calls = []
    events = []
    seen = {}

    class Capture:
        def features(self, value):
            calls.append(value)
            events.append('feature')
            return value.flatten(1)[:, :160]

    class Model:
        def source_readout(self, changed, prior):
            events.append('altered')
            assert prior is None
            return changed, {'relative_solve_residual':torch.zeros(2)}

    def original_joint(model, capture, received, clean, targets, scale, objective, **kwargs):
        base = received * 1.01
        capture.features(base)
        return base, {'relative_solve_residual':torch.zeros(2)}, torch.ones(2), {}

    monkeypatch.setattr(probe.drift.feature.original.control, 'joint_readout', original_joint)
    monkeypatch.setattr(probe.drift.feature.original, 'known_parameters', lambda trace: (None,None,None,None))
    monkeypatch.setattr(probe.drift.feature.original, 'dense_action', lambda *args: None)
    monkeypatch.setattr(probe.drift.feature.original, 'inject', lambda received,delta,action: received+delta)
    monkeypatch.setattr(probe, 'response_loss', lambda *args: torch.zeros(2))
    def feature_loss(base, changed, source, changed_source, scale):
        seen['target'] = changed_source
        return torch.zeros(2)
    monkeypatch.setattr(probe, 'feature_response_loss', feature_loss)
    probe.cached_readout(Model(), Capture(), received, clean, targets, scale, None,
                         indices, 1, {'received':received}, cache)
    _, choice = selected_perturbation(clean, indices, 1)
    assert events == ['feature', 'altered', 'feature', 'feature']
    assert len(calls) == 3
    assert calls[0] is calls[1]
    assert calls[2] is not calls[0]
    assert torch.equal(seen['target'], cache[torch.arange(2), choice])

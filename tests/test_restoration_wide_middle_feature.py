import pytest
import torch
from torch import nn


class TinyReader(nn.Module):
    def __init__(self, repeat=False):
        super().__init__()
        self.core = nn.Module()
        self.core.id_backbone = nn.Module()
        self.core.id_backbone.t2 = nn.Module()
        self.core.id_backbone.t2.pool = nn.AvgPool1d(2)
        self.repeat = repeat

    def features(self, x):
        y = self.core.id_backbone.t2.pool(x)
        if self.repeat:
            self.core.id_backbone.t2.pool(x)
        return y.mean(-1).repeat(1, 80)


def test_midlayer_capture_removes_hook_and_preserves_gradient():
    from experiments.pamf_phase3_restoration.wide_middle_feature_trial import middle_features
    reader = TinyReader()
    x = torch.randn(2, 2, 256, requires_grad=True)
    final, middle = middle_features(reader, x)
    assert final.shape == (2, 160) and middle.shape == (2, 2, 128)
    assert torch.autograd.grad(middle.square().mean(), x)[0].abs().sum() > 0
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_midlayer_capture_rejects_repeated_calls_without_hook_leaks():
    from experiments.pamf_phase3_restoration.wide_middle_feature_trial import middle_features
    reader = TinyReader(repeat=True)
    with pytest.raises(ValueError, match='exactly once'):
        middle_features(reader, torch.ones(1, 2, 256))
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_midlayer_gradient_is_capped_per_record_against_iq_loss():
    from experiments.pamf_phase3_restoration.wide_middle_feature_trial import capped_middle_term
    clean = torch.ones(2, 2, 256)
    restored = (clean + torch.stack([torch.ones(2, 256), .1 * torch.ones(2, 256)])).requires_grad_()
    middle = restored * .001
    target = torch.full_like(middle, .000001)
    term, metrics = capped_middle_term(restored, clean, middle, target)
    assert term.shape == (2,) and (metrics['source_middle_cap_weight'] < 1).all()
    grad = torch.autograd.grad(term.sum(), restored)[0].flatten(1).norm(dim=1)
    iq_norm = 2 * (restored.detach()-clean).flatten(1).norm(dim=1) / 512
    assert torch.all(grad <= .5 * iq_norm * (1 + 1e-5))
    assert not metrics['source_middle_cap_weight'].requires_grad


def test_readout_adds_only_original_restored_midlayer_loss(monkeypatch):
    from experiments.pamf_phase3_restoration import wide_middle_feature_trial as trial
    reader = TinyReader()
    clean = torch.ones(2, 2, 256)
    received = clean + .3
    restored_holder = {}

    def fake_response(_model, wrapped, values, *_args, **_kwargs):
        restored = values.clone().requires_grad_()
        restored_holder['value'] = restored
        wrapped.features(restored)
        with torch.no_grad():
            wrapped.features(clean + .02)
        wrapped.features(restored)
        wrapped.features(restored * .9)
        return restored, {'relative_solve_residual': torch.zeros(len(values))}, restored.square().mean((1, 2)), {}

    monkeypatch.setattr(trial.response, 'response_readout', fake_response)
    restored, aux, loss, metrics = trial.middle_readout(None, reader, received, clean,
        torch.zeros(2, 160), torch.ones(160), None, source_indices=None, epoch=1,
        priors=None, assignment=None, source_trace=None)
    assert restored is restored_holder['value']
    assert metrics['source_middle_weighted_loss'].shape == (2,)
    assert (metrics['source_middle_weighted_loss'] > 0).all()
    assert torch.autograd.grad(loss.sum(), restored)[0].abs().sum() > 0
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_new_arm_preserves_fixed_source_and_budget():
    from experiments.pamf_phase3_restoration import wide_middle_feature_trial as trial
    with pytest.raises(ValueError):
        trial.configuration('other')
    cfg = trial.configuration(trial.ARM)
    assert cfg['epochs'] == 30 and cfg['updates'] == 7410 and cfg['training_stages'] == 1
    assert cfg['checkpoint_selection'] == 'final30_only'
    assert cfg['TX_labels_read'] == cfg['V_reads'] == cfg['OE_reads'] == cfg['target_training_reads'] == 0
    assert cfg['training_F0_forward_rows'] == 5 * 1890000
    assert cfg['default_DURA_changed'] is False


def test_original_scorer_is_loaded_only_in_score_stage(monkeypatch, tmp_path):
    from experiments.pamf_phase3_restoration import wide_middle_feature_trial as trial
    seen = []
    monkeypatch.setattr(trial, 'original_scorer', lambda root: seen.append(root))
    for module, names in ((trial.trial, ('RUN', 'MODULE', 'ARMS', 'configuration', 'factory', 'train', 'controller')),
                          (trial.resource, ('RUN', 'ARM', 'admitted_gpus'))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(trial.sys, 'argv', ['module', 'train', '--root', str(tmp_path)])
    trial.install()
    assert not seen
    monkeypatch.setattr(trial.sys, 'argv', ['module', 'score', '--root', str(tmp_path)])
    trial.install()
    assert seen == [tmp_path]

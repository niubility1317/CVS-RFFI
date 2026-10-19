"""Additional self-supervised boundary, endpoint and label-free selection audit."""
from pathlib import Path
import sys
import copy
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'code')]
from experiments.pamf_restoration.models import ConditionalDiffusion, ConditionalFlow, build
from experiments.pamf_restoration.views import PAMFViews, objective
from experiments.pamf_restoration.unlabeled_selection import CANDIDATES, select

torch.set_num_threads(2)


def test_reconstruction_training_uses_no_checkpoint_classification_CE_or_PL(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Supervised/teacher classifier is forbidden in restoration training')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'cross_entropy', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'kl_div', forbidden)
    extractor = PAMFViews()
    monkeypatch.setattr(extractor.probe, 'classify_features', forbidden)
    monkeypatch.setattr(extractor.probe.core.id_backbone.cls_head, 'classify', forbidden)
    model = build('unet')
    clean = torch.randn(2, 2, 256)*.5
    loss, _ = objective(model(clean+.1*torch.randn_like(clean)), clean, extractor)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in extractor.parameters())


def test_diffusion_v_training_and_complete_sampler_recover_analytic_endpoint(monkeypatch):
    model = ConditionalDiffusion()
    clean, received = torch.randn(2, 2, 256), torch.randn(2, 2, 256)
    def exact_velocity(noisy, condition, time):
        assert torch.equal(condition, received)
        alpha = model.alpha_bar[time.long(), None, None]
        return (alpha.sqrt()*noisy-clean)/(1-alpha).sqrt()
    monkeypatch.setattr(model, 'velocity', exact_velocity)
    reconstructed, v_loss = model.training_prediction(clean, received, torch.Generator().manual_seed(7))
    assert torch.allclose(reconstructed, clean, atol=2e-5, rtol=2e-5)
    assert float(v_loss) < 1e-8
    sampled = model(received, torch.randn_like(received))
    assert torch.allclose(sampled, clean, atol=2e-5, rtol=2e-5)


def test_flow_full_integrator_reaches_analytic_clean_endpoint(monkeypatch):
    model = ConditionalFlow()
    clean, received, noise = (torch.randn(2, 2, 256) for _ in range(3))
    start = received+model.sigma*noise
    def exact_velocity(point, condition, time):
        assert torch.equal(condition, received)
        return clean-start
    monkeypatch.setattr(model.network, 'velocity', exact_velocity)
    sampled = model(received, noise)
    assert torch.allclose(sampled, clean, atol=3e-6, rtol=3e-6)


def test_actual_predictor_never_calls_clean_dependent_training_path(monkeypatch):
    from experiments.pamf_restoration.runner import predict_batch
    model = ConditionalDiffusion().eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('Training noisy-clean construction is forbidden at inference')
    monkeypatch.setattr(model, 'training_prediction', forbidden)
    output = predict_batch(model, torch.randn(2, 2, 256), ['a', 'b'])
    assert output.shape == (2, 2, 256) and torch.isfinite(output).all()


def test_label_free_selection_rejects_TX_scores_and_is_fixed_on_waveforms():
    packet = dict(source_run='fixed', code_commit='fixed', rows=[dict(candidate=name,
                  clean_nmse=.01, mean_practical_nmse=.1+index*.01, identity_mean_nmse=1.)
                  for index, name in enumerate(CANDIDATES)])
    assert select(packet)['candidate'] == 'rescnn'
    bad = copy.deepcopy(packet)
    bad['rows'][0]['TX_accuracy'] = 100.
    with pytest.raises(ValueError, match='TX/F/recognition'):
        select(bad)
    bad = copy.deepcopy(packet)
    bad['F_predictions'] = 'forbidden'
    with pytest.raises(ValueError, match='Only waveform'):
        select(bad)


def test_truth_path_layout_resolution_is_metadata_only_and_unambiguous(tmp_path, monkeypatch):
    from experiments.pamf_restoration.confirmation import target_truth_path
    capsule = tmp_path/'dataset'/'data'/'capsule'
    capsule.mkdir(parents=True)
    remote = capsule.parent/'truth.json'
    local = capsule.parent.parent/'scoring'/'truth.json'
    with pytest.raises(ValueError, match='Exactly one'):
        target_truth_path(capsule)
    remote.write_text('must not open', encoding='utf-8')
    def forbidden(*args, **kwargs):
        raise AssertionError('Path lookup must never open truth')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    assert target_truth_path(capsule) == remote
    local.parent.mkdir()
    local.touch()
    with pytest.raises(ValueError, match='Exactly one'):
        target_truth_path(capsule)
    remote.unlink()
    assert target_truth_path(capsule) == local

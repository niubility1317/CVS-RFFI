import pytest
import torch

from experiments.pamf_phase3_restoration import wide_timefreq_response_trial as subject
from experiments.pamf_phase3_restoration.response_joint_prior_dense_trial import DenseResponseJointPrior


def test_local_complex_spectrogram_has_finite_input_and_parameter_gradients():
    torch.set_num_threads(2)
    network = subject.SpectralContext()
    received = torch.randn(2, 2, 256, requires_grad=True)
    posterior = torch.randn(2, 2, 256, requires_grad=True)
    output = network(received, posterior)
    assert output.shape == received.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for value in (received.grad, posterior.grad, network.net[0].weight.grad):
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0


def test_detail_starts_as_identity_and_spectral_branch_receives_gradients(monkeypatch):
    torch.set_num_threads(2)
    def posterior_stub(self, received, prior):
        posterior = received * 0.7
        return posterior, {'variance': received.new_ones(len(received)), 'base_iq': posterior}
    monkeypatch.setattr(DenseResponseJointPrior, '_readout', posterior_stub)
    model = subject.factory(subject.ARM)
    received = torch.randn(1, 2, 256)
    restored, aux = model._readout(received, None)
    torch.testing.assert_close(restored, aux['base_iq'], rtol=0, atol=0)
    with torch.no_grad():
        model.detail.refinement.output.weight.normal_(std=0.001)
    restored, _ = model._readout(received, None)
    restored.square().mean().backward()
    for value in (model.detail.refinement.output.weight.grad,
                  model.detail.spectral_context.net[0].weight.grad):
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0


def test_protocol_rejects_another_arm_and_keeps_single_stage():
    with pytest.raises(ValueError):
        subject.configuration('other')
    cfg = subject.configuration(subject.ARM)
    assert cfg['epochs'] == 30 and cfg['updates'] == 7410 and cfg['training_stages'] == 1
    assert cfg['TX_labels_read'] == cfg['V_reads'] == cfg['OE_reads'] == cfg['target_training_reads'] == 0
    assert cfg['checkpoint_selection'] == 'final30_only'


def test_original_scorer_is_loaded_only_for_score_process(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(subject, 'original_scorer', lambda root: seen.append(root))
    for module, names in ((subject.trial, ('RUN', 'MODULE', 'ARMS', 'configuration', 'factory', 'train', 'controller')),
                          (subject.resource, ('RUN', 'ARM', 'admitted_gpus'))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(subject.sys, 'argv', ['module', 'train', '--root', str(tmp_path)])
    subject.install()
    assert not seen
    monkeypatch.setattr(subject.sys, 'argv', ['module', 'score', '--root', str(tmp_path)])
    subject.install()
    assert seen == [tmp_path]

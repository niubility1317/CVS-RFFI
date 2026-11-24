import pytest
import torch

from experiments.shen_wisig.model import ReceiverAdversarialCNN, spectrogram
from experiments.shen_wisig.runner import source_contract


def test_channel_independent_spectrogram_is_packet_local_and_gain_invariant():
    iq = torch.randn((2, 2, 256), generator=torch.Generator().manual_seed(4))
    images = spectrogram(iq)
    assert images.shape == (2, 1, 26, 14)
    torch.testing.assert_close(images[:1], spectrogram(iq[:1] * 2.5), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(spectrogram(torch.zeros_like(iq))).all()


def test_receiver_loss_reverses_only_the_gradient_to_shared_features():
    model = ReceiverAdversarialCNN(6, 5)
    features = torch.randn(3, 512, requires_grad=True)
    parameters = (features, model.rx[0].weight)
    reference = torch.autograd.grad(model.rx(features).sum(), parameters)
    reversed_grads = torch.autograd.grad(model.receiver_logits(features).sum(), parameters)
    torch.testing.assert_close(-reference[0], reversed_grads[0])
    torch.testing.assert_close(reference[1], reversed_grads[1])
    logits = model.classify_features(model.features(torch.randn(3, 2, 256)))
    assert logits.shape == (3, 6)


def test_training_rejects_target_config_before_opening_the_source_contract():
    with pytest.raises(ValueError, match='registered source paths'):
        source_contract(dict(run_id='test', row_id='test', code_commit='code', model_seed=1,
                             source_cache='source', source_contract='missing', output_root='out',
                             tests={'cross_rx': {'views_root': 'target'}}))

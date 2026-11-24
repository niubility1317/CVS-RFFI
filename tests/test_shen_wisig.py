import pytest
import torch

from experiments.shen_wisig.model import ReceiverAdversarialCNN, spectrogram
from experiments.shen_wisig.runner import make_optimizers, source_contract, train_loss
from experiments.shen_wisig.channel import exponential_tdl, linear_fir


@pytest.mark.parametrize(('front_end', 'shape'), [
    ('original', (26, 14)), ('stable', (26, 14)), ('wifi', (32, 21)),
])
def test_channel_independent_spectrogram_is_packet_local_and_gain_invariant(front_end, shape):
    iq = torch.randn((2, 2, 256), generator=torch.Generator().manual_seed(4))
    images = spectrogram(iq, front_end)
    assert images.shape == (2, 1, *shape)
    torch.testing.assert_close(images[:1], spectrogram(iq[:1] * 2.5, front_end), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(spectrogram(torch.zeros_like(iq), front_end)).all()
    model = ReceiverAdversarialCNN(6, 5, front_end)
    assert model.classify_features(model.features(iq)).shape == (2, 6)


def test_receiver_loss_reverses_only_the_gradient_to_shared_features():
    model = ReceiverAdversarialCNN(6, 5)
    features = torch.randn(3, 512, requires_grad=True)
    parameters = (features, model.rx[0].weight)
    reference = torch.autograd.grad(model.rx(features).sum(), parameters)
    reversed_grads = torch.autograd.grad(model.receiver_logits(features).sum(), parameters)
    torch.testing.assert_close(-reference[0], reversed_grads[0])
    torch.testing.assert_close(reference[1], reversed_grads[1])
    scaled = torch.autograd.grad(model.receiver_logits(features, .2).sum(), parameters)
    torch.testing.assert_close(-.2 * reference[0], scaled[0])
    torch.testing.assert_close(reference[1], scaled[1])
    logits = model.classify_features(model.features(torch.randn(3, 2, 256)))
    assert logits.shape == (3, 6)


def test_receiver_best_response_updates_head_before_adversarial_backward():
    model = ReceiverAdversarialCNN(6, 5, 'wifi')
    optimizer, adversary = make_optimizers(model, 'joint')
    images = spectrogram(torch.randn(2, 2, 256), 'wifi')
    before = model.rx[0].weight.detach().clone()
    loss, _ = train_loss(model, images, images, torch.tensor([0, 1]), torch.tensor([0, 1]), adversary)
    after = model.rx[0].weight.detach().clone()
    assert not torch.equal(before, after)
    assert all(p.grad is None for p in model.backbone.parameters())
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(model.rx[0].weight, after)
    assert all(torch.isfinite(p.grad).all() for p in model.backbone.parameters())


def test_training_rejects_target_config_before_opening_the_source_contract():
    with pytest.raises(ValueError, match='registered source paths'):
        source_contract(dict(run_id='test', row_id='test', code_commit='code', model_seed=1,
                             source_cache='source', source_contract='missing', output_root='out',
                             tests={'cross_rx': {'views_root': 'target'}}))


def test_source_multipath_is_causal_packet_local_and_reproducible():
    signal = torch.tensor([[0., 0., 0., 1.]], dtype=torch.complex64)
    taps = torch.tensor([[0., 1.]], dtype=torch.complex64)
    torch.testing.assert_close(linear_fir(signal, taps), torch.zeros_like(signal))
    iq = torch.ones(3, 2, 256)
    generator = torch.Generator().manual_seed(17)
    torch.testing.assert_close(exponential_tdl(iq, generator, 0.), iq)
    result = exponential_tdl(iq, generator.manual_seed(17), 1.)
    torch.testing.assert_close(result, exponential_tdl(iq, generator.manual_seed(17), 1.))
    assert result.shape == iq.shape and torch.isfinite(result).all()
    assert not torch.equal(result[0], result[1])

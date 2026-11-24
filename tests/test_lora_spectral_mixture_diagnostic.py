import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.spectral_mixture import SpectralMixture
from experiments.pamf_lora_main.spectral_mixture_diagnostic import CONTEXT, FIELDS, MODES, oracle_readouts
from experiments.pamf_lora_main.spectral_mixture_source import MODES as PARENT_MODES, recover


def test_only_isolated_directions_change_when_reference_is_replaced():
    rng = torch.Generator().manual_seed(71)
    tracker = PacketRestorer('consensus').eval()
    source = torch.randn(46, 6, 2, 256, generator=rng)
    model = SpectralMixture().eval()
    model.fit(lambda: (coordinates(tracker, batch)[0] for batch in source.split(16)))
    truth = source[:5]
    noise = .4 * torch.randn(truth.shape, generator=rng)
    iq = truth + noise
    references = torch.stack((noise, truth), dim=1)
    actual, count, metadata = oracle_readouts(iq, references, tracker, model, 0.)
    assert (len(actual), count, metadata.shape) == (len(MODES)-2, len(iq), (len(iq), len(FIELDS)))
    assert torch.isfinite(actual).all() and metadata[:, FIELDS.index('rms_lock_error')].max() < 1e-6
    original, _, _ = recover(iq, tracker, model, 0.)
    inherited = [PARENT_MODES.index(mode)-2 for mode in CONTEXT[2:]]
    torch.testing.assert_close(actual[:3], original[inherited])

    altered, _, _ = oracle_readouts(iq, -references, tracker, model, 0.)
    assert torch.equal(actual[:3], altered[:3]) and torch.equal(actual[3], altered[3])
    assert not torch.equal(actual[-2], altered[-2])

    zeros = torch.zeros_like(truth)
    clean, active, _ = oracle_readouts(truth, torch.stack((zeros, truth), dim=1),
                                       tracker, model, float('inf'))
    assert active == 0 and torch.equal(clean, truth.expand_as(clean))

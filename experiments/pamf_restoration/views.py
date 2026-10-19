"""Use PAMF's existing analytic branch inputs; never train an identity head."""
import torch
from torch import nn
from .models import complex_iq, pair


class PAMFViews(nn.Module):
    def __init__(self):
        super().__init__()
        from experiments.standard_ce_baselines.sigtime import SIGTimeCVS
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026092701)
            self.probe = SIGTimeCVS(2026092701).eval().requires_grad_(False)

    def forward(self, x):
        # These transforms have no learned identity parameters. The Sinc bank
        # is the declared default initialization, not a trained classifier's
        # filters. Learned embeddings/accuracy are not scored by this probe.
        backbone = self.probe.core.id_backbone
        freq, rho, dac_stats, pa_stats = backbone._mirror_compressed_features(x)
        return dict(iq=x, fft=pair(torch.fft.fft(complex_iq(x), norm='ortho')),
                    frequency=freq, circularity=rho, pa_stats=pa_stats,
                    pa=backbone.pa_lift(x), sinc=backbone._sinc_on_iq(x),
                    highfreq=backbone.hf(x), reference=self.probe.response(x))


def sample_nmse(estimate, target):
    dimensions = tuple(range(1, target.ndim))
    return (estimate-target).square().mean(dimensions) / target.square().mean(dimensions).clamp_min(1e-4)


def scalar_aligned(estimate, target):
    """Scoring/training only. Never used to form inference outputs.

    Reports shape recovery modulo one complex gain/phase, separately from
    absolute waveform recovery. It does not prove recovery of lost samples.
    """
    z, ref = complex_iq(estimate), complex_iq(target)
    gain = (z.conj()*ref).sum(-1)/(z.abs().square().sum(-1)+1e-6)
    return pair(z*gain[:, None])


def objective(estimate, clean, extractor, target_views=None):
    aligned = scalar_aligned(estimate, clean)
    reference = extractor(clean) if target_views is None else target_views
    predicted = extractor(estimate)
    raw = sample_nmse(estimate, clean).mean()
    shape = sample_nmse(aligned, clean).mean()
    # Complex FFT L2 duplicates IQ L2. Compare log amplitude instead.
    log_spectrum = (torch.log1p(complex_iq(predicted['fft']).abs()) -
                    torch.log1p(complex_iq(reference['fft']).abs())).square().mean()
    losses = dict(raw_iq=raw, aligned_iq=shape, log_fft=log_spectrum)
    for name in ('frequency', 'pa', 'sinc', 'highfreq', 'reference'):
        losses[name] = sample_nmse(predicted[name], reference[name]).mean()
    # Oracle scalar alignment is diagnostic only, not the training target or
    # the scientific selection rule. Actual inference cannot read clean IQ.
    total = raw + .2*log_spectrum
    total = total + .05*sum(losses[k] for k in ('frequency', 'pa', 'sinc', 'highfreq', 'reference'))
    return total, losses


@torch.no_grad()
def metrics(estimate, clean, extractor, target_views=None):
    reference = extractor(clean) if target_views is None else target_views
    direct = extractor(estimate)
    aligned = scalar_aligned(estimate, clean)
    values = {name+'_nmse': sample_nmse(direct[name], reference[name]) for name in reference}
    values['aligned_iq_nmse'] = sample_nmse(aligned, clean)
    values['prefix232_iq_nmse'] = sample_nmse(estimate[..., :232], clean[..., :232])
    values['fft_log_mse'] = (torch.log1p(complex_iq(direct['fft']).abs())-
                             torch.log1p(complex_iq(reference['fft']).abs())).square().mean(1)
    return values


class RestoredPAMF(nn.Module):
    """Inference adapter: all original PAMF branches consume the same IQ.

    A caller may supply its own provenance-verified, already trained PAMF.
    This adapter never fits/replaces its default classifier.
    """
    def __init__(self, restorer, pamf):
        super().__init__()
        self.restorer, self.pamf = restorer, pamf

    def forward(self, received, noise=None):
        from .models import ConditionalDiffusion, ConditionalFlow
        restored = self.restorer(received, noise) if isinstance(self.restorer, (ConditionalDiffusion, ConditionalFlow)) else self.restorer(received)
        return self.pamf(restored)

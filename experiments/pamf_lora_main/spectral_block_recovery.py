"""LoRa coarse spectral prediction and observed-complex-residual refinement."""

import torch
from torch import nn

from .weak_signal_probe import observed_noise_rms


def head(inputs, outputs):
    return nn.Sequential(nn.Conv1d(inputs, 32, 7, padding=3), nn.SiLU(),
                         nn.Conv1d(32, 48, 5, padding=4, dilation=2), nn.SiLU(),
                         nn.Conv1d(48, 32, 5, padding=2), nn.SiLU(), nn.Conv1d(32, outputs, 1))


def repeat_confidence(spectrum):
    """Estimate repeatable complex spectrum power from same-packet cross products."""
    common = spectrum.mean(dim=1, keepdim=True)
    residual_power = (spectrum-common).abs().square().mean(dim=1, keepdim=True)
    variance = residual_power/(spectrum.shape[1]-1)
    shared_power = (common.abs().square()-variance).clamp_min(0.)
    return shared_power/(shared_power+variance+1e-6)


def constrain_update(change, confidence, *, protect):
    protected = change-confidence*change.mean(dim=1, keepdim=True)
    if protect:
        return protected
    # The control uses the same correction norm at each packet frequency.
    # Keep its original complex direction across the chirps.
    norm = torch.linalg.vector_norm(change, dim=1, keepdim=True)
    amount = torch.linalg.vector_norm(protected, dim=1, keepdim=True)
    scale = (amount/torch.where(norm>0,norm,1.)).clamp_max(1.)
    return change*scale


class SpectralBlockRecovery(nn.Module):
    """Predict a complex prior in interleaved blocks, then constrain it with IQ."""

    def __init__(self, *, masked, protection=None):
        super().__init__()
        self.masked = masked
        self.protection = protection
        n = 256
        groups = torch.arange(n).div(2, rounding_mode='floor') % 4
        self.register_buffer('hidden', groups == torch.arange(4)[:, None], persistent=False)
        frequency = torch.fft.fftshift(torch.fft.fftfreq(n))
        self.register_buffer('frequency', frequency, persistent=False)
        self.prior = head(7, 2)
        self.refine = head(9, 3)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def predict_prior(self, spectrum):
        mask = self.hidden[:, None, None]
        copy = spectrum.unsqueeze(0).expand(len(mask), -1, -1, -1)
        visible = copy.masked_fill(mask, 0.) if self.masked else copy
        common = visible.mean(2, keepdim=True).expand_as(visible)
        spread = (visible-common).abs().square().mean(2, keepdim=True).expand_as(visible).log1p()
        fields = torch.stack((visible.real, visible.imag, common.real, common.imag, spread,
                              self.frequency.expand_as(spread), mask.expand_as(spread).to(spread.dtype)), dim=-2)
        out = self.prior(fields.flatten(0, 2)).float().reshape(*visible.shape[:-1], 2, visible.shape[-1])
        return (torch.complex(out[..., 0, :], out[..., 1, :])*mask).sum(0)

    def forward(self, observed, quality):
        spec = torch.fft.fftshift(torch.fft.fft(observed, norm='ortho'), dim=-1)
        level = observed_noise_rms(observed).square()
        confidence = level/(level+.02)
        repeat = spec.mean(dim=1, keepdim=True)
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            prior = self.predict_prior(spec)
        innovation = prior-spec
        noncarrier = self.frequency != 0
        coarse_change = .5*confidence*innovation*noncarrier
        repeat_evidence = repeat_confidence(spec) if self.protection is not None else None
        if repeat_evidence is not None:
            coarse_change = constrain_update(coarse_change, repeat_evidence, protect=self.protection)
        coarse = spec + coarse_change
        context = coarse if repeat_evidence is not None else prior
        fields = torch.stack((spec.real, spec.imag, context.real, context.imag,
                              (spec-repeat).abs().log1p(), innovation.abs().log1p(),
                              quality.expand_as(spec.real), level.log1p().expand_as(spec.real),
                              self.frequency.expand_as(spec.real)), dim=2)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            prediction = self.refine(fields.flatten(0,1))
        logit, a, b = prediction.float().unflatten(0, spec.shape[:2]).unbind(2)
        extra = .15*torch.complex(a.tanh(), b.tanh())
        fine_change = confidence*(logit.sigmoid()*innovation + extra)*noncarrier
        if repeat_evidence is not None:
            fine_change = constrain_update(fine_change, repeat_evidence, protect=self.protection)
        final = spec+fine_change
        return tuple(torch.fft.ifft(torch.fft.ifftshift(s, dim=-1), norm='ortho') for s in (coarse, final))


def varied_observation(signal, rng):
    """Independently generated generic distortions, exposing only the output IQ."""
    device, dtype = signal.device, signal.real.dtype
    batch, chirps, n = signal.shape

    def uniform(shape):
        return torch.rand(shape, device=device, dtype=dtype, generator=rng)

    def random_complex(shape):
        return torch.complex(torch.randn(shape, device=device, dtype=dtype, generator=rng),
                             torch.randn(shape, device=device, dtype=dtype, generator=rng)) * (2**-.5)

    shared_size = (batch, 1, 1)
    donor = random_complex((batch, 1, n))
    independent = random_complex(signal.shape)
    mix = .8*uniform(shared_size)
    noise = (1-mix).sqrt()*independent + mix.sqrt()*donor
    tilt = (2*uniform(shared_size)-1)*.85
    noise = (noise+tilt*noise.roll(1, dims=-1))/(1+tilt.square()).sqrt()
    time = torch.arange(n, device=device, dtype=dtype)/n
    envelope = 1 + .6*torch.cos(2*torch.pi*(time+uniform((batch,chirps,1))))
    scale = uniform((batch, chirps, 1)).square()
    amplitude = signal.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    echo = .14*random_complex(shared_size)*signal.roll(2, dims=-1)
    return signal + echo + amplitude*scale*envelope*noise

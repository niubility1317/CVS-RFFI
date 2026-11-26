"""Combine two fixed source restorers in packet-local complex spectral coordinates."""

import torch
from torch import nn

from experiments.pamf_lora_main.w24_packet_unet import PacketUNet


BANDS = ((1, 9), (9, 33), (33, 81), (81, 193))


class SpectralExperts(nn.Module):
    def __init__(self, dura, *, local):
        super().__init__()
        self.dura = dura.requires_grad_(False)
        self.unet = PacketUNet(across_chirps=True).requires_grad_(False)
        self.local = local
        self.router = nn.Sequential(nn.Linear(12, 48), nn.LayerNorm(48), nn.SiLU(),
                                    nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 5))
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([0., 1., -1., 1., -1.]))
        bins = (384 * torch.fft.fftfreq(384)).round().abs()
        self.register_buffer('mask', torch.stack([(bins >= lo) & (bins < hi) for lo, hi in BANDS]),
                             persistent=False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.dura.eval()
        self.unet.eval()
        return self

    def band_mean(self, values):
        return torch.stack([values[..., band].mean(-1) for band in self.mask], dim=-1)

    @torch.no_grad()
    def sources(self, y, quality, rms, edges):
        dura = self.dura.propose(y, quality, rms, edges)[0][-1]
        unet = self.unet.propose(y, quality, rms, edges)[0][-1]
        observation, a, b = (torch.fft.fft(x, norm='ortho') for x in (y, dura - y, unet - y))
        noise = (observation[..., self.mask[-1]].abs().square().median(-1, keepdim=True).values
                 / .69314718056).clamp(1e-5, 4.)
        pa, pb, py = (self.band_mean(x.abs().square()) for x in (a, b, observation))

        def coherence(x, z, px, pz):
            corr = self.band_mean(x * z.conj()) / ((px + .001) * (pz + .001)).sqrt()
            return corr.real, corr.imag

        fields = ((pa / noise).log1p(), (pb / noise).log1p(), (py / noise).log1p(),
                  *coherence(a, b, pa, pb), *coherence(a, observation, pa, py),
                  *coherence(b, observation, pb, py), noise.log().expand_as(py) / 5.,
                  quality.expand_as(py), py.new_tensor((-.75, -.25, .25, .75)).expand_as(py))
        proposed = torch.stack((torch.zeros_like(a), a * .5, a, b * .5, b), dim=-2)
        return torch.stack(fields, dim=-1), observation, proposed

    def combine(self, proposed, logits):
        weights = logits.softmax(-1)
        fullband = torch.einsum('bcge,gn->bcen', weights, self.mask.to(weights))
        return (proposed * fullband).sum(-2)

    def estimate(self, y, quality, rms, edges):
        fields, observation, proposed = self.sources(y, quality, rms, edges)
        logits = self.router(fields).float()
        if not self.local:
            logits = logits.mean((1, 2), keepdim=True).expand_as(logits)
        edit = torch.fft.ifft(self.combine(proposed, logits), norm='ortho')
        return y + edit, logits, observation, proposed

    def propose(self, y, quality, rms, edges):
        restored = self.estimate(y, quality, rms, edges)[0]
        # The registered W24 renderer applies half of the returned update.
        return (y + 2. * (restored - y),),


def objective_for(archive):
    def calculate(model, tracker, raw, rng, *, with_identity, recipe, step):
        raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        clean = tracker.aperture(raw)
        a, b = (tracker.aperture(archive.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
                for _ in range(2))
        if with_identity:
            a[-2:] = b[-2:] = clean[-2:]
        observed = torch.stack((a, b), dim=1).flatten(0, 1)
        center, edges = observed[..., :384], observed[..., 384:]
        y, phase, quality, rms = archive.coordinates(tracker, archive.real_iq(center))
        frame = phase * rms
        target = clean[..., :384].repeat_interleave(2, dim=0) / frame
        restored, logits, spectrum, edits = model.estimate(y, quality, rms, edges)

        target_spectrum = torch.fft.fft(target, norm='ortho')
        power = target_spectrum.abs().square() + .001
        error = (edits + (spectrum - target_spectrum).unsqueeze(-2)).abs().square()
        risks = model.band_mean((error / power.unsqueeze(-2)).log1p()).transpose(-1, -2)
        soft_target = (-risks / .20).softmax(dim=-1)
        ordinal = -(soft_target * logits.log_softmax(dim=-1)).sum(-1).mean()
        outer = archive.waveform_risk(restored[:, (0, 7), :], target[:, (0, 7), :])
        inner = archive.waveform_risk(restored[:, 1:-1, :], target[:, 1:-1, :])
        identity = y.real.new_zeros(())
        if with_identity:
            identity = (restored[-4:] - y[-4:]).abs().square().mean()
        reports = torch.stack((outer, inner, ordinal, identity,
                               (restored - target).abs().square().mean()))
        weights = reports.new_tensor((.5, .5, .25, 2., .25))
        return reports @ weights, reports

    return calculate

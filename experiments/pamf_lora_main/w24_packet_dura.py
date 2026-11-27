"""Native LoRa DURA with two observation-consistent updates over the capture."""

import torch
from torch import nn
from torch.nn import functional as F

from .w24_packet_unet import block


CHIRPS = 8
CHIRP = 1024
PACKET = CHIRPS * CHIRP


class PacketPrior(nn.Module):
    def __init__(self, channels, outputs):
        super().__init__()
        self.stem = block(channels, 24, kernel=9)
        self.down = nn.Sequential(block(24, 32, stride=2), block(32, 48, stride=2))
        self.mix = nn.ModuleList(nn.Sequential(
            nn.Conv1d(48, 48, 5, padding=2 * dilation, dilation=dilation, groups=48),
            nn.Conv1d(48, 48, 1), nn.GroupNorm(8, 48), nn.SiLU(),
        ) for dilation in (1, 4, 16, 64))
        self.read = nn.Sequential(block(72, 32), nn.Conv1d(32, outputs, 5, padding=2))
        nn.init.normal_(self.read[-1].weight, std=.002)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self, *fields):
        # This convolutional axis is the full 8,192-sample capture, including seams.
        inputs = torch.stack(torch.broadcast_tensors(*fields), dim=1).flatten(-2)
        mixed = self.training and inputs.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            skip = self.stem(inputs)
            hidden = self.down(skip)
            for context in self.mix:
                hidden = hidden + context(hidden)
            hidden = F.interpolate(hidden, size=PACKET, mode='linear', align_corners=False)
            out = self.read(torch.cat((skip, hidden), dim=1))
        return out.float().unflatten(-1, (CHIRPS, CHIRP)).unbind(1)


def fields(*signals):
    return tuple(x for iq in signals for x in (iq.real, iq.imag))


def zero_dc(iq):
    return iq - iq.mean(-1, keepdim=True)


class NativePacketDura(nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.arange(CHIRP, dtype=torch.float64) / 1e6
        angle = 2 * torch.pi * 125e3 * (t.square() / .002048 - t / 2)
        self.register_buffer('chirp', torch.polar(torch.ones_like(angle), angle).to(torch.complex64), persistent=False)
        f = torch.fft.fftfreq(CHIRP).abs()
        self.register_buffer('noise_band', f >= .35, persistent=False)
        self.register_buffer('detail_band', (f > 0) & (f <= .04), persistent=False)
        self.coarse = PacketPrior(8, 2)
        self.priors = nn.ModuleList(PacketPrior(12, 4) for _ in range(2))

    def coordinates(self, raw):
        native = raw.unflatten(-1, (CHIRPS, CHIRP))
        scale = native.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        wave = native / scale
        dechirped = wave * self.chirp.conj()
        bins = torch.fft.fft(dechirped, n=4 * CHIRP).abs().argmax(-1, keepdim=True)
        t = torch.arange(CHIRP, device=raw.device)
        phase = 2 * torch.pi * bins * t / (4 * CHIRP)
        rotation = self.chirp * torch.polar(torch.ones_like(phase), phase)
        residual = wave * rotation.conj()
        lag = CHIRP // 16
        slope = (residual[..., lag:] * residual[..., :-lag].conj()).mean(-1, keepdim=True).angle() / lag
        phase = slope.clamp(-torch.pi / (4 * CHIRP), torch.pi / (4 * CHIRP)) * t
        rotation = rotation * torch.polar(torch.ones_like(phase), phase)
        aligned = wave * rotation.conj()
        carrier = aligned.mean(-1, keepdim=True)
        rotation = rotation * torch.polar(torch.ones_like(carrier.real), carrier.angle())
        aligned = wave * rotation.conj()
        quality = (carrier.abs().square() / aligned.abs().square().mean(-1, keepdim=True).clamp_min(1e-12)).clamp(0, 1)
        return aligned, wave, rotation * scale, quality

    def forward(self, raw):
        if raw.ndim != 2 or raw.shape[-1] != PACKET or not raw.is_complex():
            raise ValueError('Expected a complete native complex LoRa capture')
        y, wave, frame, quality = self.coordinates(raw)
        spec = torch.fft.fft(y, norm='ortho')
        variance = (spec[..., self.noise_band].abs().square().median(-1, keepdim=True).values / .69314718056
                    ).clamp(1e-6, 4.)
        log_noise = variance.log().clamp(-12, 2) / 6
        shared = y.mean(-2, keepdim=True)

        def bounded(change):
            centered = zero_dc(change)
            rms = centered.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
            return centered * (.85 / rms).clamp(max=1.)

        r, i = self.coarse(*fields(y, wave, y - shared), quality, log_noise)
        coarse = y + bounded(torch.complex(r.tanh(), i.tanh()))
        current = coarse
        stages = [coarse]
        for solver in self.priors:
            r, i, log_observed, log_prior = solver(*fields(y, coarse, current, y-current, y-shared),
                                                   quality, log_noise)
            prior = current + bounded(torch.complex(r.tanh(), i.tanh()))
            observed_noise = variance * (1.25 * log_observed.tanh()).exp()
            prior_variance = .035 * (1.25 * log_prior.tanh()).exp()
            # Identity-operator MAP followed by a carrier-preserving RMS projection.
            # Both updates stay anchored to the original received observation.
            trust = observed_noise / (observed_noise + prior_variance)
            current = y + bounded(trust * (prior - y))
            stages.append(current)
        return tuple((z * frame).flatten(-2) for z in stages), frame

    def restore_native(self, raw):
        return self(raw)[0][-1]


def native_additive_pair(raw, rng, *, with_identity):
    from .w24_additive_reference import action_pair, additive_noise

    if raw.ndim != 2 or raw.shape[-1] != PACKET or not raw.is_complex():
        raise ValueError('Native source pairs require full complex captures')
    original = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    after_action, _ = action_pair(original, original, rng)
    received = after_action + additive_noise(after_action, rng)
    if with_identity:
        after_action[-2:] = received[-2:] = original[-2:]
    return received, (original, after_action)


def source_objective(model, received, reference, *, with_identity, audit_references=()):
    stages, frame = model(received)
    actual = torch.lerp(received, stages[-1], .5)
    error = actual - reference
    denominator = reference.abs().square().mean(-1).clamp_min(1e-9)

    def risk(value):
        return (value.abs().square().mean(-1) / denominator).mean()

    trajectory = tuple(risk(torch.lerp(received, stage, .5) - reference) for stage in stages[:-1])
    detail = error.unflatten(-1, (CHIRPS, CHIRP)) / frame
    detail = torch.fft.fft(detail, norm='ortho')[..., model.detail_band]
    common = detail.mean(-2, keepdim=True)
    private = (detail-common).abs().square().mean()
    boundary = error.unflatten(-1, (CHIRPS, CHIRP))
    boundary_cost = ((boundary[..., :24].abs().square().mean((-2, -1))
                      + boundary[..., -24:].abs().square().mean((-2, -1))) / (2 * denominator)).mean()
    identity = (actual[-2:] - received[-2:]).abs().square().mean() if with_identity else error.real.new_zeros(())
    parts = torch.stack((risk(error), *trajectory, common.abs().square().mean(), private, boundary_cost, identity))
    loss = parts @ parts.new_tensor((1., .18, .18, .30, .65, .20, .80))
    if audit_references:
        with torch.no_grad():
            audit = torch.stack([((actual - wave).abs().square().mean(-1)
                                  / wave.abs().square().mean(-1).clamp_min(1e-9)).mean()
                                 for wave in audit_references])
        parts = torch.cat((parts, audit))
    return loss, parts

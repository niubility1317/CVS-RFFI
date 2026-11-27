"""Packet-wide dual-domain LoRa waveform reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 64
CENTER = 384


def block(cin, cout, kernel=5, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2),
                         nn.GroupNorm(8, cout), nn.SiLU())


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.pre_feed_forward = nn.LayerNorm(WIDTH)
        self.feed_forward = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH))

    def forward(self, tokens):
        q = self.pre_attention(tokens)
        tokens = tokens + self.attention(q, q, q, need_weights=False)[0]
        return tokens + self.feed_forward(self.pre_feed_forward(tokens))


class PacketUNet(nn.Module):
    """Preserve each chirp's skip features while mixing coarse time and packet context."""

    def __init__(self, *, across_chirps, with_state=False, center=CENTER, with_reliability=False):
        super().__init__()
        self.center = center
        self.across_chirps = across_chirps
        self.with_state = with_state
        self.stem = block(14 if with_state else 12, WIDTH, 7)
        self.reliability_stem = None
        if with_reliability:
            self.reliability_stem = nn.Conv1d(1,WIDTH,7,padding=3,bias=False)
            nn.init.zeros_(self.reliability_stem.weight)
        self.down = nn.ModuleList(block(WIDTH, WIDTH, stride=2) for _ in range(3))
        self.up = nn.ModuleList(block(2 * WIDTH, WIDTH) for _ in range(3))
        self.temporal_position = nn.Parameter(torch.empty(center // 8, WIDTH))
        self.chirp_position = nn.Parameter(torch.empty(8, WIDTH))
        nn.init.normal_(self.temporal_position, std=.02)
        nn.init.normal_(self.chirp_position, std=.02)
        self.temporal = AttentionBlock()
        self.axis = AttentionBlock()
        self.time_edit = nn.Conv1d(WIDTH, 2, 5, padding=2)
        self.context = nn.Linear(WIDTH, 32)
        self.spectral = nn.Sequential(block(38, 48), block(48, 48, 9), block(48, 32, 9),
                                      nn.Conv1d(32, 2, 5, padding=2))
        for output in (self.time_edit, self.spectral[-1]):
            nn.init.normal_(output.weight, std=.002)
            nn.init.zeros_(output.bias)
        bins = (torch.fft.fftfreq(center) * center).round()
        self.register_buffer('frequency', bins / (center / 2), persistent=False)
        self.register_buffer('noise_band', bins.abs() >= center / 3, persistent=False)

    def mix(self, feature, batch, chirps):
        tokens = feature.transpose(-1, -2).unflatten(0, (batch, chirps))
        tokens = tokens + self.temporal_position.to(tokens) + self.chirp_position.to(tokens)[None, :, None, :]
        tokens = self.temporal(tokens.flatten(0, 1)).unflatten(0, (batch, chirps))
        if self.across_chirps:
            across = tokens.transpose(1, 2).contiguous()
            tokens = self.axis(across.flatten(0, 1)).unflatten(0, across.shape[:2]).transpose(1, 2)
        else:
            # Same mixer, parameter count and sequence length on each arm.
            tokens = self.axis(tokens.reshape(-1, chirps, WIDTH)).reshape(tokens.shape)
        return tokens.flatten(0, 1).transpose(-1, -2)

    @staticmethod
    def complex_edit(value, shape):
        real, imag = value.float().unflatten(0, shape).unbind(2)
        return torch.complex(real, imag)

    def forward(self, y, quality, rms, edges, *, state=None, reliability=None):
        batch, chirps, _ = y.shape
        prefix = (batch, chirps)
        spectrum = torch.fft.fft(y, norm='ortho')
        noise = (spectrum[..., self.noise_band].abs().square().median(-1, keepdim=True).values
                 / .69314718056).clamp(1e-5, 4.)
        sn = noise.log() / 5.
        sides = edges.unflatten(-1, (2, 24))
        power = sides.abs().square().mean(-1).clamp_min(1e-9)
        step = (sides[..., 1:] * sides[..., :-1].conj()).mean(-1) / power
        step = torch.stack((step.real, step.imag), dim=-1).flatten(-2)
        side_power = (power / rms.square()).log().clamp(-16., 6.) / 5.
        dy = torch.diff(y, dim=-1, prepend=y[..., :1])
        conditions = torch.cat((quality, sn, side_power, step), dim=-1).unsqueeze(-1).expand(-1, -1, -1, self.center)
        inputs = torch.cat((torch.stack((y.real, y.imag, dy.real, dy.imag), dim=-2), conditions), dim=-2)
        if self.with_state:
            inputs = torch.cat((inputs,torch.stack((state.real,state.imag),dim=-2)),dim=-2)
        inputs = inputs.flatten(0,1)
        if self.reliability_stem is None:
            hidden = self.stem(inputs)
        else:
            added = self.reliability_stem(reliability.flatten(0,1))
            hidden = self.stem[2](self.stem[1](self.stem[0](inputs)+added))
        skips = []
        for down in self.down:
            skips.append(hidden)
            hidden = down(hidden)
        hidden = self.mix(hidden, batch, chirps)
        spectral_context = self.context(hidden.mean(-1)).unsqueeze(-1).expand(-1, -1, self.center)
        for up, skip in zip(self.up, reversed(skips), strict=True):
            hidden = F.interpolate(hidden, size=skip.shape[-1], mode='linear', align_corners=False)
            hidden = up(torch.cat((hidden, skip), dim=1))
        time = self.complex_edit(self.time_edit(hidden), prefix)
        spectral_inputs = torch.stack((spectrum.real, spectrum.imag, spectrum.abs().log1p(),
                                       (spectrum.abs().square() / noise).log1p(),
                                       sn.expand_as(y.real), self.frequency.expand_as(y.real)), dim=-2)
        frequency = self.spectral(torch.cat((spectral_inputs.flatten(0, 1), spectral_context), dim=1))
        frequency = torch.fft.ifft(self.complex_edit(frequency, prefix), norm='ortho')
        scale = (noise / (noise + .01)).sqrt()

        def bounded(edit):
            edit = edit - edit.mean(-1, keepdim=True)
            return y + scale * edit / (1. + edit.abs().square().mean(-1, keepdim=True)).sqrt()

        return bounded(time), bounded(time + frequency)

    def propose(self, y, quality, rms, edges):
        # The registered inference renderer reads the final waveform from this tuple.
        return self(y, quality, rms, edges),


def correction_risk(edit, desired):
    """Measure signed complex direction and gain separately across usable detail bands."""
    spectra = [torch.fft.fft(view, norm='ortho') for view in (edit, desired)]
    bins = (torch.fft.fftfreq(CENTER, device=edit.device) * CENTER).round().abs()
    floor = .001
    risks = []
    for low, high in ((1, 9), (9, 33), (33, 81), (81, 193)):
        correction, target = (x[..., (bins >= low) & (bins < high)] for x in spectra)
        actual_power = correction.abs().square().mean(-1)
        needed_power = target.abs().square().mean(-1)
        product = (correction * target.conj()).real.mean(-1)
        cosine = product / ((actual_power + floor) * (needed_power + floor)).sqrt()
        support = needed_power / (needed_power + floor)
        direction = ((1. - cosine) * support).mean()
        gain = (((actual_power + floor).sqrt() / (needed_power + floor).sqrt() - 1.).square()).log1p().mean()
        risks.append(torch.stack((direction, gain)))
    return torch.stack(risks).mean(0).unbind()


def objective_for(archive, *, complex_fidelity=None, coherence_priority=None):
    if coherence_priority is not None:
        from experiments.pamf_lora_main.w24_coherence_constraint import coherence_risks, constrained_loss

    def calculate(model, tracker, raw, rng, *, with_identity, recipe, step):
        raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        clean = tracker.aperture(raw)
        center = clean[..., :CENTER]
        coordinates = archive.coordinates
        framed, phase, _, rms = coordinates(tracker, archive.real_iq(center))
        clean_frame = phase * rms
        weak = archive.weak_difference(framed)
        delta = weak * clean_frame
        a, b = (tracker.aperture(archive.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
                for _ in range(2))
        if with_identity:
            a[-2:] = b[-2:] = clean[-2:]
        views = torch.stack((a, b, torch.cat((a[..., :CENTER] + delta, a[..., CENTER:]), dim=-1)), dim=1)
        observed = views.flatten(0, 1)
        y, phase, quality, rms = coordinates(tracker, archive.real_iq(observed[..., :CENTER]))
        frame = phase * rms
        target = torch.stack((center, center, center + delta), dim=1).flatten(0, 1) / frame
        time, full = model(y, quality, rms, observed[..., CENTER:])
        half = y + .5 * (full - y)
        waveform = archive.waveform_risk
        outer = waveform(half[:, (0, 7), :], target[:, (0, 7), :])
        inner = waveform(half[:, 1:-1], target[:, 1:-1])
        absolute = ((half - target).abs().square()).mean()
        full_risk = waveform(full, target)
        time_risk = waveform(y + .5 * (time - y), target)
        finished = (half * frame).unflatten(0, views.shape[:2])
        response = archive.split((finished[:, 2] - finished[:, 0]) / clean_frame)
        desired = archive.split(weak)
        transfer = (archive.transfer_risk(response[:, :1], desired[:, :1])
                    + archive.transfer_risk(response[:, 1:], desired[:, 1:]))
        consistency = ((finished[:, 0] - finished[:, 1]) / clean_frame).abs().square().mean()
        identity = y.real.new_zeros(())
        if with_identity:
            identity_edit = (half - y).unflatten(0, views.shape[:2])[-2:, :2]
            identity = identity_edit.abs().square().mean()
        parts = torch.stack((outer, inner, absolute, full_risk, time_risk, transfer, consistency, identity))
        weights = parts.new_tensor((.5, .5, .25, .18, .15, .30, .03, 1.))
        loss = parts @ weights
        if complex_fidelity is not None:
            # Only the two same-packet disturbed views contribute. Clean rows have their own identity loss.
            count = len(raw) - 2 if with_identity else len(raw)
            paired_half, paired_target, paired_y = (x.unflatten(0, views.shape[:2])[:count, :2]
                                                     for x in (half, target, y))
            direction, gain = correction_risk(paired_half - paired_y, paired_target - paired_y)
            neighbors = waveform(paired_half[..., 1:, :] - paired_half[..., :-1, :],
                                 paired_target[..., 1:, :] - paired_target[..., :-1, :])
            extra = torch.stack((direction, gain, neighbors))
            parts = torch.cat((parts, extra))
            if complex_fidelity:
                loss = loss + extra @ extra.new_tensor((.25, .10, .20)) + identity
        if coherence_priority is not None:
            count = len(raw) - 2 if with_identity else len(raw)
            restored = finished[:count, :2] / clean_frame[:count, None]
            clean_aligned = framed[:count, None].expand_as(restored)
            frequency, chirp = coherence_risks(restored, clean_aligned)
            conflict = y.real.new_zeros(())
            if torch.is_grad_enabled():
                loss, conflict = constrained_loss(loss, .5 * (frequency + chirp), full,
                                                  observe_only=not coherence_priority)
            parts = torch.cat((parts, torch.stack((frequency, chirp, conflict))))
        return loss, parts

    return calculate

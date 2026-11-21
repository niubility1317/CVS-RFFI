"""Fixed LoRa reference and inputs to the project's practical residual channel."""
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from experiments.cric_dura_comparison.io import tensor
from .data import BANDWIDTH_HZ, FC_HZ, FS_HZ, VIEWS, WINDOW_START


def rms_normalize(x):
    return x / x.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp_min(1e-14).sqrt()


class ChirpResponse(nn.Module):
    """Make the 29-feature PAMF response on public LoRa preamble excitation."""
    def __init__(self):
        super().__init__()
        t = torch.arange(256, dtype=torch.float64) / FS_HZ + WINDOW_START / 1e6
        chirp = torch.exp(2j * torch.pi * BANDWIDTH_HZ * (t.square() / (2 * .001024) - t / 2))
        self.register_buffer('chirp', chirp.to(torch.complex64), persistent=False)
        self.register_buffer('bins', torch.linspace(-56, 56, 12).round().long() % 256, persistent=False)

    def contract(self):
        return dict(reference='LoRa SF7 upchirp', bandwidth_hz=BANDWIDTH_HZ, sample_rate_hz=FS_HZ,
                    response_features=29, explicit_CFO_identity_input=False)

    def components(self, x):
        z = torch.complex(x[:, 0], x[:, 1])
        chirp = self.chirp.to(z.dtype)
        w = z * chirp.conj()
        # Integer dechirped peak alignment is per-packet and introduces no
        # fitted state or explicit carrier-offset coordinate.
        peak = torch.fft.fft(w)
        best = peak.abs().argmax(dim=-1)
        phase = -2j * torch.pi * best[:, None] * torch.arange(256, device=x.device) / 256
        correction = phase.exp().to(z.dtype)
        gain = (w * correction).mean(dim=-1)
        nrg = z.abs().square().mean(dim=-1).clamp_min(1e-12)
        quality = gain.abs().square() / nrg
        reference = torch.fft.fft(chirp)[self.bins]
        observed = torch.fft.fft(z * correction)[:, self.bins]
        aligned = observed * reference.conj() / reference.abs().square().clamp_min(1e-6)
        relative = aligned * gain.conj()[:, None] / gain.abs().square().clamp_min(1e-6)[:, None]
        residual = relative - 1
        bounded = residual / (1 + residual.abs().square() / 16).sqrt()
        c = (w[:, 1:] * w[:, :-1].conj()).sum(dim=-1).abs() / (255 * nrg)
        concentration = peak.abs().amax(dim=-1) / peak.abs().sum(dim=-1).clamp_min(1e-8)
        center_power = z[:, 64:192].abs().square().mean(dim=-1) / nrg
        features = torch.cat((bounded.real, bounded.imag,
            torch.stack((quality, c, concentration, torch.zeros_like(c), center_power), dim=1)), dim=1)
        return dict(features=features, quality=quality, relative_transfer=relative)

    def forward(self, x):
        return self.components(x)['features']

    @torch.no_grad()
    def diagnostics(self, x):
        return dict(template_correlation_mean=float(self.components(x)['quality'].sqrt().mean()))


@lru_cache(maxsize=None)
def channel_config(scene):
    from leo_practical.channel import Config
    return Config(fs_hz=FS_HZ, fc_hz=FC_HZ, scenario=scene, processing_route='residual',
                  mode='post_sync', equalization_enabled=False,
                  reference_snr_db=30., boundary='reflect')


@lru_cache(maxsize=None)
def receiver_hardware(scene, session):
    from leo_practical.channel import receiver_for_session
    return receiver_for_session(channel_config(scene), 20261016423, session)


def practical_iq(x, scene, receiver, packet, namespace):
    from leo_practical.channel import normalize_input, stable_seed
    from leo_practical.residual import ResidualChannel
    z, _ = normalize_input(x[0].astype(np.float64) + 1j * x[1])
    seed = stable_seed(2026112101, namespace, scene, receiver, str(packet))
    hw = receiver_hardware(scene, f'{namespace}/{receiver}')
    y, meta = ResidualChannel(channel_config(scene), seed, hw).process(z)
    if meta['channel_equalization_applied'] or not np.isfinite(y).all():
        raise ValueError('LoRa practical channel must be finite and no-EQ')
    return np.stack((y.real, y.imag)).astype(np.float32)


class PracticalSource:
    """At each labeled step, use a preregistered draw from genuine practical views."""
    def __init__(self, c):
        self.coverage = {}
        self.sessions = {}
        self.cache = np.load(c['source_practical'], mmap_mode='r', allow_pickle=False)
        with np.load(c['source_cache'] + '/L_s_metadata.npz', allow_pickle=False) as m:
            self.lookup = np.full(40_000, -1, dtype=np.int64)
            self.lookup[m['base_index']] = np.arange(len(m['base_index']))

    def __call__(self, x, meta, epoch, step):
        base = np.asarray(meta['base_index'].detach().cpu().tolist())
        positions = self.lookup[base]
        if np.any(positions < 0):
            raise ValueError('Practical supervised augmentation must resolve to source L')
        scenes = (base + epoch + step) % (len(VIEWS) - 1)
        for i in np.unique(scenes):
            name = VIEWS[int(i) + 1]
            self.coverage[name] = self.coverage.get(name, 0) + int((scenes == i).sum())
        return tensor(self.cache[scenes, positions], x.device)

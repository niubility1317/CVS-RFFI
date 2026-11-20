"""Original five-path PAMF graph with an Iridium common-word response."""

from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import represent, SYMBOLS


ORDER = (0, 5, 3, 6, 2, 1, 4, 7)
PASSBAND_END_HZ = 187_500


def _reference_iq(symbols, samples):
    # The downmixer returns the first UW symbol at sample zero. Smooth the
    # symbol changes at the half-symbol boundaries to reflect band limitation.
    n = np.arange(samples)
    signal = np.full(samples, symbols[0], dtype=np.complex64)
    for index, delta in enumerate(np.diff(symbols)):
        signal += delta * (0.5 + 0.5 * np.tanh((n - 500 - 1000 * index) / 140))
    return np.stack((signal.real, signal.imag), axis=-1)


def reference_waveform():
    return represent(_reference_iq(SYMBOLS, 8000)[None])[0]


def full_downlink_reference():
    # The last of the 12 fixed symbols is outside the crop. Its leading
    # transition lies inside, so include it when constructing the template.
    symbols = np.concatenate((SYMBOLS, np.array([-1 - 1j, 1 + 1j, 1 + 1j, -1 - 1j], dtype=SYMBOLS.dtype)))
    return torch.utils.dlpack.from_dlpack(_reference_iq(symbols, 11000).T[None])


def inband_gain(frequency_hz, gains):
    """Interpolate the symmetric physical response between fitted band centers."""
    position = (frequency_hz.abs() * (len(gains) / PASSBAND_END_HZ) - .5).clamp(0, len(gains) - 1)
    left = position.long()
    right = (left + 1).clamp_max(len(gains) - 1)
    return torch.lerp(gains[left], gains[right], position - left)


class UWResponse(nn.Module):
    def __init__(self, representation, waveform_samples=8000, grid_points=256, spectral_gains=None,
                 *, reference_grid=None):
        super().__init__()
        ratio = (waveform_samples / 8000) * (256 / grid_points)
        if reference_grid is None:
            template = torch.utils.dlpack.from_dlpack(reference_waveform()[representation])
            ref = torch.complex(template[0], template[1])
            if ratio != 1:
                if representation != 1:
                    raise ValueError("Reference resampling requires the downsampled SatIQ view")
                # Keep only grid locations supported by the known 8,000-point word.
                positions = torch.arange(int((len(ref) - 1) / ratio) + 1) * ratio
                left = positions.long()
                fraction = positions - left
                right = (left + 1).clamp_max(len(ref) - 1)
                ref = ref[left] * (1 - fraction) + ref[right] * fraction
        else:
            ref = torch.complex(reference_grid[0], reference_grid[1])
        ref = ref / ref.abs().square().mean().sqrt()
        window = torch.hann_window(len(ref))
        spectrum = torch.fft.fft(ref * window)
        # Twelve usable bins preserve the native response projection width.
        bins = spectrum.abs().topk(12).indices.sort().values
        # A supplied grid has already received the same shaping as input IQ.
        if spectral_gains is not None and reference_grid is None:
            period = waveform_samples / (25_000_000 * grid_points)
            frequency = torch.fft.fftfreq(len(ref), d=period, dtype=ref.real.dtype)
            ref = torch.fft.ifft(torch.fft.fft(ref) * inband_gain(frequency, spectral_gains))
            ref = ref / ref.abs().square().mean().sqrt()
            spectrum = torch.fft.fft(ref * window)
        self.register_buffer("reference", ref, persistent=False)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("ideal", spectrum[bins], persistent=False)
        self.coherence_lag = round(16 / ratio)

    def forward(self, iq):
        used = iq[:, :, :len(self.reference)]
        x = torch.complex(used[:, 0], used[:, 1])
        power = x.abs().square().mean(-1).clamp_min(1e-8)
        x = x / power.sqrt()[:, None]
        gain = (x * self.reference.conj()).mean(-1)
        spectrum = torch.fft.fft(x * self.window, dim=-1)[:, self.bins]
        floor = self.ideal.abs().square().amax() * 1e-4
        transfer = spectrum * self.ideal.conj() / (self.ideal.abs().square() + floor)
        relative = (transfer * gain.conj()[:, None] / (gain.abs().square()[:, None] + 1e-4)) - 1
        bounded = relative / (1 + relative.abs().square() / 16).sqrt()
        residual = x - gain[:, None] * self.reference
        lag = self.coherence_lag
        midpoint = x.shape[1] // 2
        coherence = (x[:, lag:] * x[:, :-lag].conj()).mean(-1).abs()
        tail_head = torch.log((x[:, midpoint:].abs().square().mean(-1) + 1e-4)
                              / (x[:, :midpoint].abs().square().mean(-1) + 1e-4))
        stats = torch.stack((gain.abs().square(), coherence, residual.abs().square().mean(-1),
                             torch.zeros_like(power), tail_head), dim=1)
        return torch.cat((bounded.real, bounded.imag, stats), dim=1)


def build_model(runtime, seed, representation, reference_waveform_samples=8000, grid_points=256,
                spectral_gains=None, chronological_sequence=False, *, reference_grid=None):
    from experiments.pamf_phase3_open_world.source_development import _activate_code

    _activate_code(runtime)
    from experiments.cric_dura_comparison.supervised_physics import shuffled_descriptor
    from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool, CompositeReference
    from experiments.cvs_reference_identity.model import ReferenceCVS

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ReferenceCVS("reference_response")
        core = model.core.id_backbone
        core.cls_head.weight = nn.Parameter(torch.empty(66, 160))
        nn.init.xavier_uniform_(core.cls_head.weight)
        sinc = CoordinateSinc(core.sinc, "CENTER_25K")
        if representation == 1:
            ratio = 800000 / 25000000
            with torch.no_grad():
                sinc.low_origin_hz.mul_(ratio)
                sinc.band_origin_hz.mul_(ratio)
                sinc.t_.div_(ratio)
            sinc.min_low_hz *= ratio
            sinc.min_band_hz *= ratio
            sinc.scale_hz *= ratio
            sinc.sample_rate = 800000
        if grid_points == 512:
            core.input_len = grid_points
            sinc.t_.mul_(.5)
            sinc.sample_rate *= 2
        core.sinc = sinc
        model.response = UWResponse(representation, reference_waveform_samples, grid_points, spectral_gains,
                                    reference_grid=reference_grid)
        model.sequence_pool = CompositePool(core.t2.pw.out_channels, "SIG_TIME_X3", seed)
        model.__class__ = CompositeReference
        order = torch.arange(len(ORDER)) if chronological_sequence else torch.tensor(ORDER)
        for view in model.sequence_pool.views:
            view.register_buffer("window_order", order.clone(), persistent=False)
            view.forward = MethodType(shuffled_descriptor, view)
        if grid_points == 512:
            # Retain the registered four windows and their eight-position order.
            model.sequence_pool.register_forward_pre_hook(lambda _module, args: (F.avg_pool1d(args[0], 2),))
    if sum(p.numel() for p in model.parameters()) != 242593 or core.cls_head.scale != 30:
        raise ValueError("Unexpected 66-class native PAMF graph")
    return model


def capture(model, x):
    core = model.core.id_backbone
    modules = dict(time=core.t_proj, f=core.f_proj, fs=core.freq_stats_proj,
                   pa=core.pa_proj, pas=core.pa_stats_proj, path=model.sequence_pool,
                   reference=model.response_projection)
    features = {}

    def register(name, module):
        def save(_module, _inputs, value):
            if name in features:
                raise ValueError(f"Repeated branch forward: {name}")
            features[name] = value

        return module.register_forward_hook(save)

    handles = [register(name, module) for name, module in modules.items()]
    try:
        fused = model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    if features.keys() != modules.keys():
        raise ValueError(f"Expected the deployed five PAMF paths, got {list(features)}")
    branches = (features["time"], features["f"] + features["fs"], features["pa"] + .25 * features["pas"],
                features["path"], features["reference"])
    return model.classify_features(fused), fused, torch.stack(branches)


def normalized(x):
    return x / x.square().sum(1, keepdim=True).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


def views(x, rng, rate, strength):
    phase = torch.rand((len(x), 1), generator=rng, device=x.device) * (2 * torch.pi)
    z = torch.complex(x[:, 0], x[:, 1])
    ordinary = z * torch.polar(torch.ones_like(phase), phase)
    noise_iq = torch.randn(x.shape, generator=rng, device=x.device)
    noise = torch.complex(noise_iq[:, 0], noise_iq[:, 1])
    if rate == 25000000:
        # The short window has only three nonzero central FFT bins at this rate.
        # Generate at the common 8000-sample observation scale, then crop.
        noise_iq = torch.randn((len(x), 2, 8000), generator=rng, device=x.device)
        noise = torch.complex(noise_iq[:, 0], noise_iq[:, 1])
    band = torch.fft.fftfreq(noise.shape[-1], 1 / rate, device=x.device).abs() <= 100000
    noise = torch.fft.ifft(torch.fft.fft(noise) * band)[:, :256]
    noise = noise / noise.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    snr = 5 + 10 * torch.rand((len(x), 1), generator=rng, device=x.device)
    perturbation = strength * noise * torch.pow(10, -snr / 20)
    physical = z + perturbation
    return (torch.view_as_real(ordinary).transpose(1, 2).contiguous(),
            torch.view_as_real(physical).transpose(1, 2).contiguous())

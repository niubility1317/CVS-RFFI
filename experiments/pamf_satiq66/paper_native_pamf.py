"""PAMF-SUP width variants using only the existing five common-word paths."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model, capture, inband_gain
from .paper_pamf import packet_rms
from .wide_model import FEATURES


NATIVE_PATHS = ("time", "frequency", "nonlinear", "sequence", "reference")
EXTRA_WIDTH = {"native160": 0, "fused256": 96, "fused320": 160, "branches320": 160, "prefuse320": 160}


def native_frequency_taper(bins):
    return torch.cos(torch.pi * ((bins - 60) / 20).clamp(0, 1)) * .5 + .5


def pa_highband_response(physical_hz, gain):
    transition = ((physical_hz - 80_000) / 40_000).clamp(0, 1)
    return 1 + (gain - 1) * .5 * (1 - torch.cos(torch.pi * transition))


def emphasize_pa_input(module, args):
    (iq,) = args
    spectrum = torch.fft.rfft(iq, dim=-1)
    return (torch.fft.irfft(spectrum * module.satiq_pa_response, n=iq.shape[-1], dim=-1),)


class PaperNativePAMF(nn.Module):
    waveform_samples = 8000

    def __init__(self, runtime, seed, classes, expansion, satiq_sinc_init=False, center_fusion=False,
                 waveform_samples=8000, align_reference_time=False, grid_points=256,
                 reflect_boundary=False, spectral_gains=None, chronological_sequence=False,
                 pa_highband_gain=1.0):
        super().__init__()
        if waveform_samples not in (8000, 11000):
            raise ValueError("Native PAMF expects the common word or full SatIQ header")
        if grid_points not in (256, 512):
            raise ValueError("Expected a 256- or 512-point native feature grid")
        if not 1 <= pa_highband_gain <= 1.5:
            raise ValueError("SatIQ PA highband amplitude gain must be between 1 and 1.5")
        if spectral_gains is not None:
            spectral_gains = torch.as_tensor(spectral_gains, dtype=torch.float32)
            if (spectral_gains.shape != (16,)
                    or not ((spectral_gains >= .7) & (spectral_gains <= 1.4)).all()):
                raise ValueError("Expected 16 finite spectral gains between 0.7 and 1.4")
            if not align_reference_time:
                raise ValueError("Spectral shaping requires the reference on the input time axis")
        self.waveform_samples = waveform_samples
        self.grid_points = grid_points
        self.reflect_boundary = reflect_boundary
        extension = 2 if reflect_boundary else 1
        self.fft_points = grid_points * extension
        self.expansion = expansion
        self.feature_dimensions = FEATURES + EXTRA_WIDTH[expansion]
        reference_samples = waveform_samples if align_reference_time else 8000
        self.native = build_model(runtime, seed, 1, reference_waveform_samples=reference_samples,
                                  grid_points=grid_points, spectral_gains=spectral_gains,
                                  chronological_sequence=chronological_sequence)
        if pa_highband_gain > 1:
            core = self.native.core.id_backbone
            if not core.use_pa_path or core.pa_feature_source != "raw_iq":
                raise ValueError("PA highband emphasis requires the direct IQ polynomial path")
            period = waveform_samples / (25_000_000 * grid_points)
            frequency = torch.fft.rfftfreq(grid_points, d=period)
            response = pa_highband_response(frequency, pa_highband_gain)
            core.pa_lift.register_buffer("satiq_pa_response", response, persistent=False)
            # Preserve the gate's original IQ side input and the parallel spectral statistics.
            core.pa_lift.register_forward_pre_hook(emphasize_pa_input)
        self.native.core.id_backbone.cls_head.requires_grad_(False)
        if satiq_sinc_init:
            sinc = self.native.core.id_backbone.sinc
            # At 800 kS/s, use roughly two 79-tap resolution bins per band.
            # Overlapping 20 kHz bands cover the occupied baseband and its shoulders.
            low_hz = torch.arange(sinc.out_channels, dtype=sinc.low_origin_hz.dtype) * 8000
            sinc.low_origin_hz.copy_((low_hz[:, None] - sinc.min_low_hz).clamp_min(0))
            sinc.band_origin_hz.fill_(20_000 - sinc.min_band_hz)
        self.expander = None
        if expansion == "branches320":
            self.expander = nn.ModuleList(
                nn.Sequential(nn.Linear(FEATURES, 32), nn.SiLU(), nn.LayerNorm(32))
                for _ in NATIVE_PATHS
            )
        elif EXTRA_WIDTH[expansion]:
            # Reuse the existing fusion input to test the 160-coordinate bottleneck.
            input_width = FEATURES
            hidden_width = 256
            if expansion == "prefuse320":
                input_width += self.native.core.id_backbone.fuse[0].in_features
                hidden_width = 128
            self.expander = nn.Sequential(
                nn.Linear(input_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, EXTRA_WIDTH[expansion]),
                nn.LayerNorm(EXTRA_WIDTH[expansion]),
            )
        self.head = nn.Linear(self.feature_dimensions, classes, bias=False)
        nn.init.xavier_uniform_(self.head.weight)
        self.register_buffer("fusion_center", torch.zeros(self.feature_dimensions) if center_fusion else None)
        bins = torch.arange(self.fft_points // 2 + 1, dtype=torch.float32) * (8000 / (self.waveform_samples * extension))
        cutoff = native_frequency_taper(bins)
        if spectral_gains is not None:
            cutoff = cutoff * inband_gain(bins * (25_000_000 / 8000), spectral_gains)
        self.register_buffer("cutoff", cutoff, persistent=False)

    def _center_fusion(self, feature):
        if self.fusion_center is None:
            return feature
        if self.training:
            self.fusion_center.lerp_(feature.detach().mean(dim=0), .05)
        return F.normalize(feature - self.fusion_center, dim=1, eps=1e-6)

    def _encode(self, iq, retain_branches):
        if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] < self.waveform_samples:
            raise ValueError("Native PAMF requires both IQ channels of its configured header window")
        common = packet_rms(iq[:, :, :self.waveform_samples])
        if self.reflect_boundary:
            # The even extension keeps the burst tail from wrapping into the synchronization field.
            common = torch.cat((common, common.flip(-1)), dim=-1)
        spectrum = torch.fft.rfft(common, dim=-1)[..., :self.fft_points // 2 + 1]
        coarse = torch.fft.irfft(spectrum * self.cutoff, n=self.fft_points, dim=-1)
        coarse = coarse[..., :self.grid_points] * (self.fft_points / common.shape[-1])
        branches = None
        fusion_inputs = []
        handle = None
        if self.expansion == "prefuse320":
            handle = self.native.core.id_backbone.fuse.register_forward_pre_hook(
                lambda _module, args: fusion_inputs.append(args[0]))
        try:
            if retain_branches or self.expansion == "branches320":
                _, native, branches = capture(self.native, coarse)
            else:
                native = self.native.features(coarse)
        finally:
            if handle is not None:
                handle.remove()
        native = F.normalize(native, dim=1)
        if self.expander is None:
            return self._center_fusion(native), branches
        if self.expansion == "branches320":
            parts = [layer(F.normalize(x, dim=1)) for layer, x in zip(self.expander, branches, strict=True)]
            extra = torch.cat(parts, dim=1)
        elif self.expansion == "prefuse320":
            (fuse_input,) = fusion_inputs
            extra = self.expander(torch.cat((native, F.normalize(fuse_input, dim=1)), dim=1))
        else:
            extra = self.expander(native)
        feature = F.normalize(torch.cat((native, .5 * F.normalize(extra, dim=1)), dim=1), dim=1)
        return self._center_fusion(feature), branches

    def embed(self, iq):
        return self._encode(iq, False)[0]

    def encode_paths(self, iq):
        return (self.embed(iq),)

    def forward(self, iq, view_logits=False):
        feature, branches = self._encode(iq, view_logits)
        weight = F.normalize(self.head.weight, dim=1)
        logits = 30 * F.linear(feature, weight)
        if view_logits:
            branch_weight = F.normalize(self.head.weight[:, :FEATURES], dim=1)
            branch_logits = 30 * F.linear(F.normalize(branches, dim=-1), branch_weight)
            return feature, logits, branch_logits, branches
        return feature, logits


def native_classification_terms(logits, branch_logits, target, satellites):
    ce = F.cross_entropy(logits, target)
    same = satellites[None, :] == satellites[target, None]
    within = F.cross_entropy(logits.masked_fill(~same, -torch.inf), target)
    branch_ce = F.cross_entropy(branch_logits.flatten(0, 1), target.repeat(len(branch_logits)),
                                 reduction="none").reshape(len(branch_logits), -1).mean(dim=1)
    teacher = logits.detach().softmax(dim=1).expand_as(branch_logits)
    agreement = F.kl_div(branch_logits.log_softmax(dim=-1), teacher, reduction="sum")
    agreement = agreement / (len(branch_logits) * len(target))
    return ce, within, branch_ce, agreement


def native_supervised_loss(logits, branch_logits, target, satellites, weights):
    ce, within, branch_ce, agreement = native_classification_terms(logits, branch_logits, target, satellites)
    loss = (ce + weights["same_satellite_ce"] * within + weights["view_ce"] * branch_ce.mean()
            + weights["view_kl"] * agreement)
    stats = dict(ce=ce.detach(), within_satellite_ce=within.detach(),
                 native_branch_ce=branch_ce.mean().detach(), view_kl=agreement.detach())
    satellite_weight = weights.get("satellite_marginal_ce", 0.0)
    if satellite_weight:
        # Full beam CE decomposes into satellite mass and conditional beam CE.
        satellite_ce = ce - within
        loss = loss + satellite_weight * satellite_ce
        stats["satellite_marginal_ce"] = satellite_ce.detach()
    hard_fraction = weights.get("hard_rank_ce_fraction", 0.0)
    if hard_fraction:
        positive = logits.gather(1, target[:, None])
        negatives = logits.scatter(1, target[:, None], -torch.inf)
        hardest = negatives.topk(min(8, logits.shape[1] - 1), dim=1).values
        hard_ce = F.cross_entropy(torch.cat((positive, hardest), dim=1), torch.zeros_like(target))
        loss = loss + hard_fraction * (hard_ce - ce)
        stats["hard_rank_ce"] = hard_ce.detach()
    for name, value in zip(NATIVE_PATHS, branch_ce.detach(), strict=True):
        stats[f"{name}_branch_ce"] = value
    return loss, stats
